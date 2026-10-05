#include <ATen/DLConvertor.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/extension.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <array>
#include <list>
#include <map>
#include <mutex>

namespace py = pybind11;
using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

namespace {
using Tensors = std::array<Tensor, 14>;
using Key = std::vector<int64_t>;
struct Entry {
    tvm::ffi::Function function;
    std::list<Key>::iterator position;
};
std::map<Key, Entry> programs;
std::list<Key> recent;
std::mutex cache_mutex;

Key configuration_key(bool backward, const Tensors &tensors, bool causal, bool masked) {
    Key key = {backward, tensors[0].get_device(), causal, masked};
    key.reserve(160);
    for (const auto &tensor : tensors) {
        key.push_back(static_cast<int64_t>(tensor.scalar_type()));
        key.push_back(tensor.dim());
        auto pointer = reinterpret_cast<uintptr_t>(tensor.data_ptr());
        key.push_back(std::min<uintptr_t>(16, pointer & -pointer));
        key.insert(key.end(), tensor.sizes().begin(), tensor.sizes().end());
        key.insert(key.end(), tensor.strides().begin(), tensor.strides().end());
    }
    return key;
}

tvm::ffi::Function program(bool backward, const Tensors &tensors, bool causal, bool masked) {
    auto key = configuration_key(backward, tensors, causal, masked);
    std::unique_lock<std::mutex> lock(cache_mutex);
    auto found = programs.find(key);
    if (found != programs.end()) {
        recent.splice(recent.begin(), recent, found->second.position);
        return found->second.function;
    }
    lock.unlock();
    py::gil_scoped_acquire gil;
    lock.lock();
    found = programs.find(key);
    if (found != programs.end()) {
        recent.splice(recent.begin(), recent, found->second.position);
        return found->second.function;
    }
    py::tuple inputs(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i)
        inputs[i] = py::cast(tensors[i]);
    auto name = py::module_::import("simpletuner.helpers.training.kohaku_fa_cute.native")
                    .attr("compile_kernel")(backward ? "backward" : "forward", inputs, causal, masked)
                    .cast<std::string>();
    auto function = tvm::ffi::Function::GetGlobalRequired(name);
    tvm::ffi::Function::RemoveGlobal(name);
    recent.push_front(key);
    programs.emplace(std::move(key), Entry{function, recent.begin()});
    if (programs.size() > 128) {
        programs.erase(recent.back());
        recent.pop_back();
    }
    return function;
}

class StreamScope {
    int device;
    TVMFFIStreamHandle previous;

  public:
    explicit StreamScope(int index) : device(index) {
        TVM_FFI_CHECK_SAFE_CALL(TVMFFIEnvSetStream(
            kDLCUDA, device, reinterpret_cast<void *>(c10::cuda::getCurrentCUDAStream(device).stream()), &previous));
    }
    ~StreamScope() { TVMFFIEnvSetStream(kDLCUDA, device, previous, nullptr); }
};

void launch(bool backward, const Tensors &tensors, double scale, bool causal, bool masked) {
    auto function = program(backward, tensors, causal, masked);
    std::array<DLTensor, 14> dl;
    std::array<tvm::ffi::AnyView, 15> arguments;
    for (size_t i = 0; i < tensors.size(); ++i) {
        at::toDLPackNonOwning(tensors[i], &dl[i]);
        arguments[i] = &dl[i];
    }
    arguments[14] = scale;
    StreamScope stream(tensors[0].get_device());
    tvm::ffi::Any result;
    function.CallPacked(arguments.data(), arguments.size(), &result);
}

variable_list forward(Tensor q, Tensor k, Tensor v, Tensor mask, bool masked, double scale, bool causal) {
    c10::cuda::CUDAGuard guard(q.device());
    auto out = at::empty_like(q);
    std::vector<int64_t> stats(q.sizes().begin(), q.sizes().end() - 1);
    auto maximum = at::empty(stats, q.options().dtype(at::kFloat));
    auto logsum = at::empty_like(maximum);
    std::vector<int64_t> shape = {1, 1, 1, 1};
    if (masked) {
        shape = {mask.stride(0) == 0 ? 1 : q.size(0), mask.stride(1) == 0 ? 1 : q.size(1),
                 mask.stride(2) == 0 ? 1 : (q.size(2) + 63) / 64, mask.stride(3) == 0 ? 1 : (k.size(2) + 63) / 64};
    }
    auto blocks = at::empty(shape, q.options().dtype(at::kInt));
    launch(false, {q, k, v, mask, out, maximum, logsum, out, maximum, out, out, out, blocks, maximum}, scale, causal,
           masked);
    return {out, maximum, logsum, blocks};
}

variable_list backward(Tensor q, Tensor k, Tensor v, Tensor mask, bool masked, Tensor maximum, Tensor logsum, Tensor dout,
                       Tensor blocks, const std::optional<Tensor> &out, double scale, bool causal) {
    c10::cuda::CUDAGuard guard(q.device());
    auto dq = at::empty_like(q), dk = at::empty_like(k), dv = at::empty_like(v);
    auto delta = at::empty_like(maximum);
    auto base = out ? at::empty_like(maximum) : maximum;
    if (masked)
        blocks = blocks.expand({q.size(0), q.size(1), (q.size(2) + 63) / 64, (k.size(2) + 63) / 64});
    launch(true, {q, k, v, mask, out.value_or(dq), maximum, logsum, dout, delta, dq, dk, dv, blocks, base}, scale, causal,
           masked);
    return {dq, dk, dv};
}

class Attention : public torch::autograd::Function<Attention> {
  public:
    static Tensor forward(AutogradContext *ctx, Tensor q, Tensor k, Tensor v, Tensor mask, bool masked, double scale,
                          bool causal, bool centered) {
        auto output = ::forward(q, k, v, mask, masked, scale, causal);
        variable_list saved = {q, k, v, output[1], output[2], output[3]};
        if (masked)
            saved.push_back(mask);
        if (centered)
            saved.push_back(output[0]);
        ctx->save_for_backward(saved);
        ctx->saved_data["masked"] = masked;
        ctx->saved_data["centered"] = centered;
        ctx->saved_data["scale"] = scale;
        ctx->saved_data["causal"] = causal;
        return output[0];
    }
    static variable_list backward(AutogradContext *ctx, variable_list gradients) {
        auto saved = ctx->get_saved_variables();
        bool masked = ctx->saved_data["masked"].toBool();
        bool centered = ctx->saved_data["centered"].toBool();
        auto output = ::backward(saved[0], saved[1], saved[2], masked ? saved[6] : saved[0], masked, saved[3], saved[4],
                                 gradients[0], saved[5], centered ? std::optional<Tensor>(saved.back()) : std::nullopt,
                                 ctx->saved_data["scale"].toDouble(), ctx->saved_data["causal"].toBool());
        return {output[0], output[1], output[2], Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
    }
};
std::pair<int, int> device_capability(int device) {
    static std::map<int, std::pair<int, int>> capabilities;
    static std::mutex capability_mutex;
    std::lock_guard<std::mutex> lock(capability_mutex);
    auto found = capabilities.find(device);
    if (found != capabilities.end())
        return found->second;
    int major, minor;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    return capabilities.emplace(device, std::make_pair(major, minor)).first->second;
}

bool supported_device(int device) {
    auto [major, minor] = device_capability(device);
    return (major == 8 && minor == 9) || (major == 9 && minor == 0) || (major == 12 && minor == 0);
}

bool compatible(Tensor q, Tensor k, Tensor v, const std::optional<Tensor> &mask, double dropout, bool causal, bool gqa) {
    return dropout == 0.0 && q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && q.is_cuda() && q.device() == k.device() &&
           q.device() == v.device() && (q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16) &&
           q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type() && q.numel() > 0 && k.numel() > 0 &&
           k.sizes() == v.sizes() && q.size(0) == k.size(0) && q.size(3) == k.size(3) && q.size(3) <= 512 &&
           q.size(1) % k.size(1) == 0 && (gqa || q.size(1) == k.size(1)) &&
           (!mask || (mask->scalar_type() == at::kBool && mask->device() == q.device() && !causal)) &&
           supported_device(q.get_device()) && device_capability(q.get_device()).first != 12;
}

bool validate(Tensor q, Tensor k, Tensor v, const std::optional<Tensor> &mask) {
    TORCH_CHECK_VALUE(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "kohaku-fa requires 4D Q/K/V.");
    TORCH_CHECK_VALUE(q.is_cuda() && k.device() == q.device() && v.device() == q.device(),
                      "kohaku-fa requires Q/K/V on the same CUDA device.");
    TORCH_CHECK_VALUE((q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16) &&
                          k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
                      "kohaku-fa requires matching FP16/BF16 Q/K/V; FP32 attention is unsupported.");
    TORCH_CHECK_VALUE(q.numel() > 0 && k.numel() > 0 && k.sizes() == v.sizes() && q.size(0) == k.size(0) &&
                          q.size(3) == k.size(3),
                      "kohaku-fa requires nonempty matching batch/head dimensions and equal K/V shapes.");
    TORCH_CHECK_VALUE(q.size(1) % k.size(1) == 0 && q.size(3) <= 512,
                      "kohaku-fa requires head dimensions 1..512 and query heads divisible by K/V heads.");
    if (mask) {
        TORCH_CHECK_VALUE(mask->scalar_type() == at::kBool,
                          "kohaku-fa supports boolean attention masks only, not additive masks.");
        TORCH_CHECK_VALUE(mask->device() == q.device(), "kohaku-fa requires the attention mask on the Q/K/V device.");
    }
    TORCH_CHECK_VALUE(supported_device(q.get_device()), "CuTe attention requires sm_89, sm_90 or sm_120.");
    return device_capability(q.get_device()).first != 12;
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("automatic_sdpa", [](Tensor q, Tensor k, Tensor v, std::optional<Tensor> mask, double dropout, bool causal,
                                    std::optional<double> scale, bool gqa) {
        py::gil_scoped_release release;
        if (!compatible(q, k, v, mask, dropout, causal, gqa))
            return at::scaled_dot_product_attention(q, k, v, mask, dropout, causal, scale, gqa);
        if (mask)
            mask = mask->expand({q.size(0), q.size(1), q.size(2), k.size(2)});
        return Attention::apply(q, k, v, mask.value_or(q), mask.has_value(), scale.value_or(std::pow(q.size(3), -0.5)),
                                causal, true);
    });
    module.def("sdpa", [](Tensor q, Tensor k, Tensor v, std::optional<Tensor> mask, double dropout, bool causal,
                          std::optional<double> scale, bool gqa) {
        py::gil_scoped_release release;
        TORCH_CHECK_VALUE(dropout == 0.0, "kohaku-fa does not support attention dropout.");
        bool centered = validate(q, k, v, mask);
        TORCH_CHECK_VALUE(gqa || q.size(1) == k.size(1),
                          "kohaku-fa requires enable_gqa=True for differing Q and K/V head counts.");
        TORCH_CHECK_VALUE(!mask || !causal, "kohaku-fa requires causal visibility folded into the boolean mask.");
        if (mask)
            mask = mask->expand({q.size(0), q.size(1), q.size(2), k.size(2)});
        return Attention::apply(q, k, v, mask.value_or(q), mask.has_value(), scale.value_or(std::pow(q.size(3), -0.5)),
                                causal, centered);
    });
    module.def("attention",
               [](Tensor q, Tensor k, Tensor v, std::optional<Tensor> mask, std::optional<double> scale, bool causal) {
                   py::gil_scoped_release release;
                   bool centered = validate(q, k, v, mask);
                   if (mask)
                       mask = mask->expand({q.size(0), q.size(1), q.size(2), k.size(2)});
                   return Attention::apply(q, k, v, mask.value_or(q), mask.has_value(),
                                           scale.value_or(std::pow(q.size(3), -0.5)), causal, centered);
               });
    module.def("forward", [](Tensor q, Tensor k, Tensor v, std::optional<Tensor> mask, double scale, bool causal) {
        py::gil_scoped_release release;
        return forward(q, k, v, mask.value_or(q), mask.has_value(), scale, causal);
    });
    module.def("backward", [](Tensor q, Tensor k, Tensor v, std::optional<Tensor> mask, Tensor maximum, Tensor logsum,
                              Tensor dout, Tensor blocks, std::optional<Tensor> out, double scale, bool causal) {
        py::gil_scoped_release release;
        return backward(q, k, v, mask.value_or(q), mask.has_value(), maximum, logsum, dout, blocks, out, scale, causal);
    });
    module.def("cache_size", []() {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(cache_mutex);
        return programs.size();
    });
    module.def("clear_cache", []() {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(cache_mutex);
        programs.clear();
        recent.clear();
    });
}
