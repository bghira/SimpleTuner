#include <ATen/Context.h>
#include <ATen/DLConvertor.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/core/GradMode.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <array>
#include <cmath>
#include <cstring>
#include <list>
#include <limits>
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

// The configuration is constructed once, then retained as immutable CPU data
// through autograd. Hot calls never rebuild Python dictionaries or DLPack views.
constexpr std::array<const char *, 17> configured_fields = {
    "forward_m", "forward_n", "forward_stages", "query_in_registers", "backward_m", "backward_n",
    "query_stages", "dout_stages", "threads", "score_m_warps", "kv_m_warps", "dq_m_warps", "v_in_regs",
    "forward_threads", "maximum_multiple", "accumulator_head_dim", "causal_head_group"};
constexpr std::array<int64_t, 17> configured_defaults = {
    128, 64, 1, 0, 64, 128, 2, 2, 256, 2, 2, 2, 0, 128, 1, 0, 1};
constexpr int configuration_backward_m = 5;
constexpr int configuration_maximum_multiple = 15;
constexpr int configuration_accumulator_head_dim = 16;

std::pair<int, int> device_capability(int device);
bool full_a40(int device);
bool full_rtx5090(int device);
bool full_gb10(int device);
bool full_a100(int device);

int split_forward_count(const Tensor &q, const Tensor &k, bool masked, bool causal) {
    // Short schema queries need parallelism along the memory sequence.
    if (masked || causal || q.size(0) != 1 || q.size(1) != 16 || k.size(1) != 16 || q.size(3) != 64 ||
        q.size(2) > 256 || k.size(2) < 256 || k.size(2) > 16384)
        return 0;
    const auto capability = device_capability(q.get_device());
    if (capability != std::make_pair(8, 9) && capability != std::make_pair(9, 0))
        return 0;
    return q.size(2) > 64 || k.size(2) <= 256 ? 4 : 16;
}

Key configuration_key(bool backward, const Tensors &tensors, bool causal, bool masked, bool sm80 = false,
                      const Tensor &configuration = Tensor(), bool inference = false) {
    const auto family = configuration.defined() ? configuration.const_data_ptr<int64_t>()[0] : 0;
    Key key = {backward, tensors[0].get_device(), causal, masked, sm80, family};
    // Fourteen rank-four aliases plus the immutable configuration and mode
    // need 165 integers. Keep the default reservation unchanged.
    key.reserve(inference ? 176 : 160);
    if (family == 1)
        key.push_back(inference);
    if (sm80 || configuration.defined()) {
        if (configuration.defined()) {
            const auto *values = configuration.const_data_ptr<int64_t>();
            key.insert(key.end(), values, values + configuration.numel());
        }
        // Cache exact runtime layouts for the optional entry. Do not reuse
        // native dispatch's shape/stride equivalence classes.
        for (const auto &tensor : tensors) {
            key.push_back(static_cast<int64_t>(tensor.scalar_type()));
            key.push_back(tensor.dim());
            for (auto size : tensor.sizes())
                key.push_back(size);
            for (auto stride : tensor.strides())
                key.push_back(stride);
        }
        return key;
    }
    if (!backward && tensors[7].dim() == 5) {
        key.push_back(tensors[0].size(2) > 32);
        key.push_back(tensors[0].size(2) > 64);
    }
    // These thresholds select different Hopper forward programs.
    if (!backward && tensors[0].size(3) == 128) {
        key.push_back(tensors[0].numel() / tensors[0].size(3) >= 32768);
        key.push_back(tensors[0].size(2) >= 1536 && tensors[1].size(2) >= 1024);
    }
    // Blackwell producer/consumer dispatch has a measured short-sequence fallback.
    // Keep this class in both cache keys: its forward and backward programs differ.
    const auto dimension = tensors[0].size(3);
    if (dimension == 64 || dimension == 128 || dimension == 256 || dimension == 512) {
        const auto threshold = dimension == 64 ? 1024 : dimension == 128 ? 512 : 256;
        key.push_back(std::max(tensors[0].size(2), tensors[1].size(2)) >= threshold);
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto &tensor = tensors[i];
        key.push_back(static_cast<int64_t>(tensor.scalar_type()));
        key.push_back(tensor.dim());
        auto pointer = reinterpret_cast<uintptr_t>(tensor.data_ptr());
        key.push_back(std::min<uintptr_t>(16, pointer & -pointer));
        for (int axis = 0; axis < tensor.dim(); ++axis) {
            bool sequence = axis == 2 || (axis == 3 && (i == 12 || (i == 3 && masked)));
            key.push_back(sequence && tensor.size(axis) != 1 ? -1 : tensor.size(axis));
        }
        bool half = tensor.scalar_type() == at::kHalf || tensor.scalar_type() == at::kBFloat16;
        for (auto stride : tensor.strides()) {
            auto divisor = half ? std::min<int64_t>(8, stride & -stride) : 1;
            key.push_back(stride <= 1 ? stride : -divisor);
        }
    }
    return key;
}

tvm::ffi::Function program(bool backward, const Tensors &tensors, bool causal, bool masked, bool sm80 = false,
                          const Tensor &configuration = Tensor(), bool inference = false) {
    auto key = configuration_key(backward, tensors, causal, masked, sm80, configuration, inference);
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
    const char *mode = sm80 ? (backward ? "sm80_backward" : "sm80_forward") : (backward ? "backward" : "forward");
    auto compiler = py::module_::import("simpletuner.helpers.training.kohaku_fa_cute.native").attr("compile_kernel");
    std::string name;
    if (configuration.defined()) {
        const auto *values = configuration.const_data_ptr<int64_t>();
        py::dict parameters;
        for (size_t i = 0; i < configured_fields.size(); ++i) {
            if (values[0] == 2 && i == 16)
                continue;
            parameters[configured_fields[i]] = values[i + 1];
        }
        const char *configured_mode = values[0] == 13 ? (backward ? "short_d128_stats_backward" : "short_d128_stats_forward")
                                      : values[0] == 12 ? (backward ? "short_d128_backward" : "short_d128_forward")
                                      : values[0] == 9 ? (backward ? "short_key32_backward" : "short_key32_forward")
                                      : values[0] == 4 ? (backward ? "shortopt_backward" : "shortopt_forward")
                                      : values[0] == 3 ? (backward ? "short_backward" : "short_forward")
                                      : values[0] == 1 ? (inference ? "warp_inference" : (backward ? "warp_backward" : "warp_forward"))
                                                    : (backward ? "hopper_backward" : "hopper_forward");
        name = compiler(configured_mode, inputs, causal, masked, parameters).cast<std::string>();
    } else {
        name = compiler(mode, inputs, causal, masked).cast<std::string>();
    }
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

void launch(bool backward, const Tensors &tensors, double scale, bool causal, bool masked, bool sm80 = false,
            const Tensor &configuration = Tensor(), bool inference = false) {
    const auto family = configuration.defined() ? configuration.const_data_ptr<int64_t>()[0] : 0;
    if (backward && (sm80 || family == 1 || family == 2)) {
        // See Note [Writing Nondeterministic Operations]. The shared warp and
        // Hopper backward programs accumulate dQ with unordered FP32 atomics.
        // Check every invocation, including cached programs and policy changes
        // after forward. Owned-gradient families do not use this reduction.
        at::globalContext().alertNotDeterministic("simpletuner_cute_attention::atomic_backward");
    }
    auto function = program(backward, tensors, causal, masked, sm80, configuration, inference);
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

void canonicalize_input(Tensor &tensor) {
    // DLPack represents strides/storage, but cannot carry lazy value bits.
    // Clone the logical values once, before constructing descriptor aliases.
    if (tensor.is_neg() || tensor.is_conj())
        tensor = tensor.clone(at::MemoryFormat::Preserve);
}

void canonicalize_qkv(Tensor &q, Tensor &k, Tensor &v) {
    canonicalize_input(q);
    canonicalize_input(k);
    canonicalize_input(v);
}

Tensor contiguous_gradient(Tensor gradient) {
    if (gradient.is_neg() || gradient.is_conj() || !gradient.is_contiguous() ||
        reinterpret_cast<uintptr_t>(gradient.data_ptr()) % 16 != 0)
        return gradient.clone(at::MemoryFormat::Contiguous);
    return gradient;
}

variable_list forward_canonical(Tensor q, Tensor k, Tensor v, Tensor mask, bool masked, double scale, bool causal) {
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
    auto partial = out, split_stats = maximum;
    const auto splits = split_forward_count(q, k, masked, causal);
    if (splits) {
        partial = at::empty({q.size(0), q.size(1), q.size(2), splits, q.size(3)}, q.options().dtype(at::kFloat));
        split_stats = at::empty({q.size(0), q.size(1), q.size(2), splits, 2}, q.options().dtype(at::kFloat));
    }
    launch(false, {q, k, v, mask, out, maximum, logsum, partial, split_stats, out, out, out, blocks, maximum}, scale, causal,
           masked);
    return {out, maximum, logsum, blocks};
}

variable_list forward(Tensor q, Tensor k, Tensor v, Tensor mask, bool masked, double scale, bool causal) {
    c10::cuda::CUDAGuard guard(q.device());
    canonicalize_qkv(q, k, v);
    return forward_canonical(q, k, v, masked ? mask : q, masked, scale, causal);
}

bool packed_statistics_workspace(const Tensor &q, const Tensor &k, bool masked, bool causal) {
    // Match the complete-tile guard inside the cached RTX 4090 launch program.
    if (!causal || masked || q.size(0) != 1 || q.size(1) != 16 || k.size(1) != 16 || q.size(3) != 128 ||
        q.size(2) < 1024 || k.size(2) < 1024 || q.size(2) % 64 != 0 || k.size(2) % 64 != 0)
        return false;
    static std::map<int, bool> eligible_devices;
    static std::mutex eligibility_mutex;
    std::lock_guard<std::mutex> lock(eligibility_mutex);
    const int device = q.get_device();
    auto found = eligible_devices.find(device);
    if (found != eligible_devices.end())
        return found->second;
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    const bool eligible = properties.major == 8 && properties.minor == 9 &&
                          std::strcmp(properties.name, "NVIDIA GeForce RTX 4090") == 0;
    return eligible_devices.emplace(device, eligible).first->second;
}

variable_list backward(Tensor q, Tensor k, Tensor v, Tensor mask, bool masked, Tensor maximum, Tensor logsum, Tensor dout,
                       Tensor blocks, const std::optional<Tensor> &out, double scale, bool causal) {
    c10::cuda::CUDAGuard guard(q.device());
    canonicalize_qkv(q, k, v);
    canonicalize_input(dout);
    if (!masked)
        mask = q;
    auto dq = at::empty_like(q), dk = at::empty_like(k), dv = at::empty_like(v);
    auto delta = packed_statistics_workspace(q, k, masked, causal)
                     ? at::empty({maximum.size(0), maximum.size(1), 4 * maximum.size(2)}, maximum.options())
                     : at::empty_like(maximum);
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
        c10::cuda::CUDAGuard guard(q.device());
        canonicalize_qkv(q, k, v);
        auto output = forward_canonical(q, k, v, masked ? mask : q, masked, scale, causal);
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

class Sm80Attention : public torch::autograd::Function<Sm80Attention> {
  public:
    static Tensor forward(AutogradContext *ctx, Tensor q, Tensor k, Tensor v, double scale, bool causal) {
        c10::cuda::CUDAGuard guard(q.device());
        canonicalize_qkv(q, k, v);
        auto out = at::empty_like(q);
        auto maximum = at::empty({q.size(0), q.size(1), q.size(2)}, q.options().dtype(at::kFloat));
        auto logsum = at::empty_like(maximum);
        launch(false, {q, k, v, q, out, maximum, logsum, out, maximum, out, out, out, maximum, maximum},
               scale, causal, false, true);
        ctx->save_for_backward({q, k, v, out, maximum, logsum});
        ctx->saved_data["scale"] = scale;
        ctx->saved_data["causal"] = causal;
        return out;
    }

    static variable_list backward(AutogradContext *ctx, variable_list gradients) {
        auto saved = ctx->get_saved_variables();
        const auto &q = saved[0];
        c10::cuda::CUDAGuard guard(q.device());
        auto dout = contiguous_gradient(gradients[0]);
        auto dq = at::empty_like(q), dk = at::empty_like(saved[1]), dv = at::empty_like(saved[2]);
        constexpr int64_t tile_m = 64;
        const auto rounded = (q.size(2) + tile_m - 1) / tile_m * tile_m;
        auto delta = at::empty({q.size(0), q.size(1), rounded}, q.options().dtype(at::kFloat));
        auto lognorm = at::empty_like(delta);
        auto scratch = at::empty({q.size(0), q.size(1), rounded * q.size(3)}, delta.options());
        // Preparation initializes the complete scratch on each invocation,
        // including graph replay. No tensor pointer is cached in a program.
        launch(true, {q, saved[1], saved[2], q, saved[3], saved[4], saved[5], dout, delta, dq, dk, dv, scratch, lognorm},
               ctx->saved_data["scale"].toDouble(), ctx->saved_data["causal"].toBool(), false, true);
        return {dq, dk, dv, Tensor(), Tensor()};
    }
};

class ConfiguredAttention : public torch::autograd::Function<ConfiguredAttention> {
  public:
    static Tensor forward(AutogradContext *ctx, Tensor q, Tensor k, Tensor v, double scale, bool causal,
                          Tensor configuration) {
        c10::cuda::CUDAGuard guard(q.device());
        canonicalize_qkv(q, k, v);
        const auto *parameters = configuration.const_data_ptr<int64_t>();
        const auto multiple = parameters[configuration_maximum_multiple];
        const auto padded = (q.size(2) + multiple - 1) / multiple * multiple;
        auto out = at::empty_like(q);
        auto maximum = at::empty({q.size(0), q.size(1), padded}, q.options().dtype(at::kFloat));
        if (padded != q.size(2))
            maximum.zero_();
        auto logsum = at::empty({q.size(0), q.size(1), q.size(2)}, maximum.options());
        launch(false, {q, k, v, q, out, maximum, logsum, out, maximum, out, out, out, maximum, maximum},
               scale, causal, false, false, configuration);
        ctx->save_for_backward({q, k, v, out, maximum, logsum});
        ctx->saved_data["configuration"] = configuration;
        ctx->saved_data["scale"] = scale;
        ctx->saved_data["causal"] = causal;
        return out;
    }

    static variable_list backward(AutogradContext *ctx, variable_list gradients) {
        auto saved = ctx->get_saved_variables();
        const auto configuration = ctx->saved_data["configuration"].toTensor();
        const auto *parameters = configuration.const_data_ptr<int64_t>();
        const auto &q = saved[0];
        c10::cuda::CUDAGuard guard(q.device());
        auto dout = contiguous_gradient(gradients[0]);
        auto dq = at::empty_like(q), dk = at::empty_like(saved[1]), dv = at::empty_like(saved[2]);
        if (parameters[0] == 12 || parameters[0] == 13) {
            auto partials = at::empty({2, q.numel()}, q.options().dtype(at::kFloat));
            launch(true, {q, saved[1], saved[2], q, saved[3], saved[4], saved[5], dout,
                          saved[4], dq, dk, dv, partials, saved[4]},
                   ctx->saved_data["scale"].toDouble(), ctx->saved_data["causal"].toBool(), false, false, configuration);
            return {dq, dk, dv, Tensor(), Tensor(), Tensor()};
        }
        if (parameters[0] == 9) {
            auto partials = at::empty({4, q.numel()}, q.options().dtype(at::kFloat));
            launch(true, {q, saved[1], saved[2], q, saved[3], saved[4], saved[5], dout,
                          saved[4], dq, dk, dv, partials, saved[4]},
                   ctx->saved_data["scale"].toDouble(), ctx->saved_data["causal"].toBool(), false, false, configuration);
            return {dq, dk, dv, Tensor(), Tensor(), Tensor()};
        }
        if (parameters[0] == 3 || parameters[0] == 4) {
            // The short kernel owns all rows and uses no global workspace.
            launch(true, {q, saved[1], saved[2], q, saved[3], saved[4], saved[5], dout,
                          saved[4], dq, dk, dv, saved[4], saved[4]},
                   ctx->saved_data["scale"].toDouble(), ctx->saved_data["causal"].toBool(), false, false, configuration);
            return {dq, dk, dv, Tensor(), Tensor(), Tensor()};
        }
        const auto tile_m = parameters[configuration_backward_m];
        const auto rounded = (q.size(2) + tile_m - 1) / tile_m * tile_m;
        const auto accumulator_width = parameters[configuration_accumulator_head_dim] == 0
                                           ? q.size(3) : parameters[configuration_accumulator_head_dim];
        auto delta = at::empty({q.size(0), q.size(1), rounded}, q.options().dtype(at::kFloat));
        auto lognorm = at::empty_like(delta);
        auto scratch = at::empty({q.size(0), q.size(1), rounded * accumulator_width}, delta.options());
        launch(true, {q, saved[1], saved[2], q, saved[3], saved[4], saved[5], dout, delta, dq, dk, dv, scratch, lognorm},
               ctx->saved_data["scale"].toDouble(), ctx->saved_data["causal"].toBool(), false, false, configuration);
        return {dq, dk, dv, Tensor(), Tensor(), Tensor()};
    }
};

bool validate(Tensor q, Tensor k, Tensor v, const std::optional<Tensor> &mask);

class AttentionRunner {
    Tensor configuration;

  public:
    AttentionRunner(const std::string &family, const std::map<std::string, int64_t> &options) {
        TORCH_CHECK_VALUE(family == "warp" || family == "hopper" || family == "short" || family == "short-optimized" || family == "short_key32" || family == "short_d128" || family == "short_d128_stats",
                          "Select the warp, hopper, short or short-optimized kernel family.");
        std::vector<int64_t> values(1 + configured_fields.size());
        values[0] = family == "short_d128_stats" ? 13 : family == "short_d128" ? 12 : family == "short_key32" ? 9 : family == "short-optimized" ? 4 : family == "short" ? 3 : family == "warp" ? 1 : 2;
        TORCH_CHECK_VALUE((family != "short" && family != "short-optimized" && family != "short_key32" && family != "short_d128" && family != "short_d128_stats") || options.empty(),
                          "The short kernel has one fixed geometry.");
        size_t recognized = 0;
        for (size_t i = 0; i < configured_fields.size(); ++i) {
            auto entry = options.find(configured_fields[i]);
            values[i + 1] = entry == options.end() ? configured_defaults[i] : entry->second;
            recognized += entry != options.end();
        }
        TORCH_CHECK_VALUE(recognized == options.size(), "Unknown attention launch configuration field.");
        const auto power_of_two = [](int64_t n) { return n > 0 && (n & (n - 1)) == 0; };
        for (int index : {1, 2, 5, 6})
            TORCH_CHECK_VALUE(power_of_two(values[index]) && values[index] >= 16 && values[index] <= 256,
                              "Attention tile dimensions must be powers of two in 16..256.");
        TORCH_CHECK_VALUE(values[3] >= 1 && values[3] <= 2 && values[7] >= 1 && values[7] <= 2 &&
                              values[8] >= 1 && values[8] <= values[7], "Unsupported pipeline stage counts.");
        for (int index : {4, 13})
            TORCH_CHECK_VALUE(values[index] == 0 || values[index] == 1, "Attention configuration flags must be 0 or 1.");
        TORCH_CHECK_VALUE(power_of_two(values[15]) && values[15] <= 1024,
                          "Maximum padding must be a power of two up to 1024.");
        TORCH_CHECK_VALUE(values[16] == 0 || (values[16] >= 32 && values[16] <= 512 && values[16] % 32 == 0),
                          "Accumulator width must be zero or a multiple of 32 up to 512.");
        TORCH_CHECK_VALUE(values[17] == 1 || values[17] == 2 || values[17] == 4 ||
                              values[17] == 8 || values[17] == 16,
                          "Causal head groups must be one of 1, 2, 4, 8 or 16.");
        TORCH_CHECK_VALUE(family == "warp" || values[17] == 1,
                          "Causal head grouping requires the warp kernel family.");
        if (family != "hopper") {
            TORCH_CHECK_VALUE(values[9] == 256 && (values[14] == 128 || values[14] == 256),
                              "Warp experiments require 256 backward threads and 128/256 forward threads.");
            for (int index : {10, 11, 12})
                TORCH_CHECK_VALUE(power_of_two(values[index]) && values[index] <= 8, "Unsupported MMA warp layout.");
            TORCH_CHECK_VALUE(!(values[3] > 1 && values[4]),
                              "Register-Q with a multi-stage forward is not qualified.");
            TORCH_CHECK_VALUE(values[15] == 1,
                              "Warp experiments retain exact maximum rows.");
        } else {
            TORCH_CHECK_VALUE(values[15] % values[5] == 0,
                              "Hopper maximum padding must cover complete backward tiles.");
        }
        configuration = at::tensor(values, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
    }

    Tensor call(Tensor q, Tensor k, Tensor v, std::optional<double> scale, bool causal, bool inference = false) const {
        validate(q, k, v, std::nullopt);
        const auto target = device_capability(q.get_device());
        const auto family = configuration.const_data_ptr<int64_t>()[0];
        const auto amount = scale.value_or(std::pow(q.size(3), -0.5));
        const bool a40_width = family == 1 && !inference && q.size(3) == 256 && full_a40(q.get_device());
        const bool sm120_width = family == 1 && !inference && q.size(3) == 256 &&
                                 c10::GradMode::is_enabled() && q.requires_grad() && k.requires_grad() &&
                                 v.requires_grad() && full_rtx5090(q.get_device());
        const bool warp_device = (target.first == 8 && (target.second == 0 || target.second == 6 || target.second == 9)) ||
                                 (target.first == 12 && (target.second == 0 || target.second == 1));
        const bool qualified_a100_d256 = !inference && family == 1 && q.size(3) == 256 &&
                                         target == std::make_pair(8, 0) && full_a100(q.get_device());
        TORCH_CHECK_VALUE(((family == 1 || family == 3 || family == 4 || family == 9 || family == 12 || family == 13) && warp_device) || (family == 2 && target == std::make_pair(9, 0)),
                          "The configured attention family does not match the CUDA device.");
        const bool qualified_inference_width = inference && family == 1 &&
            (q.size(3) == 16 || q.size(3) == 256) && q.size(0) == 1 &&
            q.size(1) == 16 && (q.size(2) == 128 ||
              (q.size(3) == 256 && (q.size(2) == 512 || q.size(2) == 1024 || q.size(2) == 4096))) &&
            full_gb10(q.get_device());
        if (family == 3 || family == 4) {
            static std::mutex short_device_mutex;
            struct ShortSupport { bool basic; bool optimized; };
            static std::unordered_map<int, ShortSupport> short_devices;
            std::lock_guard<std::mutex> lock(short_device_mutex);
            auto found = short_devices.find(q.get_device());
            if (found == short_devices.end()) {
                cudaDeviceProp properties{};
                C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, q.get_device()));
                const bool supported = properties.sharedMemPerBlockOptin >= 99840 &&
                    ((properties.major == 8 && properties.minor == 6 && properties.multiProcessorCount == 82 &&
                      std::strcmp(properties.name, "NVIDIA GeForce RTX 3090") == 0) ||
                     (properties.major == 8 && properties.minor == 9 && properties.multiProcessorCount == 128 &&
                      std::strcmp(properties.name, "NVIDIA GeForce RTX 4090") == 0) ||
                     (properties.major == 8 && properties.minor == 9 && properties.multiProcessorCount == 142 &&
                      properties.sharedMemPerBlockOptin == 101376 &&
                      std::strcmp(properties.name, "NVIDIA L40S") == 0));
                const bool optimized = supported && properties.major == 8 && properties.minor == 9 &&
                    ((properties.multiProcessorCount == 128 &&
                      std::strcmp(properties.name, "NVIDIA GeForce RTX 4090") == 0) ||
                     (properties.multiProcessorCount == 142 && properties.sharedMemPerBlockOptin == 101376 &&
                      std::strcmp(properties.name, "NVIDIA L40S") == 0));
                found = short_devices.emplace(q.get_device(), ShortSupport{supported, optimized}).first;
            }
            const bool supported = family == 3 ? found->second.basic : found->second.optimized;
            TORCH_CHECK_VALUE(supported && !inference && q.size(0) == 1 && q.size(1) == 16 &&
                                  q.size(2) == 128 && q.size(3) == 16,
                              "Short attention requires a qualified full device square B1 H16 S128 D16 training call.");
        }
        if (family == 9 || family == 12 || family == 13) {
            static std::mutex short_device_mutex;
            static std::unordered_map<int, bool> short_devices;
            std::lock_guard<std::mutex> lock(short_device_mutex);
            auto found = short_devices.find(q.get_device());
            if (found == short_devices.end()) {
                cudaDeviceProp properties{};
                C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, q.get_device()));
                const bool supported = properties.major == 8 && properties.minor == 6 &&
                                       properties.multiProcessorCount == 82 &&
                                       properties.sharedMemPerBlockOptin >= 99840 &&
                                       std::strcmp(properties.name, "NVIDIA GeForce RTX 3090") == 0;
                found = short_devices.emplace(q.get_device(), supported).first;
            }
            TORCH_CHECK_VALUE((family != 13 || causal) && found->second && !inference && q.size(0) == 1 && q.size(1) == 16 &&
                                  q.size(2) == 128 && ((family == 9 && q.size(3) == 64) || ((family == 12 || family == 13) && q.size(3) == 128)),
                              "Short attention requires full RTX 3090 square B1 H16 S128 D64 or D128 training.");
        }
        TORCH_CHECK_VALUE(q.sizes() == k.sizes() && q.sizes() == v.sizes() &&
                              q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                              reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
                              reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0 &&
                              reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0 &&
                              (q.size(3) == 64 || q.size(3) == 128 || a40_width || sm120_width || qualified_inference_width ||
                               qualified_a100_d256 || family == 3 || family == 4 || family == 9 || family == 12) &&
                              std::isfinite(amount) &&
                              amount >= std::numeric_limits<float>::min() && amount <= 2e38,
                          "Configured attention requires aligned contiguous square qualified-width FP16/BF16 MHA and a positive normal FP32 scale.");
        const auto accumulator_width = configuration.const_data_ptr<int64_t>()[configuration_accumulator_head_dim];
        TORCH_CHECK_VALUE(accumulator_width == 0 || accumulator_width == q.size(3),
                          "These configured experiments require the input head width for accumulation.");
        if (inference) {
            TORCH_CHECK_VALUE(family == 1, "Explicit inference is available for the warp family only.");
            // Keep inference eligibility independent of future training-view
            // experiments: this mode has only contiguous aligned qualification.
            TORCH_CHECK_VALUE(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                                  !q.is_conj() && !k.is_conj() && !v.is_conj() &&
                                  !q.is_neg() && !k.is_neg() && !v.is_neg() &&
                                  reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
                                  reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0 &&
                                  reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0,
                              "Explicit inference requires contiguous aligned inputs without pending conjugation or negation.");
            TORCH_CHECK_VALUE(!c10::GradMode::is_enabled() || !(q.requires_grad() || k.requires_grad() || v.requires_grad()),
                              "Explicit inference cannot discard requested input gradients.");
            TORCH_CHECK_VALUE(!q._fw_grad(0).defined() && !k._fw_grad(0).defined() && !v._fw_grad(0).defined(),
                              "Explicit inference does not support forward-AD dual inputs.");
            c10::cuda::CUDAGuard guard(q.device());
            auto out = at::empty_like(q);
            // Every unused argument has the same real/fake dtype and rank as
            // out. No statistic allocation or autograd context is constructed.
            launch(false, {q, k, v, q, out, out, out, out, out, out, out, out, out, out},
                   amount, causal, false, false, configuration, true);
            return out;
        }
        return ConfiguredAttention::apply(q, k, v, amount, causal, configuration);
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
    return (major == 8 && (minor == 0 || minor == 6 || minor == 9)) || (major == 9 && minor == 0) || (major == 10 && (minor == 0 || minor == 3)) || (major == 12 && (minor == 0 || minor == 1));
}

bool full_a100(int device) {
    static std::map<int, bool> devices;
    static std::mutex device_mutex;
    std::lock_guard<std::mutex> lock(device_mutex);
    auto found = devices.find(device);
    if (found != devices.end())
        return found->second;
    cudaDeviceProp properties;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    const bool selected = properties.major == 8 && properties.minor == 0 &&
                          properties.multiProcessorCount == 108;
    return devices.emplace(device, selected).first->second;
}

bool full_a40(int device) {
    static std::map<int, bool> devices;
    static std::mutex device_mutex;
    std::lock_guard<std::mutex> lock(device_mutex);
    auto found = devices.find(device);
    if (found != devices.end())
        return found->second;
    cudaDeviceProp properties;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    const bool selected = properties.major == 8 && properties.minor == 6 &&
                          std::strcmp(properties.name, "NVIDIA A40") == 0 &&
                          properties.multiProcessorCount == 84 && properties.sharedMemPerBlockOptin == 101376;
    return devices.emplace(device, selected).first->second;
}

bool a100_medium_metadata_compatible(const Tensor &q, const Tensor &k, const Tensor &v) {
    if (!c10::GradMode::is_enabled() || !q.requires_grad() || !k.requires_grad() || !v.requires_grad() ||
        q.dim() != 4 || k.dim() != 4 || v.dim() != 4 || q.sizes() != k.sizes() || q.sizes() != v.sizes() ||
        !q.is_cuda() || q.device() != k.device() || q.device() != v.device() ||
        (q.scalar_type() != at::kHalf && q.scalar_type() != at::kBFloat16) ||
        q.scalar_type() != k.scalar_type() || q.scalar_type() != v.scalar_type() ||
        q.size(0) != 1 || q.size(1) != 8 || q.size(2) != 2048 || q.size(3) != 128 ||
        !q.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() ||
        reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 != 0)
        return false;
    int current;
    C10_CUDA_CHECK(cudaGetDevice(&current));
    return q.get_device() == current && full_a100(q.get_device());
}

bool full_rtx5090(int device) {
    static std::map<int, bool> devices;
    static std::mutex device_mutex;
    std::lock_guard<std::mutex> lock(device_mutex);
    auto found = devices.find(device);
    if (found != devices.end())
        return found->second;
    cudaDeviceProp properties;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    const bool selected = properties.major == 12 && properties.minor == 0 &&
                          std::strstr(properties.name, "5090") != nullptr &&
                          properties.multiProcessorCount == 170 && properties.sharedMemPerBlockOptin == 101376;
    return devices.emplace(device, selected).first->second;
}

bool sm120_metadata_compatible(const Tensor &q, const Tensor &k, const Tensor &v, bool causal) {
    if (!c10::GradMode::is_enabled() || !q.requires_grad() || !k.requires_grad() || !v.requires_grad() ||
        q.dim() != 4 || k.dim() != 4 || v.dim() != 4 || q.sizes() != k.sizes() || q.sizes() != v.sizes() ||
        !q.is_cuda() || q.device() != k.device() || q.device() != v.device() ||
        (q.scalar_type() != at::kHalf && q.scalar_type() != at::kBFloat16) ||
        q.scalar_type() != k.scalar_type() || q.scalar_type() != v.scalar_type() ||
        q.size(0) != 1 || q.size(1) != 16 ||
        !(q.size(2) == 4096 || (q.size(2) == 1024 &&
          (q.size(3) == 256 || (causal && (q.size(3) == 64 || q.scalar_type() == at::kBFloat16))))) ||
        (q.size(3) != 64 && q.size(3) != 128 && q.size(3) != 256) ||
        !q.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() ||
        reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 != 0)
        return false;
    int current;
    C10_CUDA_CHECK(cudaGetDevice(&current));
    return q.get_device() == current && full_rtx5090(q.get_device());
}

bool a40_metadata_compatible(const Tensor &q, const Tensor &k, const Tensor &v, bool causal) {
    if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4 || q.sizes() != k.sizes() || q.sizes() != v.sizes() ||
        !q.is_cuda() || q.device() != k.device() || q.device() != v.device() ||
        (q.scalar_type() != at::kHalf && q.scalar_type() != at::kBFloat16) ||
        q.scalar_type() != k.scalar_type() || q.scalar_type() != v.scalar_type() ||
        q.size(0) != 1 || q.size(1) != 16 || (q.size(2) != 1024 && q.size(2) != 4096 && !(q.size(2) == 128 && q.size(3) == 256)) ||
        !(q.size(3) == 128 || q.size(3) == 256 || (q.size(3) == 64 && causal)) ||
        !q.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() ||
        reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 != 0)
        return false;
    int current;
    C10_CUDA_CHECK(cudaGetDevice(&current));
    return q.get_device() == current && full_a40(q.get_device());
}

bool full_gb10(int device) {
    static std::map<int, bool> devices;
    static std::mutex device_mutex;
    std::lock_guard<std::mutex> lock(device_mutex);
    auto found = devices.find(device);
    if (found != devices.end())
        return found->second;
    cudaDeviceProp properties;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    const bool selected = properties.major == 12 && properties.minor == 1 &&
                          std::strcmp(properties.name, "NVIDIA GB10") == 0 &&
                          properties.multiProcessorCount == 48 && properties.sharedMemPerBlockOptin == 101376;
    return devices.emplace(device, selected).first->second;
}

bool gb10_inference_metadata_compatible(const Tensor &q, const Tensor &k, const Tensor &v, bool causal) {
    if (c10::GradMode::is_enabled() && (q.requires_grad() || k.requires_grad() || v.requires_grad()))
        return false;
    if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4 || q.sizes() != k.sizes() || q.sizes() != v.sizes() ||
        !q.is_cuda() || q.device() != k.device() || q.device() != v.device() ||
        (q.scalar_type() != at::kHalf && q.scalar_type() != at::kBFloat16) ||
        q.scalar_type() != k.scalar_type() || q.scalar_type() != v.scalar_type() ||
        q.size(0) != 1 || q.size(1) != 16 ||
        (q.size(2) != 128 && q.size(2) != 512 && q.size(2) != 1024 && q.size(2) != 4096) ||
        (!causal && q.size(2) == 512 && q.size(3) == 128 && q.scalar_type() == at::kBFloat16) ||
        !(q.size(3) == 128 || q.size(3) == 256 || (q.size(2) == 4096 && q.size(3) == 64) || (q.size(2) == 128 &&
          (q.size(3) == 16 || q.size(3) == 64 || q.size(3) == 256))) ||
        !q.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() ||
        q.is_conj() || k.is_conj() || v.is_conj() || q.is_neg() || k.is_neg() || v.is_neg() ||
        reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 != 0)
        return false;
    int current;
    C10_CUDA_CHECK(cudaGetDevice(&current));
    return q.get_device() == current && full_gb10(q.get_device());
}

bool full_l40s(int device) {
    static std::map<int, bool> devices;
    static std::mutex device_mutex;
    std::lock_guard<std::mutex> lock(device_mutex);
    auto found = devices.find(device);
    if (found != devices.end())
        return found->second;
    cudaDeviceProp properties;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    const bool selected = properties.major == 8 && properties.minor == 9 &&
                          std::strcmp(properties.name, "NVIDIA L40S") == 0 &&
                          properties.multiProcessorCount == 142 && properties.sharedMemPerBlockOptin == 101376;
    return devices.emplace(device, selected).first->second;
}

bool l40s_metadata_compatible(const Tensor &q, const Tensor &k, const Tensor &v, bool causal, bool short_owned = false) {
    if (!c10::GradMode::is_enabled() || !(q.requires_grad() || k.requires_grad() || v.requires_grad()) ||
        q.dim() != 4 || k.dim() != 4 || v.dim() != 4 || q.sizes() != k.sizes() || q.sizes() != v.sizes() ||
        !q.is_cuda() || q.device() != k.device() || q.device() != v.device() ||
        (q.scalar_type() != at::kHalf && q.scalar_type() != at::kBFloat16) ||
        q.scalar_type() != k.scalar_type() || q.scalar_type() != v.scalar_type() ||
        q.size(0) != 1 || q.size(1) != 16 || !((short_owned && q.size(2) == 128 && q.size(3) == 16) ||
          ((q.size(2) == 1024 || q.size(2) == 4096) && (q.size(3) == 64 || q.size(3) == 128))) ||
        !q.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() ||
        reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 != 0)
        return false;
    int current;
    C10_CUDA_CHECK(cudaGetDevice(&current));
    return q.get_device() == current && full_l40s(q.get_device());
}

bool full_rtx4090(int device) {
    static std::map<int, bool> devices;
    static std::mutex device_mutex;
    std::lock_guard<std::mutex> lock(device_mutex);
    auto found = devices.find(device);
    if (found != devices.end())
        return found->second;
    cudaDeviceProp properties;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    const bool selected = properties.major == 8 && properties.minor == 9 &&
                          std::strcmp(properties.name, "NVIDIA GeForce RTX 4090") == 0 &&
                          properties.multiProcessorCount == 128 && properties.sharedMemPerBlockOptin == 101376;
    return devices.emplace(device, selected).first->second;
}

bool rtx4090_metadata_compatible(const Tensor &q, const Tensor &k, const Tensor &v, bool /*causal*/) {
    if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4 || q.sizes() != k.sizes() || q.sizes() != v.sizes() ||
        !q.is_cuda() || q.device() != k.device() || q.device() != v.device() ||
        (q.scalar_type() != at::kHalf && q.scalar_type() != at::kBFloat16) ||
        q.scalar_type() != k.scalar_type() || q.scalar_type() != v.scalar_type() ||
        q.size(0) != 1 || q.size(1) != 16 || (q.size(2) != 1024 && q.size(2) != 4096) ||
        (q.size(3) != 64 && q.size(3) != 128) ||
        !q.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() ||
        reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 != 0)
        return false;
    int current;
    C10_CUDA_CHECK(cudaGetDevice(&current));
    return q.get_device() == current && full_rtx4090(q.get_device());
}

bool rtx4090_automatic_metadata_compatible(const Tensor &q, const Tensor &k, const Tensor &v, bool causal) {
    if (!c10::GradMode::is_enabled() || !(q.requires_grad() || k.requires_grad() || v.requires_grad()) ||
        q.dim() != 4 || k.dim() != 4 || v.dim() != 4 || q.sizes() != k.sizes() || q.sizes() != v.sizes() ||
        !q.is_cuda() || q.device() != k.device() || q.device() != v.device() ||
        (q.scalar_type() != at::kHalf && q.scalar_type() != at::kBFloat16) ||
        q.scalar_type() != k.scalar_type() || q.scalar_type() != v.scalar_type() ||
        q.size(0) != 1 || q.size(1) != 16 ||
        !((q.size(2) == 128 && q.size(3) == 16) ||
          (q.size(2) == 4096 && (q.size(3) == 64 || q.size(3) == 128))) ||
        !q.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() ||
        q.is_conj() || k.is_conj() || v.is_conj() || q.is_neg() || k.is_neg() || v.is_neg() ||
        reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 != 0)
        return false;
    int current;
    C10_CUDA_CHECK(cudaGetDevice(&current));
    return q.get_device() == current && full_rtx4090(q.get_device());
}

bool compatible(Tensor q, Tensor k, Tensor v, const std::optional<Tensor> &mask, double dropout, bool causal, bool gqa) {
    return dropout == 0.0 && q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && q.is_cuda() && q.device() == k.device() &&
           q.device() == v.device() && (q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16) &&
           q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type() && q.numel() > 0 && k.numel() > 0 &&
           k.sizes() == v.sizes() && q.size(0) == k.size(0) && q.size(3) == k.size(3) && q.size(3) <= 512 &&
           q.size(1) % k.size(1) == 0 && (gqa || q.size(1) == k.size(1)) &&
           (!mask || (mask->scalar_type() == at::kBool && mask->device() == q.device() && !causal)) &&
           supported_device(q.get_device());
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
    TORCH_CHECK_VALUE(supported_device(q.get_device()), "CuTe attention requires sm_80, sm_86, sm_89, sm_90, sm_100, sm_103, sm_120 or sm_121.");
    return true;
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("a100_medium_metadata_compatible", [](py::object q, py::object k, py::object v) {
        if (!THPVariable_Check(q.ptr()) || !THPVariable_Check(k.ptr()) || !THPVariable_Check(v.ptr()))
            return false;
        return a100_medium_metadata_compatible(THPVariable_Unpack(q.ptr()), THPVariable_Unpack(k.ptr()),
                                               THPVariable_Unpack(v.ptr()));
    });
    module.def("gb10_inference_metadata_compatible", [](py::object q, py::object k, py::object v, bool causal) {
        if (!THPVariable_Check(q.ptr()) || !THPVariable_Check(k.ptr()) || !THPVariable_Check(v.ptr()))
            return false;
        return gb10_inference_metadata_compatible(THPVariable_Unpack(q.ptr()), THPVariable_Unpack(k.ptr()),
                                                  THPVariable_Unpack(v.ptr()), causal);
    });
    module.def("l40s_metadata_compatible", [](py::object q, py::object k, py::object v, bool causal, bool short_owned) {
        if (!THPVariable_Check(q.ptr()) || !THPVariable_Check(k.ptr()) || !THPVariable_Check(v.ptr()))
            return false;
        return l40s_metadata_compatible(THPVariable_Unpack(q.ptr()), THPVariable_Unpack(k.ptr()),
                                        THPVariable_Unpack(v.ptr()), causal, short_owned);
    }, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal"), py::arg("short_owned") = false);
    module.def("rtx4090_automatic_metadata_compatible", [](py::object q, py::object k, py::object v, bool causal) {
        if (!THPVariable_Check(q.ptr()) || !THPVariable_Check(k.ptr()) || !THPVariable_Check(v.ptr()))
            return false;
        return rtx4090_automatic_metadata_compatible(THPVariable_Unpack(q.ptr()), THPVariable_Unpack(k.ptr()),
                                                    THPVariable_Unpack(v.ptr()), causal);
    });
    module.def("rtx4090_guarded_call", [](const AttentionRunner &runner, Tensor q, Tensor k, Tensor v,
                                         int64_t dimension, std::optional<double> scale, bool causal)
                                         -> std::optional<Tensor> {
        if (!c10::GradMode::is_enabled() || !(q.requires_grad() || k.requires_grad() || v.requires_grad()) ||
            q.dim() != 4 || q.size(2) != 1024 || q.size(3) != dimension ||
            q.is_neg() || k.is_neg() || v.is_neg() || q.is_conj() || k.is_conj() || v.is_conj() ||
            !rtx4090_metadata_compatible(q, k, v, causal))
            return std::nullopt;
        py::gil_scoped_release release;
        return runner.call(q, k, v, scale, causal);
    }, py::arg("runner"), py::arg("q"), py::arg("k"), py::arg("v"), py::arg("dimension"),
       py::arg("scale") = std::nullopt, py::arg("causal") = false);
    module.def("rtx4090_metadata_compatible", [](py::object q, py::object k, py::object v, bool causal) {
        if (!THPVariable_Check(q.ptr()) || !THPVariable_Check(k.ptr()) || !THPVariable_Check(v.ptr()))
            return false;
        return rtx4090_metadata_compatible(THPVariable_Unpack(q.ptr()), THPVariable_Unpack(k.ptr()),
                                           THPVariable_Unpack(v.ptr()), causal);
    });
    module.def("sm120_metadata_compatible", [](py::object q, py::object k, py::object v, bool causal) {
        if (!THPVariable_Check(q.ptr()) || !THPVariable_Check(k.ptr()) || !THPVariable_Check(v.ptr()))
            return false;
        return sm120_metadata_compatible(THPVariable_Unpack(q.ptr()), THPVariable_Unpack(k.ptr()),
                                         THPVariable_Unpack(v.ptr()), causal);
    });
    module.def("a40_metadata_compatible", [](py::object q, py::object k, py::object v, bool causal) {
        if (!THPVariable_Check(q.ptr()) || !THPVariable_Check(k.ptr()) || !THPVariable_Check(v.ptr()))
            return false;
        return a40_metadata_compatible(THPVariable_Unpack(q.ptr()), THPVariable_Unpack(k.ptr()),
                                       THPVariable_Unpack(v.ptr()), causal);
    });
    py::class_<AttentionRunner>(module, "AttentionRunner", py::module_local())
        .def(py::init<const std::string &, const std::map<std::string, int64_t> &>(),
             py::arg("family"), py::arg("configuration") = std::map<std::string, int64_t>{})
        .def("__call__", [](const AttentionRunner &runner, Tensor q, Tensor k, Tensor v,
                            std::optional<double> scale, bool causal) {
            py::gil_scoped_release release;
            return runner.call(q, k, v, scale, causal);
        }, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale") = std::nullopt, py::arg("causal") = false)
        .def("inference", [](const AttentionRunner &runner, Tensor q, Tensor k, Tensor v,
                              std::optional<double> scale, bool causal) {
            py::gil_scoped_release release;
            return runner.call(q, k, v, scale, causal, true);
        }, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale") = std::nullopt, py::arg("causal") = false);
    module.def("sm80_attention", [](Tensor q, Tensor k, Tensor v, double scale, bool causal) {
        py::gil_scoped_release release;
        validate(q, k, v, std::nullopt);
        TORCH_CHECK_VALUE(device_capability(q.get_device()) == std::make_pair(8, 0) &&
                              q.sizes() == k.sizes() && q.sizes() == v.sizes() &&
                              q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                              reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
                              reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0 &&
                              reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0 &&
                              (q.size(3) == 64 || q.size(3) == 128) &&
                              std::isfinite(scale) && scale >= std::numeric_limits<float>::min() && scale <= 2e38,
                          "The native SM80 entry requires aligned contiguous square FP16/BF16 MHA, D64/D128 and a positive scale.");
        return Sm80Attention::apply(q, k, v, scale, causal);
    });
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
