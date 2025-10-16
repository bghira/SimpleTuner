import shutil
import sys
import tempfile
import types
import unittest


def _ensure_torchao_stub():
    if "torchao.optim" in sys.modules:
        return
    torchao_module = types.ModuleType("torchao")
    optim_module = types.ModuleType("torchao.optim")
    dummy_class = type("DummyOptimizer", (), {})

    optim_module.AdamFp8 = dummy_class
    optim_module.AdamW4bit = dummy_class
    optim_module.AdamW8bit = dummy_class
    optim_module.AdamWFp8 = dummy_class
    optim_module.CPUOffloadOptimizer = dummy_class

    torchao_module.optim = optim_module
    sys.modules["torchao"] = torchao_module
    sys.modules["torchao.optim"] = optim_module


_ensure_torchao_stub()


def _ensure_optimi_stub():
    if "optimi" in sys.modules:
        return
    optimi_module = types.ModuleType("optimi")
    for cls_name in [
        "StableAdamW",
        "AdamW",
        "Lion",
        "RAdam",
        "Ranger",
        "Adan",
        "Adam",
        "SGD",
    ]:
        setattr(optimi_module, cls_name, type(cls_name, (), {}))

    def _prepare_for_gradient_release(*_args, **_kwargs):
        return None

    optimi_module.prepare_for_gradient_release = _prepare_for_gradient_release
    sys.modules["optimi"] = optimi_module


_ensure_optimi_stub()


def _ensure_fastapi_stub():
    if "fastapi" in sys.modules:
        return

    fastapi_module = types.ModuleType("fastapi")

    class FastAPI:
        def __init__(self, *_, **__):
            self.state = types.SimpleNamespace()

        def add_middleware(self, *_, **__):
            return None

        def mount(self, *_, **__):
            return None

    class APIRouter:
        pass

    class HTTPException(Exception):
        pass

    fastapi_module.FastAPI = FastAPI
    fastapi_module.APIRouter = APIRouter
    fastapi_module.HTTPException = HTTPException

    sys.modules["fastapi"] = fastapi_module

    middleware_module = types.ModuleType("fastapi.middleware")
    sys.modules["fastapi.middleware"] = middleware_module

    cors_module = types.ModuleType("fastapi.middleware.cors")

    class CORSMiddleware:
        def __init__(self, *_, **__):
            pass

    cors_module.CORSMiddleware = CORSMiddleware
    sys.modules["fastapi.middleware.cors"] = cors_module

    responses_module = types.ModuleType("fastapi.responses")

    class FileResponse:
        def __init__(self, *_, **__):
            pass

    responses_module.FileResponse = FileResponse
    sys.modules["fastapi.responses"] = responses_module

    staticfiles_module = types.ModuleType("fastapi.staticfiles")

    class StaticFiles:
        def __init__(self, *_, **__):
            pass

    staticfiles_module.StaticFiles = StaticFiles
    sys.modules["fastapi.staticfiles"] = staticfiles_module


_ensure_fastapi_stub()

from simpletuner.helpers.configuration.cmd_args import parse_cmdline_args


class TestFSDPCmdArgs(unittest.TestCase):
    def test_fsdp_cli_normalization(self):
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(tmp_dir, ignore_errors=True))
        base_args = [
            "--model_family=sdxl",
            "--model_type=full",
            f"--output_dir={tmp_dir}",
            "--optimizer=adamw_bf16",
            "--data_backend_config=dummy",
        ]
        args = parse_cmdline_args(
            input_args=base_args
            + [
                "--fsdp_enable",
                "--fsdp_state_dict_type=FULL_STATE_DICT",
                "--fsdp_auto_wrap_policy=NO_WRAP",
                "--fsdp_transformer_layer_cls_to_wrap=MyLayer,OtherLayer",
            ]
        )

        self.assertTrue(args.fsdp_enable)
        self.assertEqual(args.fsdp_version, 2)
        self.assertEqual(args.fsdp_state_dict_type, "FULL_STATE_DICT")
        self.assertEqual(args.fsdp_auto_wrap_policy, "no_wrap")
        self.assertEqual(args.fsdp_transformer_layer_cls_to_wrap, ["MyLayer", "OtherLayer"])

    def test_fsdp_and_deepspeed_mutually_exclusive(self):
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(tmp_dir, ignore_errors=True))
        base_args = [
            "--model_family=sdxl",
            "--model_type=full",
            f"--output_dir={tmp_dir}",
            "--optimizer=adamw_bf16",
            "--data_backend_config=dummy",
        ]
        with self.assertRaises(ValueError):
            parse_cmdline_args(
                input_args=base_args
                + [
                    "--fsdp_enable",
                    '--deepspeed_config={"zero_optimization": {"stage": 2}}',
                ],
                exit_on_error=False,
            )


if __name__ == "__main__":
    unittest.main()
