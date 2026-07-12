import unittest

from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput


def _get_submodule(module, path: str):
    if path == "":
        return module
    current = module
    for atom in path.split("."):
        if atom.isdigit():
            current = current[int(atom)]
        else:
            current = getattr(current, atom)
    return current


def _assert_valid_cp_plan(testcase: unittest.TestCase, module):
    plan = getattr(module, "_cp_plan", None)
    testcase.assertIsInstance(plan, dict)
    testcase.assertTrue(plan)
    for module_id, entry in plan.items():
        _get_submodule(module, module_id)
        if isinstance(entry, dict):
            testcase.assertTrue(entry)
            for value in entry.values():
                if isinstance(value, (list, tuple)):
                    testcase.assertTrue(value)
                    for item in value:
                        testcase.assertIsInstance(item, ContextParallelInput)
                else:
                    testcase.assertIsInstance(value, ContextParallelInput)
        elif isinstance(entry, (list, tuple)):
            testcase.assertTrue(entry)
            for item in entry:
                testcase.assertIsInstance(item, ContextParallelOutput)
        else:
            testcase.assertIsInstance(entry, ContextParallelOutput)


class ContextParallelPlanTests(unittest.TestCase):
    def test_anima_cp_plan_targets_exist(self):
        from simpletuner.helpers.models.anima.transformer import AnimaTransformerModel

        model = AnimaTransformerModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=4,
            num_layers=1,
            text_embed_dim=8,
            adaln_lora_dim=4,
            max_size=(2, 4, 4),
            patch_size=(1, 2, 2),
            adapter_dim=8,
            adapter_layers=1,
            adapter_heads=2,
        )

        _assert_valid_cp_plan(self, model)

    def test_heartmula_codec_cp_plan_targets_exist(self):
        from simpletuner.helpers.models.heartmula.codec.transformer import LlamaTransformer

        model = LlamaTransformer(
            num_attention_heads=2,
            attention_head_dim=4,
            in_channels=4,
            out_channels=4,
            num_layers=1,
            num_layers_2=1,
        )

        _assert_valid_cp_plan(self, model)

    def test_ideogram_cp_plan_targets_exist(self):
        from simpletuner.helpers.models.ideogram.transformer import Ideogram4Config, Ideogram4Transformer

        model = Ideogram4Transformer(
            Ideogram4Config(
                emb_dim=12,
                num_layers=1,
                num_heads=3,
                intermediate_size=16,
                adanln_dim=8,
                in_channels=4,
                llm_features_dim=12,
                mrope_section=(1, 1, 1),
            )
        )

        _assert_valid_cp_plan(self, model)

    def test_wan_s2v_cp_plan_targets_exist(self):
        from simpletuner.helpers.models.wan_s2v.transformer import WanS2VTransformer3DModel

        model = WanS2VTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=4,
            out_channels=4,
            text_dim=8,
            freq_dim=8,
            audio_dim=8,
            audio_inject_layers=(0,),
            pose_dim=4,
            ffn_dim=32,
            num_layers=1,
            num_weighted_avg_layers=1,
            rope_max_seq_len=8,
            enable_framepack=False,
            add_last_motion=False,
        )

        _assert_valid_cp_plan(self, model)

    def test_wan_cp_plan_targets_exist(self):
        from simpletuner.helpers.models.wan.transformer import WanTransformer3DModel

        model = WanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=4,
            out_channels=4,
            text_dim=8,
            freq_dim=8,
            ffn_dim=32,
            num_layers=1,
            rope_max_seq_len=8,
        )

        _assert_valid_cp_plan(self, model)
        rope_plan = model._cp_plan["rope"]
        self.assertEqual(tuple(rope_plan.keys()), (0,))
        self.assertEqual(rope_plan[0].split_dim, 2)
        self.assertEqual(rope_plan[0].expected_dims, 4)
        self.assertTrue(rope_plan[0].split_output)

    def test_zlab_i1_cp_plan_targets_exist(self):
        from simpletuner.helpers.models.zlab_i1.transformer import ZlabI1Transformer2DModel

        model = ZlabI1Transformer2DModel(
            input_size=4,
            image_resolution=32,
            patch_size=2,
            in_channels=4,
            hidden_size=24,
            depth=3,
            num_heads=3,
            text_embed_dim=24,
            text_num_tokens=4,
        )

        _assert_valid_cp_plan(self, model)


if __name__ == "__main__":
    unittest.main()
