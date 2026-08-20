import torch

from tests.test_pipelines._common import PipelineTestCase, WanPromptCleaningMixin


class TestWanPipeline(WanPromptCleaningMixin, PipelineTestCase):
    module_name = "wan"

    def test_cached_prompt_embeds_repeat_for_each_requested_video(self):
        pipeline = object.__new__(self.pipeline_module.WanPipeline)
        prompt_embeds = torch.tensor([[[1.0]], [[2.0]]])
        negative_prompt_embeds = torch.tensor([[[-1.0]], [[-2.0]]])

        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
            prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=2,
            device=torch.device("cpu"),
        )

        self.assertEqual(prompt_embeds.shape, (4, 1, 1))
        self.assertEqual(negative_prompt_embeds.shape, (4, 1, 1))
        self.assertTrue(torch.equal(prompt_embeds[:, 0, 0], torch.tensor([1.0, 1.0, 2.0, 2.0])))
        self.assertTrue(torch.equal(negative_prompt_embeds[:, 0, 0], torch.tensor([-1.0, -1.0, -2.0, -2.0])))
