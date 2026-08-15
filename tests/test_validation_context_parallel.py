import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.common import AudioModelFoundation
from simpletuner.helpers.training.validation import Validation, _ValidationWorkItem


def _validation_for_rank(rank: int, *, world_size: int = 8):
    validation = Validation.__new__(Validation)
    validation.accelerator = SimpleNamespace(
        num_processes=world_size,
        process_index=rank,
        is_main_process=rank == 0,
    )
    validation.config = SimpleNamespace(validation_multigpu="batch-parallel")
    return validation


def _work_items(count: int):
    return [
        _ValidationWorkItem(
            index=idx,
            shortname=f"item_{idx}",
            prompt=f"prompt {idx}",
            conditioning=None,
            adapter_strength=None,
        )
        for idx in range(count)
    ]


class DummyAudioModel(AudioModelFoundation):
    def validation_audio_sample_rate(self):
        return 44100

    def _encode_prompts(self, prompts, is_negative_prompt=False):
        return {}

    def convert_text_embed_for_pipeline(self, text_embedding):
        return {}

    def convert_negative_text_embed_for_pipeline(self, text_embedding):
        return {}

    def model_predict(self, prepared_batch):
        return None


class ValidationContextParallelTests(unittest.TestCase):
    @contextmanager
    def _patch_cp(self, *, data_rank: int, data_local_rank: int, data_parallel_size: int = 4):
        with (
            patch("simpletuner.helpers.training.validation.get_cp_info", return_value=(True, object(), 0, 2)),
            patch(
                "simpletuner.helpers.training.validation.get_model_replica_data_info",
                return_value=(True, data_rank, data_local_rank, 2, data_parallel_size),
            ),
        ):
            yield

    def test_context_parallel_splits_prompts_by_model_replica(self):
        items = _work_items(5)
        validation = _validation_for_rank(2)
        with self._patch_cp(data_rank=1, data_local_rank=0):
            local_items, use_distributed, worker_count = validation._split_validation_work_items(items)

        self.assertTrue(use_distributed)
        self.assertEqual(worker_count, 4)
        self.assertEqual([item.index for item in local_items], [1])

    def test_context_parallel_replicates_prompt_within_cp_group(self):
        items = _work_items(3)
        leader = _validation_for_rank(0)
        peer = _validation_for_rank(1)

        with self._patch_cp(data_rank=0, data_local_rank=0):
            leader_items, _, _ = leader._split_validation_work_items(items)
        with self._patch_cp(data_rank=0, data_local_rank=1):
            peer_items, _, _ = peer._split_validation_work_items(items)

        self.assertEqual([item.index for item in leader_items], [0])
        self.assertEqual([item.index for item in peer_items], [0])

    def test_context_parallel_only_leader_publishes_payloads(self):
        leader = _validation_for_rank(0)
        peer = _validation_for_rank(1)

        with self._patch_cp(data_rank=0, data_local_rank=0):
            self.assertTrue(leader._should_publish_validation_payloads())
        with self._patch_cp(data_rank=0, data_local_rank=1):
            self.assertFalse(peer._should_publish_validation_payloads())

    def test_context_parallel_keeps_single_prompt_distributed(self):
        validation = _validation_for_rank(0)
        items = _work_items(1)

        with self._patch_cp(data_rank=0, data_local_rank=0):
            local_items, use_distributed, worker_count = validation._split_validation_work_items(items)

        self.assertTrue(use_distributed)
        self.assertEqual(worker_count, 4)
        self.assertEqual([item.index for item in local_items], [0])

    def test_batch_parallel_gathers_audio_payloads_from_peer_ranks(self):
        validation = Validation.__new__(Validation)
        validation.accelerator = SimpleNamespace(
            num_processes=2,
            process_index=0,
            is_main_process=True,
        )
        validation.config = SimpleNamespace(validation_multigpu="batch-parallel")
        validation.model = DummyAudioModel.__new__(DummyAudioModel)
        validation.validation_prompt_metadata = {
            "validation_prompts": ["prompt 0", "prompt 1"],
            "validation_shortnames": ["song_0", "song_1"],
        }
        validation.validation_image_inputs = None
        validation.validation_prompt_dict = None
        validation.validation_resolutions = [(0, 0)]
        validation.save_dir = "validation_images"
        validation.eval_scores = {}
        validation.validation_video_paths = {}
        validation.evaluation_result = None
        validation._check_abort = MagicMock()
        validation._use_context_parallel_validation = MagicMock(return_value=False)
        validation._split_validation_work_items = MagicMock(return_value=([_work_items(2)[0]], True, 2))
        validation._should_publish_validation_payloads = MagicMock(return_value=True)

        rank0_payload = {
            "index": 0,
            "shortname": "song_0",
            "decorated_shortname": "song_0",
            "prompt": "prompt 0",
            "stitched": [],
            "checkpoint": [],
            "audio": Validation._serialise_media_list([torch.zeros(1, 4)]),
        }
        rank1_payload = {
            "index": 1,
            "shortname": "song_1",
            "decorated_shortname": "song_1",
            "prompt": "prompt 1",
            "stitched": [],
            "checkpoint": [],
            "audio": Validation._serialise_media_list([torch.ones(1, 4)]),
        }
        validation._execute_validation_work_item = MagicMock(return_value=rank0_payload)

        with (
            patch("simpletuner.helpers.training.validation.gather_object", return_value=[[rank0_payload], [rank1_payload]]),
            patch("simpletuner.helpers.training.validation.validation_audio.save_audio") as save_audio,
            patch("simpletuner.helpers.training.validation.validation_audio.log_audio_to_webhook"),
            patch("simpletuner.helpers.training.validation.validation_audio.log_audio_to_trackers"),
        ):
            validation.process_prompts(validation_type="intermediary")

        self.assertEqual(sorted(validation.validation_audios.keys()), ["song_0", "song_1"])
        torch.testing.assert_close(validation.validation_audios["song_0"][0], torch.zeros(1, 4))
        torch.testing.assert_close(validation.validation_audios["song_1"][0], torch.ones(1, 4))
        self.assertEqual(save_audio.call_count, 2)


if __name__ == "__main__":
    unittest.main()
