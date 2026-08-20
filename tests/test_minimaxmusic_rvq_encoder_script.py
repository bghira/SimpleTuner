import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn as nn
from huggingface_hub.utils import EntryNotFoundError
from safetensors.torch import load_file, save_file

from scripts.train_minimax_music_rvq_encoder import (
    DAV_HOP_SAMPLES,
    DEFAULT_CODEBOOK_VOCAB_SIZES,
    EVALUATION_SECTION_START,
    HUB_CHECKPOINT_ALLOW_PATTERNS,
    MERT_ALIGNMENT_VERSION,
    MERT_CACHE_FORMAT,
    MERT_HIDDEN_SIZE,
    MUP_DEPTH_SCOPE,
    RVQ_CACHE_FORMAT,
    EvaluationCheckpoint,
    HubCheckpointUploader,
    MiniMaxMusicRVQEncoder,
    RVQEncoderConfig,
    RVQEncoderMERTTrainingModel,
    RVQTraceRecord,
    RVQWindowDataset,
    _cache_paths,
    _mert_cache_paths,
    _mert_cache_satisfies_requirements,
    _summary_checkpoints,
    apply_mup_base_shapes,
    assert_public_text_safe,
    build_parser,
    build_pool_matrix,
    chunk_stitching_frame_latent_starts,
    collate_rvq_windows,
    create_optimizer,
    discover_evaluation_checkpoints,
    evaluate,
    extract_mert_features,
    frame_latent_starts,
    init_trackers,
    interpolate_features_at_times,
    legacy_frame_latent_starts,
    load_evaluation_model,
    log_tracker_metrics,
    mert_alignment_weight_at_step,
    mert_cosine_alignment_loss,
    parse_training_progress_log,
    prepare_hub_checkpoint_uploader,
    publish_evaluation_artifacts,
    resolve_hub_model_id,
    rvq_accuracy_counts,
    rvq_frame_center_seconds,
    rvq_head_losses,
    rvq_loss,
    rvq_topk_head_accuracy_counts,
    rvq_topk_kl_losses,
    save_checkpoint,
    tracker_config,
    train,
    update_evaluation_section,
    validate_resume_training_topology,
    wait_for_all_processes,
)


def _record(alignment: dict, emitted_frames: int = 6) -> RVQTraceRecord:
    return RVQTraceRecord(
        shard_id=7,
        shard_path="data/00000/shard-000007.zip",
        sample_id="sample-7",
        audio_file="sample/audio.flac",
        tensor_file="sample/prediction.safetensors",
        split="train",
        emitted_frames=emitted_frames,
        sampling_rate=44100,
        codebook_vocab_sizes=DEFAULT_CODEBOOK_VOCAB_SIZES,
        alignment=alignment,
    )


def _fake_mup_module(calls: list[dict]) -> types.ModuleType:
    module = types.ModuleType("mup")

    class FakeMuReadout(nn.Linear):
        def __init__(self, *args, readout_zero_init=False, output_mult=1.0, **kwargs):
            self.readout_zero_init = readout_zero_init
            self.output_mult = output_mult
            super().__init__(*args, **kwargs)

    def set_base_shapes(model, base, rescale_params=True, delta=None, savefile=None, do_assert=True):
        calls.append(
            {
                "model": model,
                "base": base,
                "delta": delta,
                "rescale_params": rescale_params,
                "savefile": savefile,
                "do_assert": do_assert,
            }
        )
        for parameter in model.parameters():
            parameter.infshape = object()
        return model

    def fake_optimizer(params, **kwargs):
        calls.append({"optimizer_params": list(params), "optimizer_kwargs": kwargs})
        return {"params": params, "kwargs": kwargs}

    def save_base_shapes(model, file):
        calls.append({"save_base_shapes_model": model, "save_base_shapes_file": file})
        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake base shapes", encoding="utf-8")

    module.MuReadout = FakeMuReadout
    module.MuAdam = fake_optimizer
    module.MuAdamW = fake_optimizer
    module.MuSGD = fake_optimizer
    module.set_base_shapes = set_base_shapes
    module.save_base_shapes = save_base_shapes
    return module


class MiniMaxMusicRVQEncoderScriptTests(unittest.TestCase):
    @unittest.skipUnless(int(os.environ.get("WORLD_SIZE", "1")) > 1, "requires a torchrun multi-process launch")
    def test_mert_zero_weight_backward_works_with_ddp(self):
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")
        try:
            encoder = MiniMaxMusicRVQEncoder(
                RVQEncoderConfig(
                    codebook_vocab_sizes=(17, 5, 5, 5, 5, 5, 5, 5),
                    d_model=16,
                    num_layers=1,
                    num_heads=2,
                    ff_mult=2,
                    dropout=0.0,
                    max_position_embeddings=4,
                )
            )
            model = nn.parallel.DistributedDataParallel(RVQEncoderMERTTrainingModel(encoder, student_layer=0))
            latents = torch.randn(2, 8, 128)
            pool = torch.zeros(2, 4, 8)
            for index in range(4):
                pool[:, index, index * 2 : index * 2 + 2] = 0.5
            for _ in range(2):
                model.zero_grad(set_to_none=True)
                logits, projected = model(latents, pool)
                rvq_term = sum(head.float().mean() for head in logits)
                mert_term = mert_cosine_alignment_loss(projected, torch.randn_like(projected))
                (rvq_term + 0.0 * mert_term).backward()
                self.assertIsNotNone(model.module.mert_projection.weight.grad)
            dist.barrier()
        finally:
            dist.destroy_process_group()

    @unittest.skipUnless(int(os.environ.get("WORLD_SIZE", "1")) > 1, "requires a torchrun multi-process launch")
    def test_depth_decoder_teacher_forcing_works_with_ddp(self):
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")
        try:
            config = RVQEncoderConfig(
                codebook_vocab_sizes=(17, 5, 5, 5, 5, 5, 5, 5),
                d_model=16,
                num_layers=1,
                num_heads=2,
                ff_mult=2,
                dropout=0.0,
                max_position_embeddings=4,
                depth_decoder=True,
                depth_decoder_dim=16,
                depth_decoder_layers=1,
                depth_decoder_heads=2,
                depth_decoder_ff_mult=2,
                depth_decoder_dropout=0.0,
            )
            model = nn.parallel.DistributedDataParallel(MiniMaxMusicRVQEncoder(config))
            latents = torch.randn(2, 8, 128)
            pool = torch.zeros(2, 4, 8)
            for index in range(4):
                pool[:, index, index * 2 : index * 2 + 2] = 0.5
            target = torch.randint(0, 5, (2, 4, 8))
            target[:, :, 0] = torch.randint(0, 17, (2, 4))
            for _ in range(2):
                model.zero_grad(set_to_none=True)
                logits = model(latents, pool, target)
                rvq_loss(logits, target, teacher_kl_weight=0.0)[0].backward()
                self.assertTrue(all(parameter.grad is not None for parameter in model.module.parameters()))
            dist.barrier()
        finally:
            dist.destroy_process_group()

    def test_legacy_frame_latent_starts_match_known_stitched_hop(self):
        starts = legacy_frame_latent_starts(4500)

        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[100], 345)
        self.assertEqual(starts[200], 690)
        self.assertEqual(starts[4500], 15524)

    def test_chunk_stitching_alignment_uses_owned_chunk_spans(self):
        chunks = [
            {
                "chunk_index": 0,
                "semantic_frame_start": 0,
                "semantic_frame_end_exclusive": 200,
                "stitched_flow_latent_start": 0,
                "stitched_flow_latent_end_exclusive": 431,
            },
            {
                "chunk_index": 1,
                "semantic_frame_start": 100,
                "semantic_frame_end_exclusive": 300,
                "stitched_flow_latent_start": 431,
                "stitched_flow_latent_end_exclusive": 776,
            },
        ]

        starts = chunk_stitching_frame_latent_starts(225, chunks)

        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[125], 431)
        self.assertEqual(starts[225], 776)
        self.assertTrue(all(right >= left for left, right in zip(starts[:-1], starts[1:])))

    def test_pool_matrix_averages_each_semantic_frame(self):
        pool = build_pool_matrix([10, 12, 15, 16])

        self.assertEqual(tuple(pool.shape), (3, 6))
        self.assertTrue(torch.allclose(pool.sum(dim=1), torch.ones(3)))
        self.assertTrue(torch.allclose(pool[0, :2], torch.full((2,), 0.5)))
        self.assertTrue(torch.allclose(pool[1, 2:5], torch.full((3,), 1 / 3)))

    def test_rvq_frame_centers_use_512_sample_dav_hop(self):
        alignment = {
            "chunk_stitching": [
                {
                    "chunk_index": 0,
                    "semantic_frame_start": 0,
                    "semantic_frame_end_exclusive": 2,
                    "stitched_flow_latent_start": 0,
                    "stitched_flow_latent_end_exclusive": 7,
                }
            ]
        }
        record = _record(alignment, emitted_frames=2)

        centers = rvq_frame_center_seconds(record)

        self.assertEqual(DAV_HOP_SAMPLES, 512)
        expected = torch.tensor([2.0, 5.5], dtype=torch.float64) * 512 / 44100
        self.assertTrue(torch.allclose(centers, expected))

    def test_mert_interpolation_uses_timestamp_centers(self):
        source_times = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        features = torch.tensor([[0.0], [10.0], [20.0]])

        aligned = interpolate_features_at_times(
            features,
            source_times,
            torch.tensor([0.25, 1.5], dtype=torch.float64),
            tolerance_seconds=0.1,
        )

        self.assertTrue(torch.allclose(aligned[:, 0], torch.tensor([2.5, 15.0])))

    def test_mert_chunk_cache_extracts_multiple_layers_in_one_forward_pass(self):
        class Processor:
            def __call__(self, chunks, **kwargs):
                max_samples = max(len(chunk) for chunk in chunks)
                values = torch.zeros(len(chunks), max_samples)
                mask = torch.zeros(len(chunks), max_samples, dtype=torch.long)
                for index, chunk in enumerate(chunks):
                    values[index, : len(chunk)] = torch.from_numpy(chunk)
                    mask[index, : len(chunk)] = 1
                return {"input_values": values, "attention_mask": mask}

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.forward_calls = 0

            def _get_feat_extract_output_lengths(self, lengths):
                return torch.div(lengths, 320, rounding_mode="floor")

            def forward(self, input_values, attention_mask, output_hidden_states):
                self.forward_calls += 1
                frames = input_values.shape[1] // 320
                states = tuple(
                    torch.full((input_values.shape[0], frames, MERT_HIDDEN_SIZE), float(layer)) for layer in range(13)
                )
                return types.SimpleNamespace(hidden_states=states)

        model = Model()
        features, times = extract_mert_features(
            torch.zeros(1, 48000),
            processor=Processor(),
            model=model,
            layers=(6, 9, 12),
            chunk_seconds=1.0,
            overlap_seconds=0.2,
            batch_size=8,
            device=torch.device("cpu"),
        )

        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(set(features), {6, 9, 12})
        self.assertEqual({tensor.shape[0] for tensor in features.values()}, {times.numel()})
        self.assertTrue(torch.all(times[1:] > times[:-1]))
        self.assertTrue(torch.all(features[9] == 9))

    def test_mert_weight_holds_then_decays_to_zero(self):
        self.assertEqual(mert_alignment_weight_at_step(0.5, 60, 100, decay_start=0.7, decay_end=0.9), 0.5)
        self.assertAlmostEqual(
            mert_alignment_weight_at_step(0.5, 80, 100, decay_start=0.7, decay_end=0.9),
            0.25,
        )
        self.assertEqual(mert_alignment_weight_at_step(0.5, 95, 100, decay_start=0.7, decay_end=0.9), 0.0)

    def test_mert_cache_rejects_stale_alignment_geometry(self):
        args = build_parser().parse_args(["--mert_alignment_weight", "0.5"])
        record = _record({}, emitted_frames=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            tensor_path, meta_path = _mert_cache_paths(cache_dir, record)
            tensor_path.parent.mkdir(parents=True)
            save_file({"mert_layer_9": torch.zeros(2, MERT_HIDDEN_SIZE)}, str(tensor_path))
            metadata = {
                "format": MERT_CACHE_FORMAT,
                "alignment_version": MERT_ALIGNMENT_VERSION,
                "dav_hop_samples": 128,
                "dav_sample_rate": 44100,
                "model_name_or_path": args.mert_model_name_or_path,
                "revision": args.mert_revision,
                "layers": [9],
                "sample_rate": 24000,
                "feature_rate": 75.0,
                "hidden_size": MERT_HIDDEN_SIZE,
                "chunk_seconds": 5.0,
                "chunk_overlap_seconds": 1.0,
                "cache_dtype": "bf16",
                "emitted_frames": 2,
            }
            meta_path.write_text(json.dumps(metadata), encoding="utf-8")

            self.assertFalse(_mert_cache_satisfies_requirements(args, cache_dir, record))
            metadata["dav_hop_samples"] = DAV_HOP_SAMPLES
            meta_path.write_text(json.dumps(metadata), encoding="utf-8")
            self.assertTrue(_mert_cache_satisfies_requirements(args, cache_dir, record))

    def test_cache_dataset_applies_priming_code_offset_and_collates_variable_latents(self):
        alignment = {
            "emitted_code_row_offset": 1,
            "chunk_stitching": [
                {
                    "chunk_index": 0,
                    "semantic_frame_start": 0,
                    "semantic_frame_end_exclusive": 6,
                    "stitched_flow_latent_start": 0,
                    "stitched_flow_latent_end_exclusive": 12,
                }
            ],
        }
        record = _record(alignment)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            tensor_path, meta_path = _cache_paths(cache_dir, record)
            tensor_path.parent.mkdir(parents=True)
            latents = torch.arange(12 * 128, dtype=torch.float32).reshape(12, 128)
            codes = torch.arange(7 * 8, dtype=torch.int16).reshape(7, 8)
            teacher_topk_ids = torch.arange(7 * 8 * 3, dtype=torch.int32).reshape(7, 8, 3) % 5
            teacher_topk_logits = torch.randn(7, 8, 3, dtype=torch.float32)
            save_file(
                {
                    "latents": latents,
                    "codes": codes,
                    "teacher_topk_ids": teacher_topk_ids,
                    "teacher_topk_logits": teacher_topk_logits,
                },
                str(tensor_path),
            )
            with meta_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "format": RVQ_CACHE_FORMAT,
                        "latent_frames": 12,
                        "latent_channels": 128,
                        "code_frames": 7,
                        "has_teacher_topk": True,
                        "teacher_topk_k": 3,
                        "alignment_source": "chunk_stitching",
                    },
                    handle,
                )

            dataset = RVQWindowDataset(
                [record],
                cache_dir=cache_dir,
                window_frames=4,
                window_stride=4,
                random_crop=False,
                require_exact_alignment=True,
                require_teacher_topk=True,
            )
            sample = dataset[0]
            batch = collate_rvq_windows([sample, sample])

        self.assertEqual(tuple(sample["latents"].shape), (8, 128))
        self.assertTrue(torch.equal(sample["target"], codes[1:5].long()))
        self.assertTrue(torch.equal(sample["teacher_topk_ids"], teacher_topk_ids[1:5].long()))
        self.assertTrue(torch.equal(sample["teacher_topk_logits"], teacher_topk_logits[1:5]))
        self.assertEqual(tuple(batch["latents"].shape), (2, 8, 128))
        self.assertEqual(tuple(batch["pool"].shape), (2, 4, 8))
        self.assertEqual(tuple(batch["target"].shape), (2, 4, 8))
        self.assertEqual(tuple(batch["teacher_topk_ids"].shape), (2, 4, 8, 3))
        self.assertEqual(tuple(batch["teacher_topk_logits"].shape), (2, 4, 8, 3))

    def test_mert_window_uses_frame_timeline_without_code_offset(self):
        alignment = {
            "emitted_code_row_offset": 1,
            "chunk_stitching": [
                {
                    "chunk_index": 0,
                    "semantic_frame_start": 0,
                    "semantic_frame_end_exclusive": 6,
                    "stitched_flow_latent_start": 0,
                    "stitched_flow_latent_end_exclusive": 12,
                }
            ],
        }
        record = _record(alignment)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            latent_dir = root / "latent"
            mert_dir = root / "mert"
            tensor_path, meta_path = _cache_paths(latent_dir, record)
            tensor_path.parent.mkdir(parents=True)
            codes = torch.arange(7 * 8, dtype=torch.int16).reshape(7, 8)
            save_file({"latents": torch.zeros(12, 128), "codes": codes}, str(tensor_path))
            meta_path.write_text(
                json.dumps(
                    {
                        "format": RVQ_CACHE_FORMAT,
                        "latent_frames": 12,
                        "code_frames": 7,
                        "has_teacher_topk": False,
                        "alignment_source": "chunk_stitching",
                    }
                ),
                encoding="utf-8",
            )
            mert_tensor_path, mert_meta_path = _mert_cache_paths(mert_dir, record)
            mert_tensor_path.parent.mkdir(parents=True)
            mert = torch.arange(6 * MERT_HIDDEN_SIZE, dtype=torch.float32).reshape(6, MERT_HIDDEN_SIZE)
            save_file({"mert_layer_9": mert}, str(mert_tensor_path))
            mert_meta_path.write_text(
                json.dumps({"format": MERT_CACHE_FORMAT, "layers": [9], "emitted_frames": 6}),
                encoding="utf-8",
            )
            dataset = RVQWindowDataset(
                [record],
                cache_dir=latent_dir,
                window_frames=4,
                window_stride=4,
                random_crop=False,
                require_exact_alignment=True,
                mert_cache_dir=mert_dir,
                mert_teacher_layer=9,
                require_mert_features=True,
            )

            sample = dataset[0]
            batch = collate_rvq_windows([sample, sample])

        self.assertTrue(torch.equal(sample["target"], codes[1:5].long()))
        self.assertTrue(torch.equal(sample["mert_features"], mert[:4]))
        self.assertEqual(tuple(batch["mert_features"].shape), (2, 4, MERT_HIDDEN_SIZE))

    def test_collate_rejects_mixed_mert_batch(self):
        sample = {
            "latents": torch.zeros(2, 128),
            "pool": torch.eye(2),
            "target": torch.zeros(2, 8, dtype=torch.long),
        }
        with_mert = {**sample, "mert_features": torch.zeros(2, MERT_HIDDEN_SIZE)}

        with self.assertRaisesRegex(ValueError, "mixed RVQ batch with MERT"):
            collate_rvq_windows([sample, with_mert])

    def test_model_heads_and_loss_follow_codebook_vocab_sizes(self):
        config = RVQEncoderConfig(
            codebook_vocab_sizes=(17, 5, 5, 5, 5, 5, 5, 5),
            d_model=16,
            num_layers=1,
            num_heads=2,
            ff_mult=2,
            dropout=0.0,
            max_position_embeddings=4,
        )
        model = MiniMaxMusicRVQEncoder(config)
        latents = torch.randn(2, 8, 128)
        pool = torch.zeros(2, 4, 8)
        pool[:, 0, 0:2] = 0.5
        pool[:, 1, 2:4] = 0.5
        pool[:, 2, 4:6] = 0.5
        pool[:, 3, 6:8] = 0.5
        target = torch.zeros(2, 4, 8, dtype=torch.long)
        target[:, :, 0] = torch.randint(0, 17, (2, 4))
        target[:, :, 1:] = torch.randint(0, 5, (2, 4, 7))

        logits = model(latents, pool)
        losses = rvq_head_losses(logits, target)
        loss = sum(losses) / len(losses)
        loss.backward()
        semantic_correct, semantic_total, acoustic_correct, acoustic_total = rvq_accuracy_counts(logits, target)

        self.assertEqual([head.shape[-1] for head in logits], [17, 5, 5, 5, 5, 5, 5, 5])
        self.assertEqual(len(losses), 8)
        self.assertEqual(int(semantic_total.item()), 8)
        self.assertEqual(int(acoustic_total.item()), 56)
        self.assertGreaterEqual(int(semantic_correct.item()), 0)
        self.assertGreaterEqual(int(acoustic_correct.item()), 0)

    def test_depth_decoder_conditions_each_acoustic_head_only_on_prior_codes(self):
        torch.manual_seed(7)
        model = MiniMaxMusicRVQEncoder(
            RVQEncoderConfig(
                codebook_vocab_sizes=(17, 5, 5, 5, 5, 5, 5, 5),
                d_model=16,
                num_layers=1,
                num_heads=2,
                ff_mult=2,
                dropout=0.0,
                max_position_embeddings=4,
                depth_decoder=True,
                depth_decoder_dim=16,
                depth_decoder_layers=2,
                depth_decoder_heads=2,
                depth_decoder_ff_mult=2,
                depth_decoder_dropout=0.0,
            )
        ).eval()
        latents = torch.randn(2, 8, 128)
        pool = torch.zeros(2, 4, 8)
        for index in range(4):
            pool[:, index, index * 2 : index * 2 + 2] = 0.5
        target_a = torch.zeros((2, 4, 8), dtype=torch.long)
        target_b = target_a.clone()
        target_b[:, :, 1] = 1

        logits_a = model(latents, pool, target_a)
        logits_b = model(latents, pool, target_b)
        free_running = model(latents, pool)
        loss = rvq_loss(logits_a, target_a, teacher_kl_weight=0.0)[0]
        loss.backward()

        self.assertEqual([tuple(head.shape) for head in free_running], [(2, 4, 17), *[(2, 4, 5)] * 7])
        self.assertTrue(torch.equal(logits_a[0], logits_b[0]))
        self.assertTrue(torch.allclose(logits_a[1], logits_b[1]))
        self.assertFalse(torch.allclose(logits_a[2], logits_b[2]))
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_depth_validation_separates_free_running_and_teacher_forced_accuracy(self):
        class AcceleratorStub:
            device = torch.device("cpu")
            mixed_precision = "no"

            def gather(self, tensor):
                return tensor

            def unwrap_model(self, model, keep_fp32_wrapper=True):
                return model

        class ControlledDepthEncoder(MiniMaxMusicRVQEncoder):
            def forward(self, latents, pool, teacher_forcing_targets=None):
                batch, frames = pool.shape[:2]
                logits = []
                for index, vocab_size in enumerate(self.config.codebook_vocab_sizes):
                    head = torch.zeros((batch, frames, vocab_size), device=latents.device)
                    if teacher_forcing_targets is not None and index > 0:
                        head.scatter_(-1, teacher_forcing_targets[:, :, index].unsqueeze(-1), 10.0)
                    else:
                        head[:, :, 0] = 10.0
                    logits.append(head)
                return logits

        config = RVQEncoderConfig(
            codebook_vocab_sizes=(17, 5, 5, 5, 5, 5, 5, 5),
            d_model=16,
            num_layers=1,
            num_heads=2,
            ff_mult=2,
            dropout=0.0,
            max_position_embeddings=4,
            depth_decoder=True,
            depth_decoder_dim=16,
            depth_decoder_layers=1,
            depth_decoder_heads=2,
            depth_decoder_ff_mult=2,
            depth_decoder_dropout=0.0,
        )
        model = ControlledDepthEncoder(config)
        target = torch.ones((1, 4, 8), dtype=torch.long)
        target[:, :, 0] = 0
        batch = {
            "latents": torch.randn(1, 8, 128),
            "pool": torch.zeros(1, 4, 8),
            "target": target,
        }
        metrics = evaluate(
            AcceleratorStub(),
            model,
            [batch],
            max_batches=0,
            teacher_kl_weight=0.0,
            teacher_kl_temperature=1.0,
        )

        self.assertEqual(metrics["semantic_top1"], 1.0)
        self.assertEqual(metrics["acoustic_top1"], 0.0)
        self.assertEqual(metrics["teacher_forced_acoustic_top1"], 1.0)
        self.assertEqual(metrics["head_7_top1"], 0.0)
        self.assertEqual(metrics["teacher_forced_head_7_top1"], 1.0)

    def test_mert_wrapper_projects_intermediate_features_with_nonzero_initialization(self):
        encoder = MiniMaxMusicRVQEncoder(
            RVQEncoderConfig(
                codebook_vocab_sizes=(17, 5, 5, 5, 5, 5, 5, 5),
                d_model=16,
                num_layers=2,
                num_heads=2,
                ff_mult=2,
                dropout=0.0,
                max_position_embeddings=4,
            )
        )
        model = RVQEncoderMERTTrainingModel(encoder, student_layer=0)
        latents = torch.randn(2, 8, 128)
        pool = torch.zeros(2, 4, 8)
        for index in range(4):
            pool[:, index, index * 2 : index * 2 + 2] = 0.5

        logits, projected = model(latents, pool)
        loss = mert_cosine_alignment_loss(projected, torch.randn_like(projected)) * 0.0
        loss.backward()

        self.assertEqual(len(logits), 8)
        self.assertEqual(tuple(projected.shape), (2, 4, MERT_HIDDEN_SIZE))
        self.assertGreater(torch.count_nonzero(model.mert_projection.weight).item(), 0)
        self.assertIsNotNone(model.mert_projection.weight.grad)
        self.assertEqual(torch.count_nonzero(model.mert_projection.weight.grad).item(), 0)

    def test_mup_model_uses_mu_readouts_and_custom_attention(self):
        calls = []
        fake_mup = _fake_mup_module(calls)
        config = RVQEncoderConfig(
            codebook_vocab_sizes=(17, 5, 5, 5, 5, 5, 5, 5),
            d_model=16,
            num_layers=1,
            num_heads=2,
            ff_mult=2,
            dropout=0.0,
            max_position_embeddings=4,
            mup=True,
            mup_output_mult=0.75,
            mup_readout_zero_init=True,
        )

        with patch.dict(sys.modules, {"mup": fake_mup}):
            model = MiniMaxMusicRVQEncoder(config)

        self.assertIsInstance(model.transformer, nn.ModuleList)
        self.assertIsInstance(model.heads[0], fake_mup.MuReadout)
        self.assertEqual(model.heads[0].output_mult, 0.75)
        self.assertTrue(model.heads[0].readout_zero_init)

    def test_apply_mup_base_shapes_builds_base_and_delta_models(self):
        calls = []
        fake_mup = _fake_mup_module(calls)
        config = RVQEncoderConfig(
            codebook_vocab_sizes=(17, 5, 5, 5, 5, 5, 5, 5),
            d_model=64,
            num_layers=1,
            num_heads=4,
            ff_mult=2,
            dropout=0.0,
            max_position_embeddings=4,
            mup=True,
        )
        args = build_parser().parse_args(
            [
                "--mup",
                "--mup_base_d_model",
                "16",
                "--mup_delta_d_model",
                "32",
                "--mup_base_heads",
                "1",
                "--mup_delta_heads",
                "2",
                "--mup_save_base_shapes",
                "rvq.bsh",
            ]
        )

        with patch.dict(sys.modules, {"mup": fake_mup}):
            model = MiniMaxMusicRVQEncoder(config)
            apply_mup_base_shapes(model, config, args)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["base"].config.d_model, 16)
        self.assertEqual(calls[0]["base"].config.num_heads, 1)
        self.assertEqual(calls[0]["delta"].config.d_model, 32)
        self.assertEqual(calls[0]["delta"].config.num_heads, 2)
        self.assertEqual(calls[0]["savefile"], "rvq.bsh")

    def test_mup_base_shapes_include_mert_training_projection(self):
        calls = []
        fake_mup = _fake_mup_module(calls)
        config = RVQEncoderConfig(
            codebook_vocab_sizes=(17, 5, 5, 5, 5, 5, 5, 5),
            d_model=32,
            num_layers=1,
            num_heads=2,
            ff_mult=2,
            dropout=0.0,
            max_position_embeddings=4,
            mup=True,
        )
        args = build_parser().parse_args(
            [
                "--mup",
                "--mert_alignment_weight",
                "0.5",
                "--mert_student_layer",
                "0",
                "--mup_base_d_model",
                "8",
                "--mup_delta_d_model",
                "16",
                "--mup_base_heads",
                "1",
                "--mup_delta_heads",
                "1",
            ]
        )

        with patch.dict(sys.modules, {"mup": fake_mup}):
            model = RVQEncoderMERTTrainingModel(MiniMaxMusicRVQEncoder(config), student_layer=0)
            apply_mup_base_shapes(model, config, args)

        self.assertIsInstance(calls[0]["base"], RVQEncoderMERTTrainingModel)
        self.assertIsInstance(calls[0]["delta"], RVQEncoderMERTTrainingModel)
        self.assertTrue(hasattr(model.mert_projection.weight, "infshape"))

    def test_mup_base_shapes_include_depth_decoder_context_projection(self):
        calls = []
        fake_mup = _fake_mup_module(calls)
        config = RVQEncoderConfig(
            codebook_vocab_sizes=(17, 5, 5, 5, 5, 5, 5, 5),
            d_model=32,
            num_layers=1,
            num_heads=2,
            ff_mult=2,
            dropout=0.0,
            max_position_embeddings=4,
            mup=True,
            depth_decoder=True,
            depth_decoder_dim=16,
            depth_decoder_layers=1,
            depth_decoder_heads=2,
            depth_decoder_ff_mult=2,
            depth_decoder_dropout=0.0,
        )
        args = build_parser().parse_args(
            [
                "--mup",
                "--depth_decoder",
                "--depth_decoder_dim",
                "16",
                "--depth_decoder_layers",
                "1",
                "--depth_decoder_heads",
                "2",
                "--depth_decoder_ff_mult",
                "2",
                "--depth_decoder_dropout",
                "0",
                "--mup_base_d_model",
                "8",
                "--mup_delta_d_model",
                "16",
                "--mup_base_heads",
                "1",
                "--mup_delta_heads",
                "1",
            ]
        )

        with patch.dict(sys.modules, {"mup": fake_mup}):
            model = MiniMaxMusicRVQEncoder(config)
            apply_mup_base_shapes(model, config, args)

        self.assertTrue(calls[0]["base"].config.depth_decoder)
        self.assertTrue(calls[0]["delta"].config.depth_decoder)
        self.assertEqual(calls[0]["base"].config.depth_decoder_dim, 16)
        self.assertTrue(hasattr(model.depth_decoder.context_projection.weight, "infshape"))

    def test_mert_mup_rejects_encoder_only_base_shapes(self):
        calls = []
        fake_mup = _fake_mup_module(calls)
        config = RVQEncoderConfig(
            codebook_vocab_sizes=(17, 5, 5, 5, 5, 5, 5, 5),
            d_model=16,
            num_layers=1,
            num_heads=2,
            ff_mult=2,
            dropout=0.0,
            max_position_embeddings=4,
            mup=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            shape_path = Path(tmpdir) / "encoder-only.bsh"
            shape_path.write_text("fake", encoding="utf-8")
            args = build_parser().parse_args(
                [
                    "--mup",
                    "--mup_base_shapes",
                    str(shape_path),
                    "--mert_alignment_weight",
                    "0.5",
                    "--mert_student_layer",
                    "0",
                ]
            )
            with patch.dict(sys.modules, {"mup": fake_mup}):
                model = RVQEncoderMERTTrainingModel(MiniMaxMusicRVQEncoder(config), student_layer=0)
                with self.assertRaisesRegex(ValueError, "Encoder-only v1/v2"):
                    apply_mup_base_shapes(model, config, args)

    def test_depth_decoder_mup_rejects_encoder_only_base_shapes(self):
        calls = []
        fake_mup = _fake_mup_module(calls)
        config = RVQEncoderConfig(
            codebook_vocab_sizes=(17, 5, 5, 5, 5, 5, 5, 5),
            d_model=16,
            num_layers=1,
            num_heads=2,
            ff_mult=2,
            dropout=0.0,
            max_position_embeddings=4,
            mup=True,
            depth_decoder=True,
            depth_decoder_dim=16,
            depth_decoder_layers=1,
            depth_decoder_heads=2,
            depth_decoder_ff_mult=2,
            depth_decoder_dropout=0.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            shape_path = Path(tmpdir) / "encoder-only.bsh"
            shape_path.write_text("fake", encoding="utf-8")
            args = build_parser().parse_args(
                [
                    "--mup",
                    "--mup_base_shapes",
                    str(shape_path),
                    "--depth_decoder",
                    "--depth_decoder_dim",
                    "16",
                    "--depth_decoder_layers",
                    "1",
                    "--depth_decoder_heads",
                    "2",
                    "--depth_decoder_ff_mult",
                    "2",
                    "--depth_decoder_dropout",
                    "0",
                ]
            )
            with patch.dict(sys.modules, {"mup": fake_mup}):
                model = MiniMaxMusicRVQEncoder(config)
                with self.assertRaisesRegex(ValueError, "requires shape metadata"):
                    apply_mup_base_shapes(model, config, args)

    def test_mup_optimizer_rejects_unvalidated_optimizer(self):
        calls = []
        fake_mup = _fake_mup_module(calls)
        args = build_parser().parse_args(["--mup"])
        model = MiniMaxMusicRVQEncoder(
            RVQEncoderConfig(
                codebook_vocab_sizes=(17, 5, 5, 5, 5, 5, 5, 5),
                d_model=16,
                num_layers=1,
                num_heads=2,
                ff_mult=2,
                dropout=0.0,
                max_position_embeddings=4,
            )
        )

        with patch.dict(sys.modules, {"mup": fake_mup}):
            with self.assertRaisesRegex(ValueError, "torch-adamw"):
                create_optimizer(args, model)

    def test_polynomial_scheduler_is_the_default(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.lr_scheduler, "polynomial")
        self.assertEqual(args.lr_warmup_steps, 500)
        self.assertEqual(args.lr_power, 1.0)

    def test_resume_rejects_depth_decoder_topology_changes(self):
        enabled = build_parser().parse_args(["--depth_decoder"])
        with self.assertRaisesRegex(ValueError, "cannot change whether the autoregressive depth decoder"):
            validate_resume_training_topology(enabled, {"train_args": {"depth_decoder": False}})

        changed_width = build_parser().parse_args(["--depth_decoder", "--depth_decoder_dim", "256"])
        previous = {
            "train_args": {
                "depth_decoder": True,
                "depth_decoder_dim": 512,
                "depth_decoder_layers": 2,
                "depth_decoder_heads": 8,
                "depth_decoder_ff_mult": 4,
                "mert_alignment_weight": 0.0,
            }
        }
        with self.assertRaisesRegex(ValueError, "depth-decoder topology fields"):
            validate_resume_training_topology(changed_width, previous)

    def test_mert_chunk_help_explains_fidelity_throughput_tradeoff(self):
        help_text = " ".join(build_parser().format_help().split())

        self.assertIn("matches MERT training excerpts", help_text)
        self.assertIn("improve cache throughput but can reduce representation fidelity", help_text)

    def test_wandb_tracker_uses_sanitized_config_and_namespaced_metrics(self):
        class AcceleratorStub:
            is_main_process = True

            def __init__(self):
                self.initialized = None
                self.logged = None

            def init_trackers(self, project_name, *, config, init_kwargs):
                self.initialized = (project_name, config, init_kwargs)

            def log(self, metrics, *, step):
                self.logged = (metrics, step)

        args = build_parser().parse_args(
            [
                "--report_to",
                "wandb",
                "--tracker_project_name",
                "rvq-encoder",
                "--tracker_run_name",
                "music3-155m-v2",
                "--output_dir",
                "/private/output",
                "--latent_cache_dir",
                "/private/cache",
            ]
        )
        accelerator = AcceleratorStub()

        init_trackers(args, accelerator, 154_736_064)
        log_tracker_metrics(args, accelerator, "train", {"loss": 2.5}, 20)

        project_name, config, init_kwargs = accelerator.initialized
        self.assertEqual(project_name, "rvq-encoder")
        self.assertEqual(config, tracker_config(args, 154_736_064))
        self.assertEqual(config["parameter_count"], 154_736_064)
        self.assertNotIn("output_dir", config)
        self.assertNotIn("latent_cache_dir", config)
        self.assertEqual(init_kwargs, {"wandb": {"name": "music3-155m-v2"}})
        self.assertEqual(accelerator.logged, ({"train/loss": 2.5}, 20))

        log_tracker_metrics(
            args,
            accelerator,
            "validation",
            {"semantic_top1": 0.4, "mert_cosine": 0.7},
            500,
        )
        self.assertEqual(
            accelerator.logged,
            ({"validation/semantic_top1": 0.4, "validation/mert_cosine": 0.7}, 500),
        )

    def test_wait_for_all_processes_uses_plain_barrier_for_mps_distributed(self):
        class AcceleratorStub:
            def __init__(self):
                self.waited = False

            def wait_for_everyone(self):
                self.waited = True

        accelerator = AcceleratorStub()
        with (
            patch("scripts.train_minimax_music_rvq_encoder.dist.is_available", return_value=True),
            patch("scripts.train_minimax_music_rvq_encoder.dist.is_initialized", return_value=True),
            patch("scripts.train_minimax_music_rvq_encoder.dist.barrier") as barrier,
            patch("scripts.train_minimax_music_rvq_encoder.torch.backends.mps.is_available", return_value=True),
            patch("scripts.train_minimax_music_rvq_encoder.torch.cuda.is_available", return_value=False),
        ):
            wait_for_all_processes(accelerator)

        barrier.assert_called_once_with()
        self.assertFalse(accelerator.waited)

    def test_wait_for_all_processes_uses_accelerate_barrier_when_cuda_is_available(self):
        class AcceleratorStub:
            def __init__(self):
                self.waited = False

            def wait_for_everyone(self):
                self.waited = True

        accelerator = AcceleratorStub()
        with (
            patch("scripts.train_minimax_music_rvq_encoder.dist.is_available", return_value=True),
            patch("scripts.train_minimax_music_rvq_encoder.dist.is_initialized", return_value=True),
            patch("scripts.train_minimax_music_rvq_encoder.dist.barrier") as barrier,
            patch("scripts.train_minimax_music_rvq_encoder.torch.backends.mps.is_available", return_value=True),
            patch("scripts.train_minimax_music_rvq_encoder.torch.cuda.is_available", return_value=True),
        ):
            wait_for_all_processes(accelerator)

        barrier.assert_not_called()
        self.assertTrue(accelerator.waited)

    def test_resolve_hub_model_id_accepts_repo_id_on_push_to_hub(self):
        args = build_parser().parse_args(["--push_to_hub", "https://huggingface.co/owner/rvq/tree/main/final"])

        self.assertEqual(resolve_hub_model_id(args), "owner/rvq")

    def test_resolve_hub_model_id_requires_repo_id_when_push_flag_is_boolean(self):
        args = build_parser().parse_args(["--push_to_hub"])

        with self.assertRaisesRegex(ValueError, "repo id"):
            resolve_hub_model_id(args)

    def test_prepare_hub_checkpoint_uploader_creates_model_repo_on_main_process(self):
        class AcceleratorStub:
            is_main_process = True

        class FakeApi:
            instances = []

            def __init__(self):
                self.create_calls = []
                FakeApi.instances.append(self)

            def create_repo(self, **kwargs):
                self.create_calls.append(kwargs)
                return types.SimpleNamespace(repo_id=kwargs["repo_id"])

        args = build_parser().parse_args(["--push_to_hub", "--hub_model_id", "owner/rvq", "--model_card_private"])
        with patch("scripts.train_minimax_music_rvq_encoder.HfApi", FakeApi):
            uploader = prepare_hub_checkpoint_uploader(args, AcceleratorStub(), Path("out"))

        self.assertIsNotNone(uploader)
        self.assertEqual(uploader.repo_id, "owner/rvq")
        self.assertEqual(FakeApi.instances[0].create_calls[0]["repo_type"], "model")
        self.assertTrue(FakeApi.instances[0].create_calls[0]["private"])

    def test_save_checkpoint_uploads_exported_artifacts_to_hub_path(self):
        class AcceleratorStub:
            is_main_process = True

            def __init__(self):
                self.wait_count = 0

            def save_state(self, path):
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "optimizer.bin").write_bytes(b"not uploaded")

            def unwrap_model(self, model, keep_fp32_wrapper=True):
                return model

            def wait_for_everyone(self):
                self.wait_count += 1

        class FakeApi:
            def __init__(self):
                self.uploads = []

            def upload_folder(self, **kwargs):
                self.uploads.append(kwargs)
                return types.SimpleNamespace(commit_url="https://huggingface.co/owner/rvq/commit/1")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "out"
            args = build_parser().parse_args(["--output_dir", str(output_root)])
            model = MiniMaxMusicRVQEncoder(
                RVQEncoderConfig(
                    codebook_vocab_sizes=(17, 5, 5, 5, 5, 5, 5, 5),
                    d_model=16,
                    num_layers=1,
                    num_heads=2,
                    ff_mult=2,
                    dropout=0.0,
                    max_position_embeddings=4,
                )
            )
            accelerator = AcceleratorStub()
            api = FakeApi()
            uploader = HubCheckpointUploader(
                accelerator=accelerator,
                api=api,
                repo_id="owner/rvq",
                output_root=output_root,
            )

            checkpoint = save_checkpoint(
                accelerator,
                args,
                model,
                output_root / "best",
                uploader,
                global_step=7,
                epoch=1,
                batch_in_epoch=3,
                best_validation_loss=1.25,
            )

            self.assertTrue((checkpoint / "rvq_encoder.safetensors").is_file())
            self.assertTrue((checkpoint / "trainer_state.json").is_file())
            self.assertEqual(api.uploads[0]["repo_id"], "owner/rvq")
            self.assertEqual(api.uploads[0]["repo_type"], "model")
            self.assertEqual(api.uploads[0]["path_in_repo"], "best/checkpoint-7")
            self.assertEqual(api.uploads[0]["allow_patterns"], list(HUB_CHECKPOINT_ALLOW_PATTERNS))
            self.assertNotIn("trainer_state.json", api.uploads[0]["allow_patterns"])

    def test_depth_decoder_checkpoint_round_trips_for_free_running_evaluation(self):
        class AcceleratorStub:
            is_main_process = True

            def save_state(self, path):
                Path(path).mkdir(parents=True, exist_ok=True)

            def unwrap_model(self, model, keep_fp32_wrapper=True):
                return model

            def wait_for_everyone(self):
                pass

        config = RVQEncoderConfig(
            codebook_vocab_sizes=(17, 5, 5, 5, 5, 5, 5, 5),
            d_model=16,
            num_layers=1,
            num_heads=2,
            ff_mult=2,
            dropout=0.0,
            max_position_embeddings=4,
            depth_decoder=True,
            depth_decoder_dim=16,
            depth_decoder_layers=1,
            depth_decoder_heads=2,
            depth_decoder_ff_mult=2,
            depth_decoder_dropout=0.0,
        )
        model = MiniMaxMusicRVQEncoder(config).eval()
        latents = torch.randn(1, 8, 128)
        pool = torch.zeros(1, 4, 8)
        for index in range(4):
            pool[:, index, index * 2 : index * 2 + 2] = 0.5
        expected = model(latents, pool)

        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_parser().parse_args(["--depth_decoder"])
            checkpoint_dir = save_checkpoint(
                AcceleratorStub(),
                args,
                model,
                Path(tmpdir),
                None,
                global_step=3,
                epoch=0,
                batch_in_epoch=1,
                best_validation_loss=None,
            )
            loaded = load_evaluation_model(EvaluationCheckpoint("checkpoint-3", checkpoint_dir, 3)).eval()
            actual = loaded(latents, pool)
            state = load_file(str(checkpoint_dir / "rvq_encoder.safetensors"))

        self.assertTrue(loaded.config.depth_decoder)
        self.assertTrue(any(name.startswith("depth_decoder.") for name in state))
        self.assertFalse(any(name.startswith("heads.1.") for name in state))
        for expected_head, actual_head in zip(expected, actual):
            self.assertTrue(torch.equal(expected_head, actual_head))

    def test_depth_decoder_mup_checkpoint_writes_and_validates_shape_scope(self):
        class AcceleratorStub:
            is_main_process = True

            def save_state(self, path):
                Path(path).mkdir(parents=True, exist_ok=True)

            def unwrap_model(self, model, keep_fp32_wrapper=True):
                return model

            def wait_for_everyone(self):
                pass

        calls = []
        fake_mup = _fake_mup_module(calls)
        config = RVQEncoderConfig(
            codebook_vocab_sizes=(17, 5, 5, 5, 5, 5, 5, 5),
            d_model=16,
            num_layers=1,
            num_heads=2,
            ff_mult=2,
            dropout=0.0,
            max_position_embeddings=4,
            mup=True,
            depth_decoder=True,
            depth_decoder_dim=16,
            depth_decoder_layers=1,
            depth_decoder_heads=2,
            depth_decoder_ff_mult=2,
            depth_decoder_dropout=0.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(sys.modules, {"mup": fake_mup}):
            model = MiniMaxMusicRVQEncoder(config)
            args = build_parser().parse_args(
                [
                    "--mup",
                    "--depth_decoder",
                    "--depth_decoder_dim",
                    "16",
                    "--depth_decoder_layers",
                    "1",
                    "--depth_decoder_heads",
                    "2",
                    "--depth_decoder_ff_mult",
                    "2",
                    "--depth_decoder_dropout",
                    "0",
                ]
            )
            checkpoint_dir = save_checkpoint(
                AcceleratorStub(),
                args,
                model,
                Path(tmpdir),
                None,
                global_step=4,
                epoch=0,
                batch_in_epoch=1,
                best_validation_loss=None,
            )
            metadata = json.loads((checkpoint_dir / "mup_base_shapes.bsh.meta.json").read_text())
            loaded = load_evaluation_model(EvaluationCheckpoint("checkpoint-4", checkpoint_dir, 4))

        self.assertEqual(metadata["scope"], MUP_DEPTH_SCOPE)
        self.assertEqual(metadata["depth_decoder_dim"], 16)
        self.assertTrue(loaded.config.depth_decoder)

    def test_mert_checkpoint_export_contains_only_encoder_weights(self):
        class AcceleratorStub:
            is_main_process = True

            def save_state(self, path):
                Path(path).mkdir(parents=True, exist_ok=True)

            def unwrap_model(self, model, keep_fp32_wrapper=True):
                return model

            def wait_for_everyone(self):
                pass

        encoder = MiniMaxMusicRVQEncoder(
            RVQEncoderConfig(
                codebook_vocab_sizes=(17, 5, 5, 5, 5, 5, 5, 5),
                d_model=16,
                num_layers=1,
                num_heads=2,
                ff_mult=2,
                dropout=0.0,
                max_position_embeddings=4,
            )
        )
        model = RVQEncoderMERTTrainingModel(encoder, student_layer=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_parser().parse_args(["--mert_alignment_weight", "0.5", "--mert_student_layer", "0"])
            checkpoint_dir = save_checkpoint(
                AcceleratorStub(),
                args,
                model,
                Path(tmpdir),
                None,
                global_step=1,
                epoch=0,
                batch_in_epoch=1,
                best_validation_loss=None,
            )
            state = load_file(str(checkpoint_dir / "rvq_encoder.safetensors"))

        self.assertTrue(state)
        self.assertTrue(all(not name.startswith("encoder.") for name in state))
        self.assertTrue(all("mert_projection" not in name for name in state))

    def test_mert_training_smoke_logs_alignment_metrics(self):
        record = _record({}, emitted_frames=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            latent_dir = root / "latent"
            mert_dir = root / "mert"
            output_dir = root / "output"
            tensor_path, meta_path = _cache_paths(latent_dir, record)
            tensor_path.parent.mkdir(parents=True)
            latent_frames = legacy_frame_latent_starts(record.emitted_frames)[-1]
            save_file(
                {
                    "latents": torch.randn(latent_frames, 128),
                    "codes": torch.randint(0, 5, (record.emitted_frames + 1, 8), dtype=torch.int16),
                },
                str(tensor_path),
            )
            meta_path.write_text(
                json.dumps(
                    {
                        "format": RVQ_CACHE_FORMAT,
                        "latent_frames": latent_frames,
                        "code_frames": record.emitted_frames + 1,
                        "has_teacher_topk": False,
                        "alignment_source": "legacy_nominal",
                    }
                ),
                encoding="utf-8",
            )
            mert_tensor_path, mert_meta_path = _mert_cache_paths(mert_dir, record)
            mert_tensor_path.parent.mkdir(parents=True)
            save_file({"mert_layer_9": torch.randn(4, MERT_HIDDEN_SIZE)}, str(mert_tensor_path))
            args = build_parser().parse_args(
                [
                    "--latent_cache_dir",
                    str(latent_dir),
                    "--mert_cache_dir",
                    str(mert_dir),
                    "--output_dir",
                    str(output_dir),
                    "--teacher_kl_weight",
                    "0",
                    "--mert_alignment_weight",
                    "0.5",
                    "--mert_student_layer",
                    "0",
                    "--mert_decay_start",
                    "0.1",
                    "--mert_decay_end",
                    "0.2",
                    "--window_frames",
                    "4",
                    "--window_stride",
                    "4",
                    "--d_model",
                    "16",
                    "--layers",
                    "1",
                    "--heads",
                    "2",
                    "--ff_mult",
                    "2",
                    "--dropout",
                    "0",
                    "--train_batch_size",
                    "1",
                    "--max_train_steps",
                    "2",
                    "--checkpointing_steps",
                    "0",
                    "--validation_steps",
                    "0",
                    "--log_steps",
                    "1",
                    "--mixed_precision",
                    "no",
                    "--optimizer",
                    "torch-adamw",
                ]
            )
            mert_metadata = {
                "format": MERT_CACHE_FORMAT,
                "alignment_version": MERT_ALIGNMENT_VERSION,
                "dav_hop_samples": DAV_HOP_SAMPLES,
                "dav_sample_rate": 44100,
                "model_name_or_path": args.mert_model_name_or_path,
                "revision": args.mert_revision,
                "layers": [9],
                "sample_rate": 24000,
                "feature_rate": 75.0,
                "hidden_size": MERT_HIDDEN_SIZE,
                "chunk_seconds": 5.0,
                "chunk_overlap_seconds": 1.0,
                "cache_dtype": "bf16",
                "emitted_frames": 4,
            }
            mert_meta_path.write_text(json.dumps(mert_metadata), encoding="utf-8")
            with patch(
                "scripts.train_minimax_music_rvq_encoder.load_records_for_accelerator",
                return_value=([record], []),
            ):
                final_checkpoint = train(args)

            records = [json.loads(line) for line in (output_dir / "training_metrics.jsonl").read_text().splitlines()]
            exported = load_file(str(final_checkpoint / "rvq_encoder.safetensors"))

        self.assertEqual(len(records), 2)
        self.assertTrue(all("mert_alignment_loss" in row for row in records))
        self.assertTrue(all("mert_cosine" in row for row in records))
        self.assertEqual(records[-1]["mert_weight"], 0.0)
        self.assertTrue(all("mert_projection" not in name for name in exported))

    def test_depth_decoder_training_smoke_logs_free_running_validation(self):
        record = _record({}, emitted_frames=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            latent_dir = root / "latent"
            output_dir = root / "output"
            tensor_path, meta_path = _cache_paths(latent_dir, record)
            tensor_path.parent.mkdir(parents=True)
            latent_frames = legacy_frame_latent_starts(record.emitted_frames)[-1]
            save_file(
                {
                    "latents": torch.randn(latent_frames, 128),
                    "codes": torch.randint(0, 5, (record.emitted_frames + 1, 8), dtype=torch.int16),
                },
                str(tensor_path),
            )
            meta_path.write_text(
                json.dumps(
                    {
                        "format": RVQ_CACHE_FORMAT,
                        "latent_frames": latent_frames,
                        "code_frames": record.emitted_frames + 1,
                        "has_teacher_topk": False,
                        "alignment_source": "legacy_nominal",
                    }
                ),
                encoding="utf-8",
            )
            args = build_parser().parse_args(
                [
                    "--latent_cache_dir",
                    str(latent_dir),
                    "--output_dir",
                    str(output_dir),
                    "--teacher_kl_weight",
                    "0",
                    "--depth_decoder",
                    "--depth_decoder_dim",
                    "16",
                    "--depth_decoder_layers",
                    "1",
                    "--depth_decoder_heads",
                    "2",
                    "--depth_decoder_ff_mult",
                    "2",
                    "--depth_decoder_dropout",
                    "0",
                    "--window_frames",
                    "4",
                    "--window_stride",
                    "4",
                    "--d_model",
                    "16",
                    "--layers",
                    "1",
                    "--heads",
                    "2",
                    "--ff_mult",
                    "2",
                    "--dropout",
                    "0",
                    "--train_batch_size",
                    "1",
                    "--validation_batch_size",
                    "1",
                    "--max_train_steps",
                    "2",
                    "--checkpointing_steps",
                    "0",
                    "--validation_steps",
                    "1",
                    "--log_steps",
                    "1",
                    "--mixed_precision",
                    "no",
                    "--optimizer",
                    "torch-adamw",
                ]
            )
            with patch(
                "scripts.train_minimax_music_rvq_encoder.load_records_for_accelerator",
                return_value=([record], [record]),
            ):
                final_checkpoint = train(args)

            records = [json.loads(line) for line in (output_dir / "training_metrics.jsonl").read_text().splitlines()]
            exported = load_file(str(final_checkpoint / "rvq_encoder.safetensors"))
            exported_config = json.loads((final_checkpoint / "rvq_encoder_config.json").read_text())

        validation_records = [row for row in records if row["type"] == "validation"]
        self.assertEqual(len(validation_records), 2)
        self.assertTrue(all("teacher_forced_acoustic_top1" in row for row in validation_records))
        self.assertTrue(all("head_7_top1" in row for row in validation_records))
        self.assertTrue(any(name.startswith("depth_decoder.") for name in exported))
        self.assertTrue(exported_config["depth_decoder"])

    def test_topk_kl_loss_uses_teacher_distribution(self):
        logits = [
            torch.tensor([[[2.0, 0.0, -1.0, -2.0]], [[-1.0, 1.5, 0.0, -0.5]]], requires_grad=True),
            torch.tensor([[[0.0, 1.0, -1.0]], [[1.0, -0.5, 0.25]]], requires_grad=True),
        ]
        target = torch.tensor([[[0, 1]], [[1, 0]]], dtype=torch.long)
        teacher_topk_ids = torch.tensor(
            [
                [[[0, 1], [1, 0]]],
                [[[1, 2], [0, 2]]],
            ],
            dtype=torch.long,
        )
        teacher_topk_logits = torch.tensor(
            [
                [[[0.0, -0.25], [0.0, -0.5]]],
                [[[0.0, -0.75], [0.0, -0.25]]],
            ],
            dtype=torch.float32,
        )

        kl_losses = rvq_topk_kl_losses(
            logits,
            teacher_topk_ids,
            teacher_topk_logits,
            temperature=1.0,
            target=target,
        )
        total_loss, ce_loss, kl_loss = rvq_loss(
            logits,
            target,
            teacher_topk_ids=teacher_topk_ids,
            teacher_topk_logits=teacher_topk_logits,
            teacher_kl_weight=0.25,
            teacher_kl_temperature=1.0,
        )
        total_loss.backward()

        self.assertEqual(len(kl_losses), 2)
        self.assertGreater(float(kl_loss.item()), 0.0)
        self.assertTrue(torch.allclose(total_loss, ce_loss + 0.25 * kl_loss))
        self.assertIsNotNone(logits[0].grad)
        self.assertIsNotNone(logits[1].grad)

    def test_topk_kl_weight_requires_teacher_tensors(self):
        logits = [torch.zeros(1, 1, 3), torch.zeros(1, 1, 3)]
        target = torch.zeros(1, 1, 2, dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "teacher_topk_ids"):
            rvq_loss(logits, target, teacher_kl_weight=0.25)

    def test_topk_kl_excludes_teacher_eos_and_invalid_ids(self):
        logits = [
            torch.tensor(
                [[[2.0, 0.0, -1.0, -2.0], [-1.0, 1.5, 0.0, -0.5]]],
                requires_grad=True,
            )
        ]
        target = torch.tensor([[[0], [1]]], dtype=torch.long)
        teacher_topk_ids = torch.tensor([[[[0, 4]], [[-1, 4]]]], dtype=torch.long)
        teacher_topk_logits = torch.tensor([[[[0.0, 10.0]], [[1.0, 0.0]]]], dtype=torch.float32)

        (kl_loss,) = rvq_topk_kl_losses(
            logits,
            teacher_topk_ids,
            teacher_topk_logits,
            temperature=1.0,
            target=target,
        )
        expected = -torch.log_softmax(logits[0][0, 0], dim=-1)[0]
        kl_loss.backward()

        self.assertTrue(torch.allclose(kl_loss, expected))
        self.assertIsNotNone(logits[0].grad)

    def test_frame_latent_starts_prefers_chunk_stitching_when_present(self):
        alignment = {
            "chunk_stitching": [
                {
                    "chunk_index": 0,
                    "semantic_frame_start": 0,
                    "semantic_frame_end_exclusive": 4,
                    "stitched_flow_latent_start": 0,
                    "stitched_flow_latent_end_exclusive": 8,
                }
            ]
        }

        starts, source = frame_latent_starts(4, alignment)

        self.assertEqual(source, "chunk_stitching")
        self.assertEqual(starts, [0, 2, 4, 6, 8])

    def test_topk_head_accuracy_counts_masks_padding(self):
        logits = [
            torch.tensor([[[4.0, 3.0, 2.0], [0.0, 2.0, 1.0]]]),
            torch.tensor([[[1.0, 0.0, 3.0], [3.0, 2.0, 1.0]]]),
        ]
        target = torch.tensor([[[1, 0], [-100, 1]]])

        top1_correct, top1_total = rvq_topk_head_accuracy_counts(logits, target, top_k=1)
        top2_correct, top2_total = rvq_topk_head_accuracy_counts(logits, target, top_k=2)

        self.assertTrue(torch.equal(top1_correct, torch.tensor([0, 0])))
        self.assertTrue(torch.equal(top1_total, torch.tensor([1, 2])))
        self.assertTrue(torch.equal(top2_correct, torch.tensor([1, 2])))
        self.assertTrue(torch.equal(top2_total, torch.tensor([1, 2])))

    def test_discover_evaluation_checkpoints_orders_numbered_and_final(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ("checkpoint-1000", "checkpoint-500", "best/checkpoint-500"):
                checkpoint = root / name
                checkpoint.mkdir(parents=True)
                (checkpoint / "rvq_encoder.safetensors").write_bytes(b"weights")
            final = root / "final"
            final.mkdir()
            (final / "rvq_encoder.safetensors").write_bytes(b"weights")
            (final / "trainer_state.json").write_text('{"global_step": 1125}\n', encoding="utf-8")

            checkpoints = discover_evaluation_checkpoints(root)

        self.assertEqual(
            [(item.name, item.step) for item in checkpoints],
            [("checkpoint-500", 500), ("checkpoint-1000", 1000), ("final", 1125)],
        )

    def test_parse_training_progress_log_keeps_logged_steps(self):
        log = "\n".join(
            (
                "RVQ encoder steps:  20%| 20/100 [ac=0.10, ce=4.0, kl=2.0, loss=4.5, lr=1e-4, sem=0.20]",
                "RVQ encoder steps:  21%| 21/100 [ac=0.10, ce=4.0, kl=2.0, loss=4.5, lr=1e-4, sem=0.20]",
                "RVQ encoder steps:  40%| 40/100 [ac=0.15, ce=3.5, kl=1.5, loss=3.875, lr=5e-5, sem=0.30]",
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train.log"
            path.write_text(log, encoding="utf-8")
            records = parse_training_progress_log(path, log_steps=20)

        self.assertEqual([record["step"] for record in records], [20, 40])
        self.assertEqual(records[0]["ce_loss"], 4.0)
        self.assertEqual(records[1]["learning_rate"], 5e-5)

    def test_update_evaluation_section_replaces_only_managed_content(self):
        original = "# Model\n\nBefore\n\n<!-- simpletuner-rvq-evaluation-start -->\nold\n<!-- simpletuner-rvq-evaluation-end -->\n\nAfter\n"

        updated = update_evaluation_section(original, "## Evaluation\n\nnew\n")

        self.assertIn("Before", updated)
        self.assertIn("After", updated)
        self.assertIn("## Evaluation\n\nnew", updated)
        self.assertNotIn("\nold\n", updated)
        self.assertEqual(updated.count("<!-- simpletuner-rvq-evaluation-start -->"), 1)

    def test_publish_evaluation_artifacts_creates_missing_readme(self):
        class FakeApi:
            operations = None

            def create_commit(self, **kwargs):
                self.__class__.operations = kwargs["operations"]

        args = build_parser().parse_args(
            [
                "--push_to_hub",
                "owner/model",
                "--eval_output_subdir",
                "evaluation/v4",
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = Path(tmpdir)
            (eval_dir / "evaluation-metrics.json").write_text("{}\n", encoding="utf-8")
            with (
                patch("scripts.train_minimax_music_rvq_encoder.HfApi", FakeApi),
                patch(
                    "scripts.train_minimax_music_rvq_encoder.hf_hub_download",
                    side_effect=EntryNotFoundError("missing README"),
                ),
            ):
                publish_evaluation_artifacts(args, eval_dir, "## Evaluation\n")

        self.assertIsNotNone(FakeApi.operations)
        self.assertEqual(FakeApi.operations[0].path_in_repo, "README.md")
        self.assertIn(EVALUATION_SECTION_START, FakeApi.operations[0].path_or_fileobj.getvalue().decode())

    def test_summary_checkpoints_combines_selection_labels(self):
        rows = [
            {"checkpoint": "checkpoint-500", "loss": 2.0, "semantic_top1": 0.2, "acoustic_top1": 0.1},
            {"checkpoint": "final", "loss": 1.0, "semantic_top1": 0.3, "acoustic_top1": 0.2},
        ]

        selected = _summary_checkpoints(rows)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][1]["checkpoint"], "final")
        self.assertEqual(
            selected[0][0],
            "lowest loss; best semantic top-1; best acoustic top-1; final",
        )

    def test_public_text_scan_rejects_local_identity_paths(self):
        with self.assertRaisesRegex(ValueError, "Blocked: local machine identity was found in public text"):
            assert_public_text_safe("results were written under /workspace/private-run")


if __name__ == "__main__":
    unittest.main()
