"""Run a two-rank CLIP cache and training probe against an R2 capture manifest.

From the repository root, use PYTHONPATH=. and pass --source s3://BUCKET and
--manifest KEY. Supply a JSON object with url, access_id and secret_key on stdin.
Requires two CUDA devices, the project JXL extra and access to the CLIP weights.
Only the temporary directory on the host receives writes. Use --cpu-fixture
for a local Parquet and tiny CLIP run without credentials or downloads.
"""

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from bisect import bisect_right
from collections import Counter
from contextlib import ExitStack, closing
from datetime import timedelta
from io import BytesIO
from itertools import accumulate
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.parse import urlsplit

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from simpletuner.helpers.caching.text_embeds import TextEmbeddingCache
from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
from simpletuner.helpers.data_backend.factory import FactoryRegistry, build_text_embed_store, build_text_preparation_listing
from simpletuner.helpers.data_backend.webshart import WebshartDataBackend
from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
from simpletuner.helpers.metadata.backends.webshart_epoch import EpochSpec
from simpletuner.helpers.models.common import TextEmbedCacheKey
from simpletuner.helpers.training.collate import compute_prompt_embeddings
from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.simpletuner_sdk.server.services.cache_job_service import CacheJobService, _CacheArgsNamespace


class ClipEncoder:
    def __init__(self, device, *, fixture=False):
        self.device = device
        self.calls = 0
        self.encoded_prompts = 0
        if fixture:
            torch.manual_seed(19)
            self.tokenizer = None
            self.encoder = (
                CLIPTextModel(
                    CLIPTextConfig(
                        vocab_size=256,
                        hidden_size=16,
                        intermediate_size=32,
                        num_attention_heads=2,
                        num_hidden_layers=1,
                        bos_token_id=1,
                        eos_token_id=2,
                        pad_token_id=0,
                        max_position_embeddings=24,
                    )
                )
                .eval()
                .to(device)
            )
        else:
            name = "openai/clip-vit-base-patch32"
            self.tokenizer = CLIPTokenizer.from_pretrained(name)
            self.encoder = CLIPTextModel.from_pretrained(name).eval().to(device)

    def text_embed_cache_key(self):
        return TextEmbedCacheKey.CAPTION

    def uses_text_embeddings_cache(self):
        return True

    def requires_text_embed_image_context(self):
        return False

    @torch.no_grad()
    def encode_text_batch(self, prompts, **kwargs):
        self.calls += 1
        self.encoded_prompts += len(prompts)
        if self.tokenizer is None:
            rows = [[1, *prompt.encode()[:22], 2] for prompt in prompts]
            input_ids = torch.tensor([row + [0] * (24 - len(row)) for row in rows], device=self.device)
            tokens = {"input_ids": input_ids, "attention_mask": input_ids.ne(0)}
        else:
            tokens = self.tokenizer(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
            tokens = tokens.to(self.device)
        return {"prompt_embeds": self.encoder(**tokens).last_hidden_state}

    def pack_text_embeddings_for_cache(self, embeddings):
        return {key: value.detach().cpu() for key, value in embeddings.items()}

    def unpack_text_embeddings_from_cache(self, embeddings):
        return embeddings

    def collate_prompt_embeds(self, samples):
        return {"prompt_embeds": torch.cat([sample["prompt_embeds"] for sample in samples])}


def open_cache(stack, args, accelerator, model, directory, *, writes, local_s3=None, encode_only=False):
    spec = {"id": directory.name, "type": "local", "cache_dir": str(directory), "dataset_type": "text_embeds"}
    if local_s3 is not None:
        spec.update(
            type="aws",
            aws_bucket_name="embeddings",
            aws_region_name="us-east-1",
            aws_endpoint_url=f"http://127.0.0.1:{local_s3[0]}",
            aws_access_key_id="qualification",
            aws_secret_access_key=local_s3[1],
            aws_data_prefix=directory.name,
        )
    store, prefix = build_text_embed_store(spec, args, accelerator)
    stack.enter_context(patch.object(store, "list_files", side_effect=AssertionError("Artifact store inventory")))
    if encode_only:
        for method in ("create_directory", "iter_file_keys", "bulk_exists", "read_cached_tensors", "torch_save"):
            stack.enter_context(
                patch.object(store, method, side_effect=AssertionError(f"Encode-only store access: {method}"))
            )
    cache = TextEmbeddingCache(
        store.id,
        store,
        [model],
        [],
        accelerator,
        model=model,
        cache_dir=prefix,
        text_encoder_batch_size=4,
        text_cache_disable=not writes,
        text_cache_encode_only=encode_only,
    )
    return stack.enter_context(closing(cache))


def measure_s3(client):
    from threading import Lock

    counts = Counter()
    lock = Lock()
    start = time.monotonic()

    def sent(request, event_name, **kwargs):
        with lock:
            counts[event_name.split(".")[-1]] += 1
            if request.method == "PUT":
                counts["upload_bytes"] += int(request.headers.get("Content-Length", 0))

    def received(http_response, parsed, **kwargs):
        with lock:
            counts["retries"] += parsed.get("ResponseMetadata", {}).get("RetryAttempts", 0)
            if http_response.status_code == 200 and "Body" in parsed:
                counts["download_bytes"] += int(http_response.headers.get("Content-Length", 0))

    client.meta.events.register("before-send.s3.*", sent)
    client.meta.events.register("after-call.s3.*", received)

    def snapshot():
        nonlocal start
        with lock:
            result = {**counts, "seconds": round(time.monotonic() - start, 4)}
            counts.clear()
            start = time.monotonic()
        return result

    return snapshot


def compare_sampling(rank, source, params, config, root, options, device):
    # Bound the experiment to the same rows and shards for both policies.
    config = {**config, "webshart": {"sampling_policy": "shard_local"}}
    StateTracker.set_data_backend_config(source.id, config)
    metadata = WebshartMetadataBackend(
        **{
            **params,
            "cache_file": str(root / "policy-buckets"),
            "metadata_file": str(root / "policy-geometry"),
            "max_num_samples": None,
        },
        preparation_only=True,
    )
    chosen = [i * (source.num_shards() - 1) // (options.sampling_shards - 1) for i in range(options.sampling_shards)]

    characteristics = {}

    def entries(shard, **options):
        info = source.dataset.get_shard_info(shard)
        group_ends = list(accumulate(source.dataset._file(shard).info()["row_groups"]))
        rows = source.dataset.read_metadata_rows(
            shard,
            range(min(128, info["num_rows"])),
            fields=["url_status", "stored_bytes", "stored_format", "width", "height", "url"],
        )
        accepted = []
        for row in rows:
            if (
                row.get("url_status", "ok") != "ok"
                or not row.get("stored_bytes", 1)
                or not row.get("width")
                or not row.get("height")
            ):
                continue
            ext = "jxl" if row.get("stored_format") == "jxl" else "jpg"
            index = row["source_row"]
            characteristics[shard, index] = (bisect_right(group_ends, index), urlsplit(row.get("url") or "").hostname)
            accepted.append({**row, "filename": f"{row['source_path']}/{index:012d}.{ext}", "sample_idx": index})
            if len(accepted) == 12:
                break
        return accepted

    if rank == 0:
        with (
            patch.object(metadata, "_all_shard_indices", return_value=chosen),
            patch.object(source.loader, "iter_samples_in_shard", side_effect=entries),
        ):
            metadata.compute_aspect_ratio_bucket_indices()
    dist.barrier()
    shared = [characteristics]
    dist.broadcast_object_list(shared, src=0)
    characteristics = shared[0]
    metadata.reload_cache()
    assert not metadata.image_metadata and not metadata.aspect_ratio_bucket_indices
    runtime = DatasetRuntime(source.id, source, metadata, config)
    results = {}
    unique = {}
    policies = ("shard_local", "global") if options.shard_local_first else ("global", "shard_local")
    for policy in policies:
        if options.sampling_cache_gb:
            source.dataset.enable_shard_cache(str(root / f"payload-{policy}"), options.sampling_cache_gb, 2)
        spec = EpochSpec(source.id, policy, 19, 1, 2, 2, rank, 1, 0, True, (2, 1))
        counters = source.dataset.io_stats()
        peak_disk_bytes = 0
        with closing(runtime.open_epoch(spec)) as cursor:
            local = []
            shard_counts = []
            groups, domains, buckets = [], [], []
            missing_domain = 0
            lru = []
            misses = 0
            timings = []
            for _ in range(cursor.batches):
                start = time.monotonic()
                bucket, refs = cursor.next_batch()
                buckets.append(bucket)
                images = runtime.read_samples(refs, fields={"image"})
                values = [item["image"].convert("RGB").resize((1, 1)).getpixel((0, 0)) for item in images]
                checksum = torch.tensor(values, device=device, dtype=torch.float32).sum()
                assert torch.isfinite(checksum).item()
                addresses = [source.parse_sample_id(ref) for ref in refs]
                shards = [address.shard_idx for address in addresses]
                metadata_rows = [characteristics[address.shard_idx, address.sample_idx] for address in addresses]
                groups.append(len({(shard, group) for shard, (group, _) in zip(shards, metadata_rows)}))
                domains.append(len({domain for _, domain in metadata_rows if domain}))
                missing_domain += sum(domain is None for _, domain in metadata_rows)
                shard_counts.append(len(set(shards)))
                for shard in shards:
                    if shard in lru:
                        lru.remove(shard)
                    else:
                        misses += 1
                    lru.append(shard)
                    lru = lru[-2:]
                if options.sampling_cache_gb:
                    sizes = []
                    for path in (root / f"payload-{policy}").rglob("*.parquet"):
                        try:
                            sizes.append(path.stat().st_size)
                        except FileNotFoundError:
                            pass  # A peer can evict an idle shard between directory traversal and stat.
                    peak_disk_bytes = max(peak_disk_bytes, sum(sizes))
                local.extend(refs)
                timings.append(time.monotonic() - start)
        gathered = [None, None]
        io = {key: value - counters[key] for key, value in source.dataset.io_stats().items()}
        dist.all_gather_object(
            gathered,
            dict(
                refs=local,
                shards=shard_counts,
                misses=misses,
                seconds=timings,
                io=io,
                disk=peak_disk_bytes,
                groups=groups,
                domains=domains,
                buckets=buckets,
                missing_domain=missing_domain,
            ),
        )
        unique[policy] = set(ref for item in gathered for ref in item["refs"])
        latencies = sorted(t for item in gathered for t in item["seconds"])
        results[policy] = dict(
            transport={key: sum(item["io"][key] for item in gathered) for key in io},
            peak_observed_cache_bytes=max(item["disk"] for item in gathered),
            accepted=len(unique[policy]),
            batches_per_rank=len(local) // 2,
            range_read_mode=not bool(options.sampling_cache_gb),
            configured_disk_cache_gb=options.sampling_cache_gb,
            shard_fanout=dict(Counter(x for item in gathered for x in item["shards"])),
            row_group_fanout=dict(Counter(n for item in gathered for n in item["groups"])),
            source_domain_fanout=dict(Counter(n for item in gathered for n in item["domains"])),
            occurrences_without_domain=sum(item["missing_domain"] for item in gathered),
            prepared_buckets=dict(Counter(b for item in gathered for b in item["buckets"])),
            adjacent_bucket_changes=sum(
                sum(a != b for a, b in zip(item["buckets"], item["buckets"][1:])) for item in gathered
            ),
            simulated_two_shard_lru_misses=sum(item["misses"] for item in gathered),
            mean_batch_seconds=sum(latencies) / len(latencies),
            p95_batch_seconds=latencies[int(0.95 * (len(latencies) - 1))],
        )
    assert unique["global"] == unique["shard_local"]
    assert len(unique["global"]) == metadata.describe().accepted_samples
    return results


def run_rank(rank, connection, options, directory, port):
    os.environ.update(LOCAL_RANK=str(rank), RANK=str(rank), WORLD_SIZE="2", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    if not options.cpu_fixture:
        torch.cuda.set_device(rank)
    accelerator = Accelerator(
        cpu=options.cpu_fixture, kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=2))]
    )
    root = Path(directory)
    args = _CacheArgsNamespace(
        dict(
            seed=19,
            output_dir=directory,
            skip_file_discovery="",
            caption_dropout_probability=0,
            allow_dataset_oversubscription=False,
            disable_bucket_pruning=False,
            offload_during_startup=False,
        )
    )
    config = dict(
        id="probe",
        caption_strategy="webshart",
        resolution=64,
        resolution_type="pixel",
        crop=False,
        webshart={"sampling_policy": "shard_local"},
    )
    StateTracker.set_accelerator(accelerator)
    StateTracker.set_args(args)
    StateTracker.set_data_backend_config(config["id"], config)
    model = ClipEncoder(accelerator.device, fixture=options.cpu_fixture)
    source = WebshartDataBackend(
        accelerator,
        config["id"],
        options.source,
        format="parquet",
        manifest=options.manifest,
        cache_dir=str(root / "source"),
        shard_cache_gb=0,
        aws_region_name="auto",
        aws_endpoint_url=connection.get("url"),
        aws_access_key_id=connection.get("access_id"),
        aws_secret_access_key=connection.get("secret_key"),
    )
    params = dict(
        id=source.id,
        instance_data_dir=source.cache_dir,
        cache_file=str(root / "buckets"),
        metadata_file=str(root / "geometry"),
        data_backend=source,
        accelerator=None,
        batch_size=1,
        resolution=64,
        resolution_type="pixel",
        max_num_samples=16,
        dataset_config=config,
        runtime_args=args,
    )
    if rank == 0:
        with closing(WebshartMetadataBackend(**params)) as writer:
            writer.compute_aspect_ratio_bucket_indices()
    dist.barrier()
    reader = WebshartMetadataBackend(**params, preparation_only=True)
    runtime = DatasetRuntime(source.id, source, reader, config)
    results = {
        "device": "cpu" if options.cpu_fixture else torch.cuda.get_device_name(),
        "ranks": 2,
        "source_revision": source.dataset.fingerprint,
        "view_revision": runtime.describe().view_revision,
    }
    try:
        with ExitStack() as stack:
            for method in ("list_files", "iter_file_keys"):
                stack.enter_context(
                    patch.object(source, method, side_effect=AssertionError(f"Dataset inventory via {method}"))
                )
            records = list(runtime.iter_preparation_text_records(model, worker=rank, workers=2))
            assert len(records) == 8, "Select a source with 16 accepted single-caption rows"
            refs = [record["metadata"]["image_path"] for record in records]
            all_refs = [None, None]
            dist.all_gather_object(all_refs, refs)
            assert len(set(all_refs[0] + all_refs[1])) == 16
            offline = open_cache(stack, args, accelerator, model, root / "offline", writes=True, local_s3=options.local_s3)
            runtime.text_embed_cache = offline
            physical_dir = root / "offline" if options.local_s3 is None else root / "objects" / "embeddings" / "offline"
            snapshot = measure_s3(offline.data_backend.client) if options.local_s3 else None
            factory = FactoryRegistry(args, accelerator, None, None, model=model)
            bundle = dict(
                id=source.id,
                config=dict(config),
                data_backend=source,
                metadata_backend=reader,
                dataset_runtime=runtime,
                text_embed_cache=offline,
                instance_data_dir=None,
            )

            def prepare(action):
                offline.preparation_listing = build_text_preparation_listing(
                    {"text_cache_use_listing": options.s3_listing},
                    offline.data_backend,
                    offline.cache_dir,
                    args,
                    accelerator,
                )
                try:
                    result = action()
                    dist.barrier()
                    return result
                finally:
                    offline.finish_preparation()

            prepare(lambda: factory._process_text_embeddings(config, bundle, None))
            assert len(list(physical_dir.glob("*.pt"))) == 16
            results["cold_encoder_calls"] = model.calls
            if snapshot:
                results["cold_s3"] = snapshot()
            first = records[0]
            cached = offline.load_from_cache(offline.hash_prompt_with_path(first))["prompt_embeds"]
            expected = model.encode_text_batch([first["prompt"]])["prompt_embeds"]
            torch.testing.assert_close(cached.cpu(), expected.cpu(), rtol=2e-4, atol=5e-5)
            calls = model.calls
            if snapshot:
                results["readback_s3"] = snapshot()
            warm = CacheJobService._iter_prompt_records(model, runtime)
            assert prepare(lambda: offline.precompute_embeddings(warm, batch_size=8)) == 16
            assert model.calls == calls
            results["warm_calls"] = model.calls - calls
            if snapshot:
                results["warm_s3"] = snapshot()
            dist.barrier()
            for record in records[:2]:
                offline.data_backend.delete(offline.hash_prompt_with_path(record))
            dist.barrier()
            prepare(lambda: factory._process_text_embeddings(config, bundle, None))
            assert model.calls == calls + 1
            assert len(list(physical_dir.glob("*.pt"))) == 16
            results["mixed_missing_per_rank"] = 2
            if snapshot:
                results["mixed_s3"] = snapshot()
                namespace = ("cold_s3", "warm_s3", "mixed_s3")
                observed = [
                    sum(count for key, count in results[name].items() if key.startswith("List")) for name in namespace
                ]
                listing_requests = [None, None]
                dist.all_gather_object(listing_requests, observed)
                expected = [int(options.s3_listing)] * len(namespace)
                assert [sum(counts) for counts in zip(*listing_requests, strict=True)] == expected
                results["store_listing"] = {
                    "enabled": options.s3_listing,
                    "total_requests_per_phase": dict(zip(namespace, expected)),
                }
                if options.s3_listing:
                    assert results["warm_s3"].get("HeadObject", 0) == 0
                    assert not list((root / ".artifact-listings").glob("*.sqlite"))
                assert results["cold_s3"].get("PutObject") == 8
                assert results["warm_s3"].get("PutObject", 0) == 0
                assert results["mixed_s3"].get("PutObject") == 2

            online = open_cache(stack, args, accelerator, model, root / f"online-{rank}", writes=False)
            online.data_backend.torch_save({"prompt_embeds": cached}, online.hash_prompt_with_path(first))
            calls = model.calls
            encoded = model.encoded_prompts
            torch.manual_seed(123)
            head = DistributedDataParallel(
                torch.nn.Linear(model.encoder.config.hidden_size, 1).to(accelerator.device),
                device_ids=None if options.cpu_fixture else [rank],
            )
            optimizer = torch.optim.SGD(head.parameters(), lr=1e-3)
            for batch in (records[:4], records[4:]):
                images = runtime.read_samples([record["metadata"]["image_path"] for record in batch], fields={"image"})
                labels = [item["image"].convert("RGB").resize((1, 1)).getpixel((0, 0))[0] / 255 for item in images]
                embeddings = compute_prompt_embeddings(batch, online, model)["prompt_embeds"]
                loss = (
                    (head(embeddings.mean(dim=1)).flatten() - torch.tensor(labels, device=accelerator.device))
                    .square()
                    .mean()
                )
                assert torch.isfinite(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            assert model.calls == calls + 2
            assert model.encoded_prompts - encoded == len(records) - 1
            assert len(list(Path(online.cache_dir).glob("*.pt"))) == 1
            results["read_existing_cache_hits_per_rank"] = 1

            encode_only = open_cache(
                stack, args, accelerator, model, root / f"encode-only-{rank}", writes=False, encode_only=True
            )
            calls = model.calls
            encoded = model.encoded_prompts
            selected = [records[0], records[1], records[0]]
            on_demand = compute_prompt_embeddings(selected, encode_only, model)["prompt_embeds"]
            known = compute_prompt_embeddings(selected, offline, model)["prompt_embeds"]
            torch.testing.assert_close(on_demand.cpu(), known.cpu(), rtol=2e-4, atol=5e-5)
            assert model.calls == calls + 1
            assert model.encoded_prompts - encoded == 2
            assert not Path(encode_only.cache_dir).exists()
            results["encode_only_calls_per_rank"] = model.calls - calls
            weight = head.module.weight.detach().contiguous()
            weights = [torch.empty_like(weight) for _ in range(2)]
            dist.all_gather(weights, weight)
            torch.testing.assert_close(weights[0], weights[1], rtol=0, atol=0)
            assert not reader.image_metadata and not reader.aspect_ratio_bucket_indices
            results.update(records=16, optimizer_steps=2, online_calls_per_rank=2, weights_equal=True)
            if options.sampling_shards:
                results["sampling"] = compare_sampling(rank, source, params, config, root, options, accelerator.device)
            if rank == 0:
                print(json.dumps(results), flush=True)
    finally:
        runtime.close()
        dist.destroy_process_group()


def create_parquet_fixture(directory):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    root = Path(directory) / "fixtures"
    root.mkdir()
    rows = []
    for index in range(16):
        image = BytesIO()
        Image.new("RGB", (64, 64), (index * 16, 0, 0)).save(image, format="PNG")
        rows.append(
            {
                "image": {"bytes": image.getvalue(), "path": None},
                "captions": f"fixture caption {index}",
                "width": 64,
                "height": 64,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), root / "samples.parquet", row_group_size=4)
    return str(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source")
    parser.add_argument("--manifest")
    parser.add_argument("--cpu-fixture", action="store_true")
    parser.add_argument("--local-s3-binary", type=Path, help="rclone binary for a loopback-only on-disk S3 artifact test")
    parser.add_argument("--s3-listing", action="store_true", help="Use one shared S3 presence listing per preparation phase")
    parser.add_argument("--sampling-shards", type=int, default=0)
    parser.add_argument("--shard-local-first", action="store_true")
    parser.add_argument("--sampling-cache-gb", type=float, default=0)
    options = parser.parse_args()
    if options.s3_listing and options.local_s3_binary is None:
        parser.error("--s3-listing requires --local-s3-binary")
    if options.sampling_shards < 0 or options.sampling_shards == 1:
        parser.error("--sampling-shards must be zero or at least two")
    if options.cpu_fixture:
        if options.source or options.manifest or options.sampling_shards:
            parser.error("--cpu-fixture supplies its own source and does not measure sampling locality")
        connection = {}
    else:
        if not options.source or not options.manifest:
            parser.error("--source and --manifest are required for R2 verification")
        connection = json.load(sys.stdin)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with TemporaryDirectory() as directory, ExitStack() as stack:
        if options.cpu_fixture:
            options.source = create_parquet_fixture(directory)
        options.local_s3 = None
        if options.local_s3_binary:
            objects = Path(directory) / "objects"
            (objects / "embeddings").mkdir(parents=True)
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                addr = listener.getsockname()
            secret = secrets.token_hex(24)
            server = stack.enter_context(
                subprocess.Popen(
                    [
                        str(options.local_s3_binary.resolve()),
                        "serve",
                        "s3",
                        str(objects),
                        "--addr",
                        f"127.0.0.1:{addr[1]}",
                        "--auth-key",
                        f"qualification,{secret}",
                        "--config",
                        os.devnull,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
            stack.callback(server.terminate)
            for attempt in range(50):
                if server.poll() is not None:
                    raise RuntimeError("Pod-local S3 server failed to start")
                try:
                    with socket.create_connection(addr, timeout=1):
                        break
                except ConnectionRefusedError:
                    time.sleep(0.1)
            else:
                raise TimeoutError("Pod-local S3 server did not become ready")
            options.local_s3 = (addr[1], secret)
        spawn(run_rank, args=(connection, options, directory, port), nprocs=2, join=True)
