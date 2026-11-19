# Data Backend Factory Architecture

## Dataset construction

The factory validates identifiers, expands generated datasets, and orders metadata sources before their dependents. Builders open the source and metadata providers. `DatasetRuntime` binds the providers, resolved configuration, related datasets, and artifact caches. `StateTracker` registers the handles and still mirrors entries for legacy callers.

```text
FactoryRegistry → builders → DatasetRuntime
                              ├─ BaseDataBackend
                              ├─ MetadataBackend
                              ├─ TextEmbeddingCache / VAECache
                              └─ training dataloader
```

## Provider contracts

The source provider reads files or objects. The metadata provider publishes accepted membership and crop geometry. `describe()` reports counts and revisions when available, `read_page()` returns a bounded page, and `read_samples()` reads the requested fields in reference order with duplicates preserved. Keep sample references inside their provider's identity scheme.

Use `iter_preparation_batches()` and `iter_preparation_text_records()` for preprocessing. Providers divide accepted samples into worker partitions independently of training repeats and padding. Artifact bindings can point at a different store from the source. Online text encoding uses the current batch.

Local discovery, AWS, and CSV use shared legacy metadata and scheduling code. Hugging Face can partition prepared transforms by Arrow shard. Native Webshart Parquet uses an indexed membership view, per-shard transforms for supported image crops, and a reconstructible epoch cursor.

## Training and resume

The trainer asks the runtime for scheduled batches. The dataset mixer selects among them and the batch fetcher can run ahead. After consuming a batch, the trainer acknowledges its continuation. Checkpoints save that consumed position and the mixer state. Native resume requires the same accepted view, sampling options, and distributed layout.

## Verification

For a new provider, test reference stability, ordered projected reads, preparation coverage, and abort or reopen behavior through the runtime. Add a cursor replay test when the provider implements native scheduling. Test a replacement publication while another consumer holds an older view. Verify the same cache keys through both preparation and training when adding a new artifact capability.

```bash
.venv/bin/python -m unittest -v -f tests.test_dataset_runtime tests.test_webshart_parquet_backend tests.test_vae_cache_traversal
```
