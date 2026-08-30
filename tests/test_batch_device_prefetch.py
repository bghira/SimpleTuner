import unittest
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.data_backend.runtime.device_prefetch import (
    CudaBatchPrefetcher,
    DevicePrefetchedBatch,
    _map_tensors,
    cpu_tensor_bytes,
)


class BatchDevicePrefetchTests(unittest.TestCase):
    def test_device_prefetch_requires_an_accelerator_device(self):
        with self.assertRaisesRegex(ValueError, "accelerator device"):
            CudaBatchPrefetcher(None, minimum_bytes=1)

    def test_cpu_tensor_bytes_counts_unique_nested_tensors(self):
        tensor = torch.zeros(4, dtype=torch.float32)
        batch = {"first": tensor, "nested": [tensor, torch.zeros(2, dtype=torch.int16)]}

        self.assertEqual(cpu_tensor_bytes(batch), 20)

    def test_map_tensors_preserves_aliases(self):
        tensor = torch.ones(2)
        mapped = _map_tensors({"first": tensor, "second": [tensor]}, lambda value: value + 1)

        self.assertIs(mapped["first"], mapped["second"][0])
        torch.testing.assert_close(mapped["first"], torch.full((2,), 2.0))

    @patch("simpletuner.helpers.data_backend.runtime.device_prefetch.torch.cuda.Stream")
    @patch("simpletuner.helpers.data_backend.runtime.device_prefetch.torch.cuda.device")
    def test_payload_below_threshold_stays_on_cpu(self, cuda_device, stream_factory):
        cuda_device.return_value = MagicMock()
        prefetcher = CudaBatchPrefetcher(torch.device("cuda:0"), minimum_bytes=1024)
        batch = {"tensor": torch.zeros(4)}

        self.assertIs(prefetcher.prefetch(batch), batch)
        stream_factory.assert_called_once_with(device=torch.device("cuda:0"))

    @patch("simpletuner.helpers.data_backend.runtime.device_prefetch._record_device_stream")
    @patch("simpletuner.helpers.data_backend.runtime.device_prefetch._move_cpu_tensors")
    @patch("simpletuner.helpers.data_backend.runtime.device_prefetch._pin_cpu_tensors")
    @patch("simpletuner.helpers.data_backend.runtime.device_prefetch.torch.cuda.Event")
    @patch("simpletuner.helpers.data_backend.runtime.device_prefetch.torch.cuda.Stream")
    @patch("simpletuner.helpers.data_backend.runtime.device_prefetch.torch.cuda.stream")
    @patch("simpletuner.helpers.data_backend.runtime.device_prefetch.torch.cuda.device")
    @patch("simpletuner.helpers.data_backend.runtime.device_prefetch.torch.cuda.current_stream")
    def test_large_batch_transfers_and_waits_on_consumer_stream(
        self,
        current_stream,
        cuda_device,
        cuda_stream,
        stream_factory,
        event_factory,
        pin_batch,
        move_batch,
        record_stream,
    ):
        cuda_device.return_value = MagicMock()
        cuda_stream.return_value = MagicMock()
        transfer_stream = stream_factory.return_value
        ready_event = event_factory.return_value
        ready_event.query.return_value = False
        source_batch = {"tensor": "pinned"}
        device_batch = {"tensor": "cuda"}
        consumed_batch = {"tensor": "recorded"}
        pin_batch.return_value = source_batch
        move_batch.return_value = device_batch
        record_stream.return_value = consumed_batch

        prefetcher = CudaBatchPrefetcher(torch.device("cuda:0"), minimum_bytes=1)
        queued = prefetcher.prefetch({"tensor": torch.zeros(2)})

        self.assertIsInstance(queued, DevicePrefetchedBatch)
        self.assertIs(queued.source_batch, source_batch)
        move_batch.assert_called_once_with(source_batch, torch.device("cuda:0"))
        ready_event.record.assert_called_once_with(transfer_stream)

        result = prefetcher.consume(queued)

        self.assertIs(result, consumed_batch)
        current_stream.return_value.wait_event.assert_called_once_with(ready_event)
        record_stream.assert_called_once_with(device_batch, current_stream.return_value)
        self.assertEqual(prefetcher._pending_sources, [(ready_event, source_batch)])

    def test_prefetch_threshold_field_is_registered(self):
        from simpletuner.simpletuner_sdk.server.services.field_registry.registry import FieldRegistry

        field = FieldRegistry().get_field("dataloader_prefetch_device_threshold_mb")

        self.assertEqual(field.arg_name, "--dataloader_prefetch_device_threshold_mb")
        self.assertEqual(field.default_value, 0)


if __name__ == "__main__":
    unittest.main()
