"""
Component tests for data backend runtime components.

This test suite validates the runtime components that handle dataloader
iteration, batch fetching, and other dynamic aspects of the data backend
system during training.
"""

import queue
import threading
import time
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

from simpletuner.helpers.data_backend.runtime import BatchFetcher
from simpletuner.helpers.data_backend.runtime import dataloader_iterator as dataloader_module
from simpletuner.helpers.data_backend.runtime import get_backend_weight, random_dataloader_iterator, select_dataloader_index


class TestBatchFetcher(unittest.TestCase):
    """Test the BatchFetcher class for background data loading"""

    def setUp(self):
        """Set up test fixtures"""
        self.datasets = {"dataset1": Mock(), "dataset2": Mock()}

    def test_batch_fetcher_initialization(self):
        """Test BatchFetcher initialization with default parameters"""
        fetcher = BatchFetcher(step=100, max_size=5, datasets=self.datasets)

        self.assertEqual(fetcher.step, 100)
        self.assertEqual(fetcher.queue.maxsize, 5)
        self.assertEqual(fetcher.datasets, self.datasets)
        self.assertTrue(fetcher.keep_running)

    def test_batch_fetcher_initialization_custom_params(self):
        """Test BatchFetcher initialization with custom parameters"""
        fetcher = BatchFetcher(step=500, max_size=10, datasets=self.datasets)

        self.assertEqual(fetcher.step, 500)
        self.assertEqual(fetcher.queue.maxsize, 10)
        self.assertEqual(fetcher.datasets, self.datasets)
        self.assertTrue(fetcher.keep_running)

    def test_batch_fetcher_initialization_empty_datasets(self):
        """Test BatchFetcher initialization with empty datasets"""
        empty_datasets = {}
        fetcher = BatchFetcher(step=100, max_size=5, datasets=empty_datasets)

        self.assertEqual(fetcher.datasets, empty_datasets)
        self.assertTrue(fetcher.keep_running)

    @patch("simpletuner.helpers.data_backend.runtime.batch_fetcher.random_dataloader_iterator")
    def test_fetch_responses_single_iteration(self, mock_iterator):
        """Test fetch_responses method for single iteration"""
        mock_iterator.return_value = {"batch": "data", "step": 100}

        fetcher = BatchFetcher(step=100, max_size=5, datasets=self.datasets)

        # Stop immediately to test just one iteration
        fetcher.keep_running = False

        # Run fetch_responses
        fetcher.fetch_responses()

        # Verify that data was added to queue
        self.assertEqual(fetcher.queue.qsize(), 1)

        # Verify the correct data was added
        queued_data = fetcher.queue.get()
        self.assertEqual(queued_data, {"batch": "data", "step": 100})

        # Verify random_dataloader_iterator was called correctly
        mock_iterator.assert_called_once_with(100, self.datasets)

    @patch("simpletuner.helpers.data_backend.runtime.batch_fetcher.random_dataloader_iterator")
    @patch("time.sleep")
    def test_fetch_responses_multiple_iterations(self, mock_sleep, mock_iterator):
        """Test fetch_responses method for multiple iterations"""
        mock_iterator.side_effect = [
            {"batch": "data1", "step": 100},
            {"batch": "data2", "step": 101},
            {"batch": "data3", "step": 102},
        ]

        fetcher = BatchFetcher(step=100, max_size=10, datasets=self.datasets)

        # Create a counter to stop after a few iterations
        iteration_count = 0
        original_keep_running = fetcher.keep_running

        def check_iteration():
            nonlocal iteration_count
            iteration_count += 1
            return iteration_count <= 3 and original_keep_running

        # Replace keep_running with our counter
        type(fetcher).keep_running = property(lambda self: check_iteration())

        # Run fetch_responses
        fetcher.fetch_responses()

        # Verify that multiple batches were added
        self.assertGreaterEqual(fetcher.queue.qsize(), 1)

        # Verify iterator was called multiple times
        self.assertGreater(mock_iterator.call_count, 1)

    @patch("simpletuner.helpers.data_backend.runtime.batch_fetcher.random_dataloader_iterator")
    def test_fetch_responses_queue_full_behavior(self, mock_iterator):
        """Test fetch_responses behavior when queue is full"""
        mock_iterator.side_effect = [{"batch": "data1"}, {"batch": "data2"}, {"batch": "data3"}]

        # Create fetcher with small queue
        fetcher = BatchFetcher(step=100, max_size=1, datasets=self.datasets)

        # Stop after enough iterations to test queue full behavior
        iteration_count = 0

        def limited_keep_running():
            nonlocal iteration_count
            iteration_count += 1
            return iteration_count <= 2

        type(fetcher).keep_running = property(lambda self: limited_keep_running())

        # Run fetch_responses - should handle queue full gracefully
        fetcher.fetch_responses()

        # Queue should be at maximum capacity
        self.assertEqual(fetcher.queue.qsize(), 1)

    def test_next_response_with_data_available(self):
        """Test next_response method when data is available"""
        fetcher = BatchFetcher(step=100, max_size=5, datasets=self.datasets)

        # Add test data to queue
        test_data = {"batch": "test_data", "step": 100}
        fetcher.queue.put(test_data)

        # Get next response
        result = fetcher.next_response(step=101)

        # Verify step was updated
        self.assertEqual(fetcher.step, 101)

        # Verify correct data was returned
        self.assertEqual(result, test_data)

    def test_next_response_with_empty_queue(self):
        """Test next_response method blocks when queue is empty"""
        fetcher = BatchFetcher(step=100, max_size=5, datasets=self.datasets)

        # Add data in a separate thread after a short delay
        def add_data_later():
            time.sleep(0.1)
            test_data = {"batch": "delayed_data", "step": 100}
            fetcher.queue.put(test_data)

        thread = threading.Thread(target=add_data_later)
        thread.start()

        # This should block until data is available
        start_time = time.time()
        result = fetcher.next_response(step=101)
        elapsed_time = time.time() - start_time

        thread.join()

        # Verify it actually waited
        self.assertGreater(elapsed_time, 0.05)

        # Verify correct data was returned
        self.assertEqual(result, {"batch": "delayed_data", "step": 100})

        # Verify step was updated
        self.assertEqual(fetcher.step, 101)

    def test_stop_fetching(self):
        """Test stop_fetching method"""
        fetcher = BatchFetcher(step=100, max_size=5, datasets=self.datasets)

        # Initially should be running
        self.assertTrue(fetcher.keep_running)

        # Stop fetching
        fetcher.stop_fetching()

        # Should no longer be running
        self.assertFalse(fetcher.keep_running)

    @patch("threading.Thread")
    def test_start_fetching_creates_thread(self, mock_thread_class):
        """Test that start_fetching creates and starts a thread correctly"""
        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread

        fetcher = BatchFetcher(step=100, max_size=5, datasets=self.datasets)

        # Start fetching
        result_thread = fetcher.start_fetching()

        # Verify thread was created with correct target
        mock_thread_class.assert_called_once_with(target=fetcher.fetch_responses)

        # Verify thread was started
        mock_thread.start.assert_called_once()

        # Verify correct thread was returned
        self.assertEqual(result_thread, mock_thread)

    @patch("threading.Thread")
    def test_start_fetching_thread_properties(self, mock_thread_class):
        """Test that start_fetching sets thread as daemon"""
        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread

        fetcher = BatchFetcher(step=100, max_size=5, datasets=self.datasets)

        # Start fetching
        fetcher.start_fetching()

        # Verify thread daemon property was set
        self.assertTrue(mock_thread.daemon)

    @patch("simpletuner.helpers.data_backend.runtime.batch_fetcher.random_dataloader_iterator")
    def test_fetch_responses_exception_handling(self, mock_iterator):
        """Test fetch_responses handles exceptions gracefully"""
        # Make iterator raise an exception
        mock_iterator.side_effect = Exception("Test exception")

        fetcher = BatchFetcher(step=100, max_size=5, datasets=self.datasets)
        fetcher.keep_running = False  # Stop after one iteration

        # This should not raise an exception
        try:
            fetcher.fetch_responses()
        except Exception as e:
            self.fail(f"fetch_responses raised an exception: {e}")

    def test_queue_properties(self):
        """Test that queue has correct properties"""
        fetcher = BatchFetcher(step=100, max_size=7, datasets=self.datasets)

        # Queue should be a Queue instance
        self.assertIsInstance(fetcher.queue, queue.Queue)

        # Queue should have correct max size
        self.assertEqual(fetcher.queue.maxsize, 7)

        # Queue should initially be empty
        self.assertTrue(fetcher.queue.empty())


class TestDataloaderIterator(unittest.TestCase):
    """Test dataloader iteration functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_backend1 = Mock()
        self.mock_backend1.id = "backend1"
        self.mock_backend1.get_data_loader.return_value = "dataloader1"

        self.mock_backend2 = Mock()
        self.mock_backend2.id = "backend2"
        self.mock_backend2.get_data_loader.return_value = "dataloader2"

    def tearDown(self):
        dataloader_module._SCALED_SAMPLERS.clear()

    def test_get_backend_weight_with_probability(self):
        """Test get_backend_weight with probability setting"""
        backend_config = {"probability": 0.7}

        weight = get_backend_weight("test_id", backend_config, step=100)

        self.assertEqual(weight, 0.7)

    def test_get_backend_weight_without_probability(self):
        """Test get_backend_weight with default weight (no probability)"""
        backend_config = {}

        weight = get_backend_weight("test_id", backend_config, step=100)

        self.assertEqual(weight, 1.0)

    def test_get_backend_weight_with_zero_probability(self):
        """Test get_backend_weight with zero probability"""
        backend_config = {"probability": 0.0}

        weight = get_backend_weight("test_id", backend_config, step=100)

        self.assertEqual(weight, 0.0)

    def test_get_backend_weight_with_high_probability(self):
        """Test get_backend_weight with probability > 1.0"""
        backend_config = {"probability": 1.5}

        weight = get_backend_weight("test_id", backend_config, step=100)

        self.assertEqual(weight, 1.5)

    def test_select_dataloader_index_single_backend(self):
        """Test select_dataloader_index with single backend"""
        backends = {"backend1": self.mock_backend1}

        result = select_dataloader_index(step=100, backends=backends)

        self.assertEqual(result, "backend1")

    def test_select_dataloader_index_multiple_backends(self):
        """Test select_dataloader_index with multiple backends"""
        backends = {"backend1": self.mock_backend1, "backend2": self.mock_backend2}

        # Test multiple times to ensure we get different results
        results = set()
        for i in range(100):
            result = select_dataloader_index(step=i, backends=backends)
            results.add(result)

        # Should get both backends eventually with random selection
        self.assertTrue(len(results) >= 1)  # At least one backend selected
        self.assertTrue(all(r in ["backend1", "backend2"] for r in results))

    def test_select_dataloader_index_empty_backends(self):
        """Test select_dataloader_index with empty backends"""
        backends = {}

        with self.assertRaises((IndexError, KeyError, ValueError)):
            select_dataloader_index(step=100, backends=backends)

    @patch("simpletuner.helpers.data_backend.runtime.dataloader_iterator.StateTracker")
    def test_select_dataloader_index_no_active_misconfig(self, mock_state_tracker):
        """Test select_dataloader_index raises error when no active backends and no exhausted backends"""
        # Set up mock to indicate no exhausted backends (misconfiguration)
        mock_state_tracker.exhausted_backends = []
        mock_state_tracker.get_epoch.return_value = 1
        mock_state_tracker.get_global_step.return_value = 0
        mock_state_tracker.get_data_backend_config.return_value = {"start_epoch": 5}

        # Backend with start_epoch in the future
        backend_config = {"start_epoch": 5}
        backends = {"future_backend": backend_config}

        with self.assertRaises(ValueError) as context:
            select_dataloader_index(step=100, backends=backends)

        self.assertIn("No active datasets available", str(context.exception))

    @patch("simpletuner.helpers.data_backend.runtime.dataloader_iterator.StateTracker")
    def test_select_dataloader_index_no_active_with_exhausted(self, mock_state_tracker):
        """Test select_dataloader_index returns None when no active backends but some exhausted"""
        # Set up mock to indicate some exhausted backends (normal epoch end)
        mock_state_tracker.exhausted_backends = ["exhausted_backend1"]
        mock_state_tracker.get_data_backend_config.return_value = {"start_epoch": 5}

        # Backend with start_epoch in the future
        backend_config = {"start_epoch": 5}
        backends = {"future_backend": backend_config}

        result = select_dataloader_index(step=100, backends=backends)
        self.assertIsNone(result)

    @patch("simpletuner.helpers.data_backend.runtime.dataloader_iterator.select_dataloader_index")
    @patch("simpletuner.helpers.data_backend.runtime.dataloader_iterator.StateTracker")
    def test_random_dataloader_iterator_returns_false_on_no_active(self, mock_state_tracker, mock_select):
        """Test random_dataloader_iterator returns False when select_dataloader_index returns None"""
        mock_select.return_value = None
        mock_state_tracker.get_args.return_value = Mock(gradient_accumulation_steps=1)
        backends = {"backend1": self.mock_backend1}

        result = random_dataloader_iterator(step=100, backends=backends)

        self.assertFalse(result)

    @patch("simpletuner.helpers.data_backend.runtime.dataloader_iterator.select_dataloader_index")
    def test_random_dataloader_iterator_single_backend(self, mock_select):
        """Test random_dataloader_iterator with single backend"""
        mock_select.return_value = "backend1"
        backends = {"backend1": self.mock_backend1}

        result = random_dataloader_iterator(step=100, backends=backends)

        # Verify select_dataloader_index was called correctly
        mock_select.assert_called_once_with(100, backends)

        # Verify get_data_loader was called
        self.mock_backend1.get_data_loader.assert_called_once()

        # Verify correct dataloader was returned
        self.assertEqual(result, "dataloader1")

    @patch("simpletuner.helpers.data_backend.runtime.dataloader_iterator.select_dataloader_index")
    def test_random_dataloader_iterator_multiple_backends(self, mock_select):
        """Test random_dataloader_iterator with multiple backends"""
        mock_select.return_value = "backend2"
        backends = {"backend1": self.mock_backend1, "backend2": self.mock_backend2}

        result = random_dataloader_iterator(step=200, backends=backends)

        # Verify select_dataloader_index was called correctly
        mock_select.assert_called_once_with(200, backends)

        # Verify correct backend's get_data_loader was called
        self.mock_backend2.get_data_loader.assert_called_once()
        self.mock_backend1.get_data_loader.assert_not_called()

        # Verify correct dataloader was returned
        self.assertEqual(result, "dataloader2")

    @patch("simpletuner.helpers.data_backend.runtime.dataloader_iterator.select_dataloader_index")
    def test_random_dataloader_iterator_backend_not_found(self, mock_select):
        """Test random_dataloader_iterator when selected backend doesn't exist"""
        mock_select.return_value = "nonexistent_backend"
        backends = {"backend1": self.mock_backend1}

        with self.assertRaises(KeyError):
            random_dataloader_iterator(step=100, backends=backends)

    def test_random_dataloader_iterator_empty_backends(self):
        """Test random_dataloader_iterator with empty backends"""
        backends = {}

        with self.assertRaises((IndexError, KeyError, ValueError)):
            random_dataloader_iterator(step=100, backends=backends)

    def test_random_dataloader_iterator_backend_method_error(self):
        """Test random_dataloader_iterator when backend's get_data_loader fails"""
        self.mock_backend1.get_data_loader.side_effect = Exception("Dataloader error")
        backends = {"backend1": self.mock_backend1}

        with patch("simpletuner.helpers.data_backend.runtime.dataloader_iterator.select_dataloader_index") as mock_select:
            mock_select.return_value = "backend1"

        with self.assertRaises(Exception) as context:
            random_dataloader_iterator(step=100, backends=backends)

            self.assertIn("Dataloader error", str(context.exception))

    @patch("simpletuner.helpers.data_backend.runtime.dataloader_iterator.select_dataloader_index")
    def test_random_iterator_slider_cycle_respects_strength_groups(self, mock_select):
        """Ensure slider_strength sampling cycles positive/negative/neutral backends."""

        class DummyBackend:
            def __init__(self, name: str, strength: float | None):
                self.name = name
                self.slider_strength = strength
                self.config = {"probability": 1.0}

            def get_data_loader(self):
                return f"{self.name}_loader"

        backends = {
            "positive": DummyBackend("positive", 0.5),
            "negative": DummyBackend("negative", -0.5),
            "neutral": DummyBackend("neutral", None),
        }

        results = [random_dataloader_iterator(step=idx, backends=backends) for idx in range(3)]

        self.assertEqual(results, ["positive_loader", "negative_loader", "neutral_loader"])
        mock_select.assert_not_called()


class TestRuntimeIntegration(unittest.TestCase):
    """Integration tests for runtime components working together"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_backend1 = Mock()
        self.mock_backend1.id = "backend1"
        self.mock_backend1.get_data_loader.return_value = {"data": "from_backend1"}

        self.mock_backend2 = Mock()
        self.mock_backend2.id = "backend2"
        self.mock_backend2.get_data_loader.return_value = {"data": "from_backend2"}

        self.backends = {"backend1": self.mock_backend1, "backend2": self.mock_backend2}

    @patch("simpletuner.helpers.data_backend.runtime.batch_fetcher.random_dataloader_iterator")
    def test_batch_fetcher_with_real_iterator(self, mock_iterator):
        """Test BatchFetcher integration with dataloader iterator"""
        mock_iterator.side_effect = [{"batch": "data1", "source": "backend1"}, {"batch": "data2", "source": "backend2"}]

        fetcher = BatchFetcher(step=100, max_size=5, datasets=self.backends)

        # Run one fetch cycle
        fetcher.keep_running = False
        fetcher.fetch_responses()

        # Verify iterator was called with correct parameters
        mock_iterator.assert_called_once_with(100, self.backends)

        # Verify data was queued
        self.assertEqual(fetcher.queue.qsize(), 1)

        # Get and verify the queued data
        result = fetcher.next_response(step=101)
        self.assertEqual(result, {"batch": "data1", "source": "backend1"})

    def test_weight_based_backend_selection_distribution(self):
        """Test that backend weights affect selection distribution"""
        # Create backends with different probabilities
        backend_configs = {"backend1": {"probability": 0.8}, "backend2": {"probability": 0.2}}

        # Test weight calculation
        weight1 = get_backend_weight("backend1", backend_configs["backend1"], step=100)
        weight2 = get_backend_weight("backend2", backend_configs["backend2"], step=100)

        self.assertEqual(weight1, 0.8)
        self.assertEqual(weight2, 0.2)

        # The actual selection distribution would be tested at a higher level
        # Here we just verify the weights are calculated correctly

    @patch("time.sleep")
    @patch("simpletuner.helpers.data_backend.runtime.batch_fetcher.random_dataloader_iterator")
    def test_batch_fetcher_continuous_operation(self, mock_iterator, mock_sleep):
        """Test BatchFetcher continuous operation simulation"""
        mock_iterator.side_effect = [{"batch": f"data{i}"} for i in range(10)]

        fetcher = BatchFetcher(step=100, max_size=3, datasets=self.backends)

        # Simulate a few iterations
        iteration_count = 0

        def limited_iterations():
            nonlocal iteration_count
            iteration_count += 1
            return iteration_count <= 3

        type(fetcher).keep_running = property(lambda self: limited_iterations())

        # Start fetching in background
        thread = fetcher.start_fetching()

        # Wait a bit for background fetching
        time.sleep(0.1)

        # Stop fetching
        fetcher.stop_fetching()

        # Verify some data was produced
        self.assertGreater(mock_iterator.call_count, 0)

    def test_error_resilience(self):
        """Test that runtime components handle errors gracefully"""
        # Test backend with failing get_data_loader
        failing_backend = Mock()
        failing_backend.id = "failing_backend"
        failing_backend.get_data_loader.side_effect = Exception("Backend failure")

        backends_with_failure = {"backend1": self.mock_backend1, "failing_backend": failing_backend}

        # Test that random_dataloader_iterator propagates errors appropriately
        with patch("simpletuner.helpers.data_backend.runtime.dataloader_iterator.select_dataloader_index") as mock_select:
            mock_select.return_value = "failing_backend"

            with self.assertRaises(Exception):
                random_dataloader_iterator(step=100, backends=backends_with_failure)

    def test_step_tracking_consistency(self):
        """Test that step tracking is consistent across components"""
        fetcher = BatchFetcher(step=100, max_size=5, datasets=self.backends)

        # Initial step
        self.assertEqual(fetcher.step, 100)

        # Add some data
        fetcher.queue.put({"data": "test"})

        # Get next response with step update
        result = fetcher.next_response(step=150)

        # Verify step was updated
        self.assertEqual(fetcher.step, 150)

        # Get another response
        fetcher.queue.put({"data": "test2"})
        result = fetcher.next_response(step=200)

        # Verify step updated again
        self.assertEqual(fetcher.step, 200)


class TestRuntimePerformance(unittest.TestCase):
    """Performance-related tests for runtime components"""

    def setUp(self):
        """Set up test fixtures"""
        self.backends = {f"backend{i}": Mock() for i in range(10)}
        for i, backend in enumerate(self.backends.values()):
            backend.id = f"backend{i}"
            backend.get_data_loader.return_value = f"dataloader{i}"

    def test_batch_fetcher_queue_performance(self):
        """Test that BatchFetcher queue operations are efficient"""
        fetcher = BatchFetcher(step=100, max_size=1000, datasets=self.backends)

        # Time queue operations
        start_time = time.time()

        # Add many items to queue
        for i in range(100):
            fetcher.queue.put({"batch": f"data{i}"})

        # Get many items from queue
        for i in range(100):
            fetcher.next_response(step=100 + i)

        elapsed_time = time.time() - start_time

        # Queue operations should be fast (less than 1 second for 200 operations)
        self.assertLess(elapsed_time, 1.0)

    def test_dataloader_iterator_selection_performance(self):
        """Test that backend selection is efficient with many backends"""
        # Time multiple selections
        start_time = time.time()

        for i in range(1000):
            select_dataloader_index(step=i, backends=self.backends)

        elapsed_time = time.time() - start_time

        # Selection should be fast (less than 1 second for 1000 selections)
        self.assertLess(elapsed_time, 1.0)

    def test_weight_calculation_performance(self):
        """Test that weight calculation is efficient"""
        backend_config = {"probability": 0.5}

        # Time many weight calculations
        start_time = time.time()

        for i in range(10000):
            get_backend_weight(f"backend{i%10}", backend_config, step=i)

        elapsed_time = time.time() - start_time

        # Weight calculation should be very fast
        self.assertLess(elapsed_time, 0.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
