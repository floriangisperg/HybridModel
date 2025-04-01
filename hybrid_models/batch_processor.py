"""Batch processing functionality for efficient model evaluation."""
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Any, Callable, Optional
from jaxtyping import Array, Float
import time
from functools import partial
from .profiling import timed


class BatchProcessor:
    """
    Efficiently process multiple datasets or simulations in batches.
    Helps with memory management and parallelization for large workloads.
    """

    def __init__(
            self,
            model: Any,
            process_fn: Callable,
            batch_size: int = 10,
            use_pmap: bool = False
    ):
        """
        Initialize batch processor.

        Args:
            model: The model to process with
            process_fn: Function that takes (model, item) and returns a result
            batch_size: Number of items to process in each batch
            use_pmap: Whether to use pmap for multi-device parallelization
        """
        self.model = model
        self.process_fn = process_fn
        self.batch_size = batch_size
        self.use_pmap = use_pmap

        # Prepare JIT-compiled processing function
        self._jitted_process = jax.jit(
            lambda model, item: process_fn(model, item),
            static_argnames=['model']
        )

        # Optionally prepare PMAP function if using multi-device parallelization
        if use_pmap and jax.device_count() > 1:
            self._pmap_process = jax.pmap(
                lambda models, items: jax.vmap(process_fn)(models, items),
                in_axes=(None, 0)
            )

    @timed("process_items")
    def process(
            self,
            items: List[Any],
            show_progress: bool = True
    ) -> List[Any]:
        """
        Process a list of items in batches.

        Args:
            items: List of items to process
            show_progress: Whether to show progress information

        Returns:
            List of results corresponding to the input items
        """
        results = []
        num_batches = (len(items) + self.batch_size - 1) // self.batch_size

        start_time = time.time()
        total_processed = 0

        for batch_idx in range(num_batches):
            # Get the current batch
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(items))
            batch_items = items[batch_start:batch_end]

            # Process the batch
            if self.use_pmap and jax.device_count() > 1:
                # Pad batch to multiple of device count
                device_count = jax.device_count()
                pad_size = (device_count - len(batch_items) % device_count) % device_count
                padded_batch = batch_items + [batch_items[0]] * pad_size if pad_size > 0 else batch_items

                # Reshape for pmap
                reshaped_batch = [
                    padded_batch[i:i + device_count]
                    for i in range(0, len(padded_batch), device_count)
                ]

                # Process with pmap
                batch_results = []
                for sub_batch in reshaped_batch:
                    sub_results = self._pmap_process(
                        jnp.array([self.model] * device_count),
                        jnp.array(sub_batch)
                    )
                    batch_results.extend(sub_results)

                # Remove padding
                batch_results = batch_results[:len(batch_items)]
            else:
                # Process sequentially with JIT
                batch_results = [
                    self._jitted_process(self.model, item)
                    for item in batch_items
                ]

            # Extend results
            results.extend(batch_results)

            # Update progress
            total_processed += len(batch_items)

            if show_progress:
                elapsed = time.time() - start_time
                items_per_sec = total_processed / elapsed if elapsed > 0 else 0
                remaining = (len(items) - total_processed) / items_per_sec if items_per_sec > 0 else 0

                print(f"Processed batch {batch_idx + 1}/{num_batches} - "
                      f"{total_processed}/{len(items)} items - "
                      f"{items_per_sec:.2f} items/sec - "
                      f"Est. remaining: {remaining:.2f}s")

        if show_progress:
            total_time = time.time() - start_time
            print(f"Completed processing {len(items)} items in {total_time:.2f}s "
                  f"({len(items) / total_time:.2f} items/sec)")

        return results


class DatasetBatchProcessor(BatchProcessor):
    """Specialized batch processor for model datasets."""

    def __init__(
            self,
            model: Any,
            batch_size: int = 10,
            use_pmap: bool = False
    ):
        """
        Initialize dataset batch processor.

        Args:
            model: The model to process with
            batch_size: Number of datasets to process in each batch
            use_pmap: Whether to use pmap for multi-device parallelization
        """
        super().__init__(
            model=model,
            process_fn=self._process_dataset,
            batch_size=batch_size,
            use_pmap=use_pmap
        )

    @staticmethod
    def _process_dataset(model, dataset):
        """Process a single dataset with the model."""
        solution = model.solve(
            initial_state=dataset['initial_state'],
            t_span=(dataset['times'][0], dataset['times'][-1]),
            evaluation_times=dataset['times'],
            args={'time_dependent_inputs': dataset.get('time_dependent_inputs', {})}
        )

        # Calculate metrics if true values are available
        metrics = {}
        for state_name in model.state_names:
            true_key = f"{state_name}_true"
            if true_key in dataset and state_name in solution:
                y_true = dataset[true_key]
                y_pred = solution[state_name]

                # Simple MSE metric
                mse = jnp.mean(jnp.square(y_pred - y_true))
                metrics[state_name] = {'mse': float(mse)}

        return {
            'solution': solution,
            'metrics': metrics
        }