"""
Duplicator is needed to copy metadata from one backend to another.

This is useful for instance, when we auto-generate a conditioning dataset based on training split.

Instead of scanning the training split again, we can just copy the metadata from the training split to the conditioning split.
"""

import logging
from helpers.training.multi_process import _get_rank

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO if _get_rank() == 0 else logging.WARNING)


class DatasetDuplicator:

    @staticmethod
    def copy_metadata(source_backend, target_backend):
        """
        Copies metadata from source backend to target backend.

        Args:
            source_backend: The backend from which to copy metadata.
            target_backend: The backend to which metadata will be copied.
        """
        source_metadata = source_backend.get("metadata_backend", None)
        target_metadata = target_backend.get("metadata_backend", None)
        if source_metadata is None or target_metadata is None:
            raise ValueError(
                "Both source and target backends must have metadata_backend defined."
            )

        logger.info("Reloading source metadata caches...")
        source_metadata.reload_cache(set_config=False)
        logger.info("Reloading target metadata caches...")
        target_metadata.reload_cache(set_config=False)

        # Write the new metadata to the target backend
        logger.info("Copying metadata from source to target backend...")
        target_metadata.set_metadata(
            metadata=source_metadata.get_metadata(),
            update_json=True,
        )
        logger.info("Metadata copied successfully.")
        source_metadata.print_debug_info()
        target_metadata.print_debug_info()
