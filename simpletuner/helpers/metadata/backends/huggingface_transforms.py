"""Map Hugging Face sample references to the shared SQLite transform store."""

from bisect import bisect_right

from simpletuner.helpers.metadata.transform_store import SQLiteTransformStore


class HuggingFaceTransformStore(SQLiteTransformStore):
    def __init__(self, directory, shards, index_from_ref):
        self.boundaries = tuple(shard.stop for shard in shards)
        self.index_from_ref = index_from_ref
        super().__init__(directory, self._partition_for_ref)

    def _partition_for_ref(self, ref):
        try:
            row = self.index_from_ref(ref)
        except ValueError as error:
            raise KeyError(ref) from error
        if row is None or row < 0 or not self.boundaries or row >= self.boundaries[-1]:
            raise KeyError(ref)
        return bisect_right(self.boundaries, row)
