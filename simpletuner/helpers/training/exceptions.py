class MultiDatasetExhausted(Exception):
    pass


class GPUHealthError(Exception):
    """Raised when GPU health monitoring detects a critical fault."""

    def __init__(
        self,
        message: str,
        fault_type: str,
        gpu_index: int | None = None,
        gpu_name: str | None = None,
    ):
        super().__init__(message)
        self.fault_type = fault_type
        self.gpu_index = gpu_index
        self.gpu_name = gpu_name
