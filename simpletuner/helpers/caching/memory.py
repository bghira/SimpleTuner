def reclaim_memory():
    import gc

    import torch

    if torch.cuda.is_available():
        gc.collect()
        try:
            current_device = torch.cuda.current_device()
        except Exception:
            current_device = None
        for device_idx in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(device_idx)
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                continue
        if current_device is not None:
            try:
                torch.cuda.set_device(current_device)
            except Exception:
                pass

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()
        gc.collect()
