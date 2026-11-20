from contextlib import contextmanager


@contextmanager
def dataset_worker(context, target, cancelled, *args):
    """Receive progress until a standalone dataset worker exits."""
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(writer, cancelled, *args))

    def messages():
        while True:
            try:
                yield reader.recv()
            except EOFError:
                break
        process.join()
        if process.exitcode:
            raise RuntimeError(f"Dataset worker exited with code {process.exitcode}.")

    try:
        with reader, writer:
            process.start()
            writer.close()
            yield messages()
    finally:
        cancelled.set()
        if process.pid is not None:
            process.join()
        process.close()
