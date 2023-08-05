class FakeTextEncoder:
    """Just a simple class to behave as a text encoder.
    """
    def __init__(self, text_encoder, dtype):
        self.config = text_encoder.config
        self.dtype = dtype
