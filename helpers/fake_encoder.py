from transformers.modeling_utils import PreTrainedModel
class FakeTextEncoder(PreTrainedModel):
    """Just a simple class to behave as a text encoder.
    """
    def __init__(self, text_encoder, dtype):
        self.config = text_encoder.config
        self._dtype = dtype  # Use a private attribute to store the value

    @property
    def dtype(self):  # Define a getter for the dtype property
        return self._dtype

    @dtype.setter
    def dtype(self, value):  # Define a setter for the dtype property
        self._dtype = value