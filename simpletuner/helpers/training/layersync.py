from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class LayerSyncRegularizer:
    """
    A lightweight self-alignment regularizer inspired by LayerSync.

    Aligns a "student" layer to a "teacher" layer using cosine similarity over tokens.
    """

    def __init__(self, config):
        self.enabled = bool(getattr(config, "layersync_enabled", False))
        self.student_block = getattr(config, "layersync_student_block", None)
        self.teacher_block = getattr(config, "layersync_teacher_block", None)
        self.weight = float(getattr(config, "layersync_lambda", 0.0) or 0.0)
        self.detach_teacher = bool(getattr(config, "layersync_detach_teacher", True))

        if self.enabled:
            if self.student_block is None:
                raise ValueError("layersync_student_block must be set when LayerSync is enabled.")
            if self.weight <= 0:
                raise ValueError("layersync_lambda must be greater than zero when LayerSync is enabled.")

    def wants_hidden_states(self) -> bool:
        return self.enabled and self.weight > 0

    def compute_loss(self, hidden_states_buffer: Optional[dict]) -> Tuple[Optional[torch.Tensor], Optional[dict]]:
        if not self.wants_hidden_states():
            return None, None
        if hidden_states_buffer is None:
            raise ValueError("LayerSync enabled but no hidden state buffer was provided.")

        student = hidden_states_buffer.get(f"layer_{self.student_block}")
        teacher_idx = self.teacher_block if self.teacher_block is not None else self.student_block
        teacher = hidden_states_buffer.get(f"layer_{teacher_idx}")

        if student is None:
            raise ValueError(f"LayerSync could not find student layer_{self.student_block} in the buffer.")
        if teacher is None:
            raise ValueError(f"LayerSync could not find teacher layer_{teacher_idx} in the buffer.")

        student_tokens = self._flatten_tokens(student)
        teacher_tokens = self._flatten_tokens(teacher)
        if self.detach_teacher:
            teacher_tokens = teacher_tokens.detach()

        student_tokens = F.normalize(student_tokens, dim=-1)
        teacher_tokens = F.normalize(teacher_tokens, dim=-1)

        similarity = (student_tokens * teacher_tokens).sum(dim=-1)
        similarity_mean = similarity.mean()
        loss = -similarity_mean * self.weight

        logs = {
            "layersync_loss": loss.detach().item(),
            "layersync_similarity": similarity_mean.detach().item(),
        }
        return loss, logs

    @staticmethod
    def _flatten_tokens(hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 4:
            # (B, T, P, D)
            return hidden.reshape(hidden.shape[0], -1, hidden.shape[-1])
        if hidden.ndim == 3:
            # (B, S, D)
            return hidden
        raise ValueError(f"Unsupported hidden state shape for LayerSync: {hidden.shape}")
