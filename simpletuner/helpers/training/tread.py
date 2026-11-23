from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class MaskInfo:
    """Book‑keeping for one TREAD routing window."""

    mask: torch.BoolTensor  # True where token was *dropped*
    ids_keep: torch.LongTensor  # (B, num_keep)
    ids_mask: torch.LongTensor  # (B, num_mask)
    ids_shuffle: torch.LongTensor  # permutation that packs kept tokens first
    ids_restore: torch.LongTensor  # inverse permutation


class TREADRouteMasks:
    """Reuse each route's sampled image tokens across its inclusive range of blocks."""

    def __init__(self, router: "TREADRouter", routes: list[dict], depth: int, force_keep: Optional[torch.Tensor]):
        def index(i: int) -> int:
            return depth + i if i < 0 else i

        self.routes = [(index(r["start_layer_idx"]), index(r["end_layer_idx"]), r["selection_ratio"]) for r in routes]
        self.router = router
        self.force_keep = force_keep
        self.active = None
        self.mask = None

    def for_block(self, index: int, tokens: torch.Tensor) -> Optional[MaskInfo]:
        for route in self.routes:
            start, end, ratio = route
            if start <= index <= end:
                if route != self.active:
                    self.mask = self.router.get_mask(tokens, mask_ratio=ratio, force_keep=self.force_keep)
                    self.active = route
                return self.mask
        self.active = self.mask = None
        return None


class TREADRouter:
    """
    Minimal implementation of the token router used in TREAD.

    Public API
    ----------
    get_mask(x, mask_ratio, ...) -> MaskInfo
        Sample a permutation & binary mask that decides which tokens are kept.
    start_route(x, mask_info) -> x_small
        Apply the permutation and truncate to kept tokens.
    end_route(x_small, mask_info, original_x=None) -> x_full
        Undo the permutation, re‑insert the skipped tokens (either the
        original representations or a mask token), and return a full‑length
        sequence so gradients can flow everywhere.
    """

    def __init__(self, seed: int = 42, device: Any = None):
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    # --------------------------------------------------------------------- #
    # helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _importance(x: torch.Tensor) -> torch.Tensor:
        """
        Per‑token importance score  ∝  L1‑norm of the feature vector.

        Normalised to [0, 1] **per sample** to avoid scale drift.
        """
        # x: (B, S, D)
        mags = x.abs().sum(-1)  # (B,S)
        rng = mags.max(dim=1, keepdim=True)[0] - mags.min(dim=1, keepdim=True)[0]
        mags = (mags - mags.min(dim=1, keepdim=True)[0]) / (rng + 1e-8)
        return mags

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def get_mask(
        self,
        x: torch.Tensor,  # (B, S, D)
        mask_ratio: float = 0.0,
        l1_reg: float = 0.0,
        inverse: bool = False,
        force_keep: Optional[torch.BoolTensor] = None,  # (B, S)
    ) -> MaskInfo:
        """
        Decide which tokens to keep.

        * `mask_ratio` = 0.75  → keep 25 % of the *image* tokens.
        * `force_keep` marks tokens that are **never** allowed to drop.
        """
        B, S, _ = x.shape

        # ---------------------------------------------------------------------
        # 1) book‑keeping for "must‑keep" tokens
        # ---------------------------------------------------------------------
        if force_keep is None:
            force_keep = torch.zeros(B, S, dtype=torch.bool, device=x.device)
        num_force = force_keep.sum(1)  # (B,)  how many per sample

        # overall keep budget (one scalar K for the whole batch)
        base_keep = S - int(round(S * float(mask_ratio)))
        keep_budget = max(base_keep, int(num_force.max()))
        K = keep_budget  # make name explicit

        # ---------------------------------------------------------------------
        # 2) importance + randomness mix
        # ---------------------------------------------------------------------
        noise_dtype = x.real.dtype if x.is_floating_point() or x.is_complex() else torch.get_default_dtype()
        mix = torch.rand((B, S), dtype=noise_dtype, device=x.device, generator=self.generator)
        if l1_reg != 0.0:
            score = self._importance(x)
            if inverse:
                score = 1.0 - score
            mix = (1.0 - l1_reg) * mix + l1_reg * score
        mix = mix.masked_fill(force_keep, -1.0)  # force‑keep ⇒ lowest rank

        # ---------------------------------------------------------------------
        # 3) build permutations
        # ---------------------------------------------------------------------
        ids_shuffle = torch.argsort(mix, dim=1)  # (B, S) smallest first
        ids_keep = ids_shuffle[:, :K]  # (B, K)
        ids_mask = ids_shuffle[:, K:]  # (B, S-K)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # bool mask where True = *masked* (dropped) token
        mask = torch.ones(B, S, dtype=torch.bool, device=x.device)
        mask.scatter_(1, ids_keep, False)

        return MaskInfo(mask, ids_keep, ids_mask, ids_shuffle, ids_restore)

    def start_route(self, x: torch.Tensor, info: MaskInfo) -> torch.Tensor:
        """Gather the kept tokens in routing order."""
        return torch.take_along_dim(x, info.ids_keep.unsqueeze(-1), dim=1)

    def end_route(
        self,
        routed_x: torch.Tensor,  # (B, num_keep, D)
        info: MaskInfo,
        original_x: Optional[torch.Tensor] = None,
        mask_token: float | int = 0.0,
    ) -> torch.Tensor:
        """
        Rebuild a sequence of length S so gradients can flow everywhere.

        If `original_x` is provided we copy the *actual* skipped
        representations back in (recommended for training).  Otherwise we
        fill them with `mask_token`.
        """
        if original_x is None:
            restored = routed_x.new_full((*info.mask.shape, routed_x.shape[2]), mask_token)
        else:
            restored = original_x.to(
                device=routed_x.device, dtype=routed_x.dtype, copy=True, memory_format=torch.contiguous_format
            )

        routed_indices = info.ids_shuffle[:, : routed_x.size(1)].unsqueeze(-1).expand_as(routed_x)
        return restored.scatter_(1, routed_indices, routed_x)
