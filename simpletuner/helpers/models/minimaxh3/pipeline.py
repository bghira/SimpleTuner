from .modular_pipeline import MiniMaxH3ModularPipeline


class MiniMaxH3Pipeline(MiniMaxH3ModularPipeline):
    """Conventional SimpleTuner entry point for MiniMax-H3 T2VA/FL2VA."""


__all__ = ["MiniMaxH3Pipeline"]
