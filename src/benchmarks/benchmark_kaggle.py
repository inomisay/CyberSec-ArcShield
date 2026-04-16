"""Compatibility wrapper for legacy imports."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("src.benchmarks.engines.benchmark_kaggle", run_name="__main__")
else:
    from src.benchmarks.engines.benchmark_kaggle import *  # noqa: F401,F403
