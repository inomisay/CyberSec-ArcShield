"""Compatibility wrapper for legacy imports."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("src.benchmarks.analysis.attack_type_heatmap", run_name="__main__")
else:
    from src.benchmarks.analysis.attack_type_heatmap import *  # noqa: F401,F403
