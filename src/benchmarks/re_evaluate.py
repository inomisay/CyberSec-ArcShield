"""Compatibility wrapper for legacy imports."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("src.benchmarks.tools.re_evaluate", run_name="__main__")
else:
    from src.benchmarks.tools.re_evaluate import *  # noqa: F401,F403
