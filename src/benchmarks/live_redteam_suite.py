"""Compatibility wrapper for legacy imports."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("src.benchmarks.suites.live_redteam_suite", run_name="__main__")
else:
    from src.benchmarks.suites.live_redteam_suite import *  # noqa: F401,F403
