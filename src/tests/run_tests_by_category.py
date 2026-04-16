#!/usr/bin/env python3
"""
Run tests by category and display results organized in console with statistics.
Each test category (Core Logic, Data Layer, API, Benchmark, Dashboard, Extension, Reliability)
is executed separately with clear section headers and pass/fail counts.
"""
# python src/tests/run_tests_by_category.py
import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List


class Tee:
    """Write stdout to multiple streams at once (console + file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

@dataclass
class CategoryResult:
    name: str
    path: str
    passed: int = 0
    failed: int = 0
    errors: int = 0
    duration: float = 0.0
    
    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors
    
    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.passed / self.total) * 100

# Define test categories
CATEGORIES = [
    CategoryResult("Core Logic", "src/tests/core_logic"),
    CategoryResult("Data Layer", "src/tests/data_layer"),
    CategoryResult("API", "src/tests/api"),
    CategoryResult("Benchmark", "src/tests/benchmark"),
    CategoryResult("Dashboard", "src/tests/dashboard"),
    CategoryResult("Extension", "src/tests/extension"),
    CategoryResult("Reliability", "src/tests/reliability"),
]

def run_category_tests(category: CategoryResult) -> CategoryResult:
    """Run tests for a specific category and capture results."""
    cmd = [
        sys.executable, "-m", "pytest",
        category.path,
        "-m", "unit",
        "-v",
        "--tb=short",
        "-q"
    ]
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        end_time = time.time()
        category.duration = end_time - start_time
        
        output = result.stdout + result.stderr
        
        # Parse output for counts
        lines = output.split('\n')
        for line in lines:
            if 'passed' in line or 'failed' in line or 'error' in line:
                # Extract counts from summary line
                if 'passed' in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if 'passed' in part and i > 0:
                                category.passed = int(parts[i-1])
                            elif 'failed' in part and i > 0:
                                category.failed = int(parts[i-1])
                            elif 'error' in part and i > 0:
                                category.errors = int(parts[i-1])
                    except (ValueError, IndexError):
                        pass
        
        return category, output
    except subprocess.TimeoutExpired:
        print(f"⏱️  Timeout running tests for {category.name}")
        return category, ""
    except Exception as e:
        print(f"❌ Error running tests for {category.name}: {e}")
        return category, ""

def print_header(text: str, width: int = 80):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)

def print_category_summary(category: CategoryResult):
    """Print summary stats for a category with visual bar."""
    status = "✅" if category.failed == 0 and category.errors == 0 else "⚠️ "
    
    # Create visual bar
    total = category.total
    if total > 0:
        passed_width = int((category.passed / total) * 40)
        failed_width = int((category.failed / total) * 40)
        error_width = 40 - passed_width - failed_width
        
        bar = "█" * passed_width + "⚠" * failed_width + "✖" * error_width
    else:
        bar = "(no tests)"
    
    print(f"\n{status} {category.name:20} | {bar:40} | {category.passed:3} ✓ {category.failed:3} ✗ {category.errors:3} E | ⏱️  {category.duration:.2f}s")
    if total > 0:
        print(f"{'':24} | Success: {category.success_rate:5.1f}% ({category.passed}/{category.total} tests)")

def run_report():
    """Run the report and print results."""
    print_header("TEST EXECUTION BY SCENARIO CATEGORY", 80)
    print("Running tests for each category with detailed output...\n")
    
    all_results: List[tuple] = []
    total_passed = 0
    total_failed = 0
    total_errors = 0
    total_duration = 0.0
    
    # Run tests for each category
    for category in CATEGORIES:
        print(f"\n⏳ Running {category.name} tests...")
        updated_category, output = run_category_tests(category)
        
        # Update counts
        total_passed += updated_category.passed
        total_failed += updated_category.failed
        total_errors += updated_category.errors
        total_duration += updated_category.duration
        
        all_results.append((updated_category, output))
        
        # Print category details
        if updated_category.total > 0:
            print(f"   📊 Results: {updated_category.passed} passed, {updated_category.failed} failed, {updated_category.errors} errors | ⏱️  {updated_category.duration:.2f}s")
        else:
            print(f"   💭 No tests collected in this category")
    
    # Print detailed results for each category
    print_header("DETAILED RESULTS BY CATEGORY", 80)
    
    for category, output in all_results:
        print_category_summary(category)
        
        # Show detailed output if there were issues
        if (category.failed > 0 or category.errors > 0) and output:
            print(f"\n   Details:")
            for line in output.split('\n')[:15]:  # Show first 15 lines
                if line.strip():
                    print(f"   {line}")
            if output.count('\n') > 15:
                print(f"   ... (truncated, run category test directly for full output)")
    
    # Print overall summary
    print_header("OVERALL SUMMARY", 80)
    
    total_tests = total_passed + total_failed + total_errors
    overall_success = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n📈 Overall Statistics:")
    print(f"   Total Tests: {total_tests}")
    print(f"   ✅ Passed:   {total_passed}")
    print(f"   ❌ Failed:   {total_failed}")
    print(f"   ⚠️  Errors:   {total_errors}")
    print(f"   📊 Success Rate: {overall_success:.1f}%")
    print(f"   ⏱️  Total Time: {total_duration:.2f}s")
    
    # Visual summary
    if total_tests > 0:
        passed_width = int((total_passed / total_tests) * 50)
        failed_width = int((total_failed / total_tests) * 50)
        error_width = 50 - passed_width - failed_width
        
        bar = "█" * passed_width + "⚠" * failed_width + "✖" * error_width
        print(f"\n   {bar}")
    
    print("\n" + "=" * 80)
    
    # Return exit code based on results
    return 0 if (total_failed == 0 and total_errors == 0) else 1


def main():
    """Main entry point with optional txt output path."""
    parser = argparse.ArgumentParser(description="Run category tests and save console report to txt")
    parser.add_argument(
        "--output",
        help="Optional output txt path (default: logs/test_results_by_category.txt)",
    )
    args = parser.parse_args()

    output_path = args.output or os.path.join("logs", "test_results_by_category.txt")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    original_stdout = sys.stdout
    try:
        with open(output_path, "w", encoding="utf-8") as report_file:
            sys.stdout = Tee(original_stdout, report_file)
            exit_code = run_report()
    finally:
        sys.stdout = original_stdout

    print(f"Report saved to: {output_path}")
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
