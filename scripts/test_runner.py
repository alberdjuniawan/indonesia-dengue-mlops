from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Add project root and src to path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "src"))

test_modules = [
    "tests.features.test_features",
    "tests.modeling.test_modeling",
    "tests.monitoring.test_monitoring",
    "tests.serving.test_serving",
    "tests.ingestion.population.test_validator",
    "tests.ingestion.population.test_service",
    "tests.ingestion.population.test_parser",
    "tests.ingestion.population.test_discovery",
    "tests.ingestion.skdr.test_validator",
    "tests.ingestion.skdr.test_ocr_parser",
    "tests.ingestion.skdr.test_discovery",
]


def run_all_tests() -> bool:
    total_passed = 0
    total_failed = 0
    total_skipped = 0

    print("=" * 70)
    print("RUNNING UNIT TEST SUITE")
    print("=" * 70)

    for mod_name in test_modules:
        print(f"\n[{mod_name}]")
        try:
            import importlib

            mod = importlib.import_module(mod_name)
        except (ImportError, ModuleNotFoundError, AttributeError, ValueError) as e:
            print(f"  FAILED to import {mod_name}: {e}")
            total_failed += 1
            continue

        test_funcs = [
            getattr(mod, attr)
            for attr in dir(mod)
            if attr.startswith("test_") and callable(getattr(mod, attr))
        ]

        for func in test_funcs:
            fname = func.__name__
            try:
                # Handle test functions that take tmp_path or no args
                import inspect

                sig = inspect.signature(func)
                if len(sig.parameters) == 0:
                    func()
                elif "tmp_path" in sig.parameters:
                    import tempfile

                    with tempfile.TemporaryDirectory() as td:
                        func(Path(td))
                else:
                    print(f"  SKIPPED {fname} (unsupported fixture)")
                    total_skipped += 1
                    continue

                print(f"  PASSED {fname}")
                total_passed += 1
            except (AssertionError, ValueError, TypeError, KeyError) as e:
                print(f"  FAILED {fname}: {type(e).__name__}: {e}")
                traceback.print_exc()
                total_failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {total_passed} Passed, {total_failed} Failed, {total_skipped} Skipped")
    print("=" * 70)
    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
