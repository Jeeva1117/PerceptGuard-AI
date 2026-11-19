import importlib
import time
import sys
from pathlib import Path

root = Path(".").resolve()
sys.path.insert(0, str(root))
src_dir = root / "src"

modules = []
for p in sorted(src_dir.rglob("*.py")):
    rel = p.relative_to(root).with_suffix("")
    mod_name = ".".join(rel.parts)
    modules.append(mod_name)

print(f"Testing in-process import for {len(modules)} modules...")

failures = []
t0_all = time.perf_counter()

for mod in modules:
    t0 = time.perf_counter()
    try:
        importlib.import_module(mod)
        dt = time.perf_counter() - t0
        # print(f"  [OK] {mod} ({dt*1000:.1f}ms)")
    except Exception as e:
        failures.append((mod, str(e)))
        print(f"  [FAIL] {mod}: {e}")

dt_total = time.perf_counter() - t0_all
print(f"\nDone in {dt_total:.2f}s!")
print(f"Passed: {len(modules) - len(failures)} / {len(modules)}")
print(f"Failed: {len(failures)}")
