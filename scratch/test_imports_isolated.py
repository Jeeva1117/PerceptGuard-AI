import subprocess
import sys
from pathlib import Path

root = Path(".").resolve()
src_dir = root / "src"

modules = []
for p in sorted(src_dir.rglob("*.py")):
    rel = p.relative_to(root).with_suffix("")
    mod_name = ".".join(rel.parts)
    modules.append(mod_name)

print(f"Testing isolated import for {len(modules)} modules...")

failures = []
slow_imports = []

for mod in modules:
    code = f"import time; t0 = time.perf_counter(); import {mod}; dt = time.perf_counter() - t0; print(f'OK:{{dt:.3f}}')"
    cmd = [sys.executable, "-c", code]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10, cwd=str(root))
        if res.returncode != 0:
            failures.append({
                "module": mod,
                "error": res.stderr.strip() or res.stdout.strip()
            })
            print(f"  [FAIL] {mod}: {res.stderr.strip().splitlines()[-1] if res.stderr.strip() else 'Exit code non-zero'}")
        else:
            out_str = res.stdout.strip()
            if "OK:" in out_str:
                dt_str = out_str.split("OK:")[1].strip()
                dt = float(dt_str)
                if dt > 1.5:
                    slow_imports.append((mod, dt))
                    print(f"  [SLOW] {mod}: {dt:.3f}s")
    except subprocess.TimeoutExpired:
        failures.append({
            "module": mod,
            "error": "Import timed out (> 10s)"
        })
        print(f"  [TIMEOUT] {mod}")

print(f"\nImport test finished:")
print(f"  Total tested: {len(modules)}")
print(f"  Passed: {len(modules) - len(failures)}")
print(f"  Failed: {len(failures)}")
print(f"  Slow imports (>1.5s): {len(slow_imports)}")

if failures:
    print("\nDetailed Failures:")
    for f in failures:
        print(f"  {f['module']}: {f['error']}")
