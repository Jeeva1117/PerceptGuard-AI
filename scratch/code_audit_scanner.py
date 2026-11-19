import ast
import os
from pathlib import Path
import re

root = Path(".").resolve()
src_dir = root / "src"

findings = []

for py_path in sorted(src_dir.rglob("*.py")):
    rel_path = py_path.relative_to(root).as_posix()
    with open(py_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # 1. AST parse
    try:
        tree = ast.parse(content, filename=str(py_path))
    except SyntaxError as se:
        findings.append({
            "file": rel_path,
            "type": "SYNTAX_ERROR",
            "severity": "CRITICAL",
            "message": str(se)
        })
        continue

    # 2. Check for bare except
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                findings.append({
                    "file": rel_path,
                    "line": node.lineno,
                    "type": "BARE_EXCEPT",
                    "severity": "HIGH",
                    "message": "Bare 'except:' handler suppresses all exceptions including SystemExit and KeyboardInterrupt."
                })
            elif isinstance(node.type, ast.Name) and node.type.id == "BaseException":
                findings.append({
                    "file": rel_path,
                    "line": node.lineno,
                    "type": "BASE_EXCEPTION",
                    "severity": "HIGH",
                    "message": "Catching 'BaseException' directly."
                })

    # 3. Check for hardcoded local absolute paths (C:\ or /Users/ or /home/)
    abs_matches = re.finditer(r'(["\'])([A-Za-z]:\\[^"\']+|/home/[^"\']+|/Users/[^"\']+)\1', content)
    for m in abs_matches:
        line_no = content[:m.start()].count("\n") + 1
        path_str = m.group(2)
        # Exclude docstrings or comments if possible, but flag
        findings.append({
            "file": rel_path,
            "line": line_no,
            "type": "HARDCODED_ABSOLUTE_PATH",
            "severity": "MEDIUM",
            "message": f"Hardcoded absolute filesystem path: {path_str}"
        })

    # 4. Check for bare assert in production src (outside tests)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            # Only flag if not in test function
            findings.append({
                "file": rel_path,
                "line": node.lineno,
                "type": "BARE_ASSERT",
                "severity": "LOW",
                "message": "Use of 'assert' in production source code (assert statements can be optimized away with -O)."
            })

    # 5. Check imports for circular dependencies or side-effects
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Record
            pass

print(f"Scanned {len(list(src_dir.rglob('*.py')))} files in src/")
print(f"Total findings: {len(findings)}")

import json
with open("scratch/code_audit_findings.json", "w", encoding="utf-8") as f:
    json.dump(findings, f, indent=2)

# Print summary by type
from collections import Counter
counts = Counter(f["type"] for f in findings)
for t, cnt in counts.items():
    print(f"  {t:30s}: {cnt}")
