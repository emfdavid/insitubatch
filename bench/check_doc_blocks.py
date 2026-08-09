"""Run every ```python block of a markdown file in its OWN process.

Concatenating them (the earlier mistake) lets a later block borrow names from an
earlier one, so a block that is broken when copied on its own still passes.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tempfile

failed = 0
for path in sys.argv[1:]:
    md = pathlib.Path(path).read_text()
    blocks = re.findall(r"```python\n(.*?)```", md, re.S)
    print(f"\n=== {path}: {len(blocks)} python block(s) ===")
    for i, block in enumerate(blocks, 1):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(block)
            tmp = fh.name
        proc = subprocess.run([sys.executable, tmp], capture_output=True, text=True, timeout=600)
        first = block.strip().splitlines()[0][:58]
        if proc.returncode == 0:
            print(f"  block {i}: OK    | {first}")
        else:
            failed += 1
            last = [ln for ln in proc.stderr.strip().splitlines() if ln.strip()][-1:]
            print(f"  block {i}: FAIL  | {first}\n            {last[0][:110] if last else ''}")

print(f"\n{failed} failing block(s)")
sys.exit(1 if failed else 0)
