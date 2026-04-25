"""
Paste Apify JSON (array) in the Windows clipboard, then run from repo root:
  py ingest/exports/clipboard_to_json.py

Writes apify_results.json at the repo root (pretty-printed, UTF-8).
"""

import json
import subprocess
import sys
from pathlib import Path

# Same file as a manual Apify download at repo root (see ingest/paths.py)
OUT = Path(__file__).resolve().parents[2] / "apify_results.json"


def main() -> int:
    try:
        s = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"],
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as e:
        print("Clipboard read failed:", e, file=sys.stderr)
        return 1
    s = s.strip()
    if not s:
        print("Clipboard is empty. Copy the JSON array from Apify first.", file=sys.stderr)
        return 1
    try:
        data = json.loads(s)
    except json.JSONDecodeError as e:
        print("Invalid JSON:", e, file=sys.stderr)
        return 1
    if not isinstance(data, list):
        print("Expected a JSON array (list of post objects).", file=sys.stderr)
        return 1
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT.resolve()} ({len(data)} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
