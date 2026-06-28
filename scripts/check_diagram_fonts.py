#!/usr/bin/env python3
"""Check that course Excalidraw diagrams keep the Excalifont style."""

from __future__ import annotations

import json
import sys
from pathlib import Path


EXPECTED_FONT_FAMILY = 5


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    diagram_dir = root / "diagrams"
    issues: list[str] = []
    text_count = 0

    for path in sorted(diagram_dir.glob("*.excalidraw")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for element in data.get("elements", []):
            if element.get("isDeleted") or element.get("type") != "text":
                continue
            text_count += 1
            font_family = element.get("fontFamily")
            if font_family != EXPECTED_FONT_FAMILY:
                text = str(element.get("text", "")).replace("\n", "\\n")
                issues.append(
                    f"{path.relative_to(root)}: text={text!r} "
                    f"fontFamily={font_family!r}, expected {EXPECTED_FONT_FAMILY}"
                )

    print(f"checked_diagrams={len(list(diagram_dir.glob('*.excalidraw')))}")
    print(f"checked_text_elements={text_count}")
    print(f"font_issues={len(issues)}")
    for issue in issues:
        print(issue)

    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
