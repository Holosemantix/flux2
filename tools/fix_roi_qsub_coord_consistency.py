#!/usr/bin/env python3
"""Fix ROI q-sub RoPE coordinate consistency in code/ files.

The FLUX/FLUX.2 image ids use integer grid coordinates (arange) as native token
centers. The first q-sub patch accidentally changed some K/virtual coordinate
helpers to y+0.5 .. y2-0.5, while q_sub positions still probe around integer
centers. That makes q and K live in slightly different H/W coordinate conventions.

This script makes all ROI helper coordinates consistent again:
  - native rectangle K ids: y1 .. y2-1, x1 .. x2-1;
  - virtual ROI ids: y1 .. y2-1, x1 .. x2-1;
  - pe3 maps source endpoint range to target endpoint range;
  - ref_hr high-res ids map into target endpoint range.

Run from repo root:

    python tools/fix_roi_qsub_coord_consistency.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRANSFORMER = ROOT / "code" / "transformer_flux2.py"
REFINE = ROOT / "code" / "refine_model.py"


def patch_transformer(text: str) -> str:
    # Existing-model convention: image ids are integer grid centers, not +0.5 centers.
    replacements = [
        (
            "    ys = torch.linspace(float(y1) + 0.5, float(y2) - 0.5, P, device=device)\n"
            "    xs = torch.linspace(float(x1) + 0.5, float(x2) - 0.5, P, device=device)\n",
            "    # FLUX image ids use integer grid coordinates as native token centers.\n"
            "    # For half-open bbox [y1,y2), the endpoint centers are y1 and y2-1.\n"
            "    ys = torch.linspace(float(y1), float(y2 - 1), P, device=device)\n"
            "    xs = torch.linspace(float(x1), float(x2 - 1), P, device=device)\n",
        ),
        (
            "    ys = torch.linspace(float(y1) + 0.5, float(y2) - 0.5, h, device=device)\n"
            "    xs = torch.linspace(float(x1) + 0.5, float(x2) - 0.5, w, device=device)\n",
            "    # Native K tokens use the same integer-center convention as q_sub.\n"
            "    ys = torch.linspace(float(y1), float(y2 - 1), h, device=device)\n"
            "    xs = torch.linspace(float(x1), float(x2 - 1), w, device=device)\n",
        ),
        (
            "        cy = (oy1 + oy2) / 2.0\n"
            "        cx = (ox1 + ox2) / 2.0\n"
            "        ry = max(y2 - y1, 1e-6) / max(oy2 - oy1, 1e-6)   # = 扩大倍数\n"
            "        rx = max(x2 - x1, 1e-6) / max(ox2 - ox1, 1e-6)\n",
            "        cy = (oy1 + oy2 - 1) / 2.0\n"
            "        cx = (ox1 + ox2 - 1) / 2.0\n"
            "        ry = max((y2 - y1) - 1, 1e-6) / max((oy2 - oy1) - 1, 1e-6)   # endpoint range ratio\n"
            "        rx = max((x2 - x1) - 1, 1e-6) / max((ox2 - ox1) - 1, 1e-6)\n",
        ),
        (
            "        ry = max(y2 - y1, 1e-6) / max(oy2 - oy1, 1e-6)\n"
            "        rx = max(x2 - x1, 1e-6) / max(ox2 - ox1, 1e-6)\n",
            "        ry = max((y2 - y1) - 1, 1e-6) / max((oy2 - oy1) - 1, 1e-6)\n"
            "        rx = max((x2 - x1) - 1, 1e-6) / max((ox2 - ox1) - 1, 1e-6)\n",
        ),
        (
            "        ny = (gy - y1) / max(y2 - y1, 1e-6)\n"
            "        nx = (gx - x1) / max(x2 - x1, 1e-6)\n"
            "        gy = ty1 + ny * (ty2 - ty1)\n"
            "        gx = tx1 + nx * (tx2 - tx1)\n",
            "        ny = (gy - float(y1)) / max((y2 - y1) - 1, 1e-6)\n"
            "        nx = (gx - float(x1)) / max((x2 - x1) - 1, 1e-6)\n"
            "        gy = float(ty1) + ny * max((ty2 - ty1) - 1, 1e-6)\n"
            "        gx = float(tx1) + nx * max((tx2 - tx1) - 1, 1e-6)\n",
        ),
        (
            "        ny = (gy - (float(y1) + 0.5)) / max(y2 - y1, 1e-6)\n"
            "        nx = (gx - (float(x1) + 0.5)) / max(x2 - x1, 1e-6)\n"
            "        gy = ty1 + 0.5 + ny * max(ty2 - ty1, 1e-6)\n"
            "        gx = tx1 + 0.5 + nx * max(tx2 - tx1, 1e-6)\n",
            "        ny = (gy - float(y1)) / max((y2 - y1) - 1, 1e-6)\n"
            "        nx = (gx - float(x1)) / max((x2 - x1) - 1, 1e-6)\n"
            "        gy = float(ty1) + ny * max((ty2 - ty1) - 1, 1e-6)\n"
            "        gx = float(tx1) + nx * max((tx2 - tx1) - 1, 1e-6)\n",
        ),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def patch_refine(text: str) -> str:
    return text.replace(
        "        # bbox 是半开区间 [y1,y2) / [x1,x2)，用 token-center 坐标避免 RoPE 端点拉伸\n"
        "        ys = torch.linspace(float(y1) + 0.5, float(y2) - 0.5, gh)\n"
        "        xs = torch.linspace(float(x1) + 0.5, float(x2) - 0.5, gw)\n",
        "        # FLUX image ids use integer grid coordinates as token centers.\n"
        "        # Map high-res ref_hr tokens into the target bbox endpoint range y1..y2-1.\n"
        "        ys = torch.linspace(float(y1), float(y2 - 1), gh)\n"
        "        xs = torch.linspace(float(x1), float(x2 - 1), gw)\n",
    )


def patch_file(path: Path, fn) -> bool:
    text = path.read_text(encoding="utf-8")
    new = fn(text)
    if new == text:
        print(f"unchanged: {path}")
        return False
    path.write_text(new, encoding="utf-8")
    print(f"patched:   {path}")
    return True


def main() -> None:
    changed = False
    changed |= patch_file(TRANSFORMER, patch_transformer)
    changed |= patch_file(REFINE, patch_refine)
    if changed:
        print("\nCoordinate convention fixed. Run py_compile next.")
    else:
        print("\nCoordinate convention already consistent.")


if __name__ == "__main__":
    main()
