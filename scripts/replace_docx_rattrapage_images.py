#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Rattrapage (2).ipynb"
SOURCE_DOCX = ROOT / "Rapport Version Souad 14.06.26.docx"
OUTPUT_DOCX = ROOT / "Rapport Version Souad 14.06.26 - images remplacees.docx"
RECOVERED_DIR = ROOT / "reports" / "figures" / "recovered_rattrapage"

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
}


def notebook_png(cell_index: int, output_index: int) -> bytes:
    nb = json.loads(NOTEBOOK.read_text())
    out = nb["cells"][cell_index]["outputs"][output_index]
    value = out["data"]["image/png"]
    encoded = "".join(value) if isinstance(value, list) else value
    return base64.b64decode(encoded)


def file_png(path: str) -> bytes:
    return (ROOT / path).read_bytes()


def image_size(image_bytes: bytes) -> tuple[int, int]:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        tmp.write(image_bytes)
        tmp.close()
        with Image.open(tmp.name) as img:
            return img.size
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def parent_map(root: ET.Element) -> dict[ET.Element, ET.Element]:
    return {child: parent for parent in root.iter() for child in parent}


def update_extent_for_rid(root: ET.Element, rid: str, source_size: tuple[int, int]) -> None:
    parents = parent_map(root)
    ratio = source_size[1] / source_size[0]
    embed_attr = f"{{{NS['r']}}}embed"

    for blip in root.findall(".//a:blip", NS):
        if blip.attrib.get(embed_attr) != rid:
            continue

        cur = blip
        inline_or_anchor = None
        while cur in parents:
            cur = parents[cur]
            if cur.tag in (f"{{{NS['wp']}}}inline", f"{{{NS['wp']}}}anchor"):
                inline_or_anchor = cur
                break
        if inline_or_anchor is None:
            continue

        wp_extent = inline_or_anchor.find("wp:extent", NS)
        a_ext = inline_or_anchor.find(".//a:xfrm/a:ext", NS)
        if wp_extent is None:
            continue

        width = int(wp_extent.attrib["cx"])
        height = str(round(width * ratio))
        wp_extent.attrib["cy"] = height
        if a_ext is not None:
            a_ext.attrib["cx"] = str(width)
            a_ext.attrib["cy"] = height


def main() -> None:
    RECOVERED_DIR.mkdir(parents=True, exist_ok=True)

    replacements = {
        "word/media/image12.png": ("fig_13_3_gradcam_efficientnetb0_normal.png", "rId53", notebook_png(32, 0)),
        "word/media/image3.png": ("fig_13_3_gradcam_efficientnetb0_lung_opacity.png", "rId54", notebook_png(32, 3)),
        "word/media/image26.png": ("fig_13_4_gradcam_resnet50_normal.png", "rId55", notebook_png(31, 0)),
        # The notebook's only ResNet50 output predicted as COVID is actually a true Normal case.
        # Keep this report section clinically consistent by using the existing true-COVID figure.
        "word/media/image28.png": (
            "fig_13_4_gradcam_resnet50_covid.png",
            "rId56",
            file_png("reports/figures/gradcam_resnet50_COVID.png"),
        ),
        "word/media/image20.png": ("fig_13_4_gradcam_resnet50_lung_opacity.png", "rId57", notebook_png(31, 3)),
        "word/media/image21.png": ("fig_14_3_shap_resnet50.png", "rId62", notebook_png(39, 0)),
    }

    for recovered_name, _, image_bytes in replacements.values():
        (RECOVERED_DIR / recovered_name).write_bytes(image_bytes)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        with zipfile.ZipFile(SOURCE_DOCX) as zin:
            zin.extractall(tmp)

        document_xml = tmp / "word" / "document.xml"
        tree = ET.parse(document_xml)
        root = tree.getroot()

        for media_path, (_, rid, image_bytes) in replacements.items():
            (tmp / media_path).write_bytes(image_bytes)
            update_extent_for_rid(root, rid, image_size(image_bytes))

        ET.register_namespace("w", "http://schemas.openxmlformats.org/wordprocessingml/2006/main")
        ET.register_namespace("r", NS["r"])
        ET.register_namespace("a", NS["a"])
        ET.register_namespace("wp", NS["wp"])
        tree.write(document_xml, encoding="UTF-8", xml_declaration=True)

        if OUTPUT_DOCX.exists():
            OUTPUT_DOCX.unlink()
        shutil.make_archive(str(OUTPUT_DOCX.with_suffix("")), "zip", tmp)
        OUTPUT_DOCX.with_suffix(".zip").rename(OUTPUT_DOCX)

    print(f"Created: {OUTPUT_DOCX}")
    print(f"Recovered images: {RECOVERED_DIR}")


if __name__ == "__main__":
    main()
