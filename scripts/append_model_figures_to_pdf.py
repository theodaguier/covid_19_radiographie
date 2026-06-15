from __future__ import annotations

from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PDF = ROOT / "Rapport Version Souad 14.06.26.pdf"
OUT_PDF = ROOT / "Rapport Version Souad 14.06.26 - avec graphiques modeles.pdf"
TMP_APPENDIX = ROOT / "reports" / "figures" / "_annexe_graphiques_modeles.pdf"
FIG_DIR = ROOT / "reports" / "figures"

PAGE_W, PAGE_H = 2550, 3300  # US Letter at 300 dpi
MARGIN_X = 210
MARGIN_TOP = 180
MARGIN_BOTTOM = 180
ACCENT = (27, 77, 114)
TEXT = (30, 35, 40)
MUTED = (95, 105, 115)
RULE = (210, 218, 226)
BG = (255, 255, 255)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


F_TITLE = font(66, True)
F_H1 = font(48, True)
F_H2 = font(37, True)
F_BODY = font(31)
F_CAP = font(27)
F_SMALL = font(24)


FIGURES = [
    {
        "title": "Synthese comparative des modeles",
        "items": [
            (
                "classes_avant_apres_modeles.png",
                "Comparaison globale des performances par classe et par modele. Cette figure permet de justifier le choix du modele final en reliant les scores aux quatre diagnostics.",
            ),
            (
                "comparison_confusions.png",
                "Matrices de confusion comparees. Les erreurs residuelles montrent surtout la proximite visuelle entre COVID et Lung Opacity.",
            ),
        ],
    },
    {
        "title": "EfficientNetB0 Fine-Tuned",
        "items": [
            (
                "learning_curves_efficientnetb0.png",
                "Courbes d'apprentissage d'EfficientNetB0 : evolution de l'accuracy et de la loss sur train et validation.",
            ),
            (
                "confusion_efficientnetb0.png",
                "Matrice de confusion d'EfficientNetB0 sur le test set fige.",
            ),
            (
                "roc_pr_efficientnetb0.png",
                "Courbes ROC et Precision-Recall par classe pour EfficientNetB0.",
            ),
        ],
    },
    {
        "title": "ResNet50 Fine-Tuned",
        "items": [
            (
                "learning_curves_resnet50.png",
                "Courbes d'apprentissage de ResNet50 : stabilite de l'entrainement et generalisation.",
            ),
            (
                "confusion_resnet50.png",
                "Matrice de confusion de ResNet50, modele quantitatif principal du rapport.",
            ),
            (
                "roc_pr_resnet50.png",
                "Courbes ROC et Precision-Recall par classe pour ResNet50.",
            ),
        ],
    },
    {
        "title": "Interpretabilite : Grad-CAM et SHAP",
        "items": [
            (
                "gradcam_efficientnetb0_COVID.png",
                "Grad-CAM d'EfficientNetB0 sur un cas COVID : visualisation des zones influencant la prediction.",
            ),
            (
                "gradcam_resnet50_COVID.png",
                "Grad-CAM de ResNet50 sur un cas COVID : attention plus focalisee et interpretable.",
            ),
            (
                "shap_efficientnetb0.png",
                "SHAP applique a EfficientNetB0 : contribution des zones de l'image a la decision du modele.",
            ),
        ],
    },
    {
        "title": "Controle du biais par masquage pulmonaire",
        "items": [
            (
                "confusion_resnet50_masked.png",
                "Matrice de confusion du ResNet50 entraine sur images masquees.",
            ),
            (
                "roc_pr_resnet50_masked.png",
                "Courbes ROC et Precision-Recall du ResNet50 masque.",
            ),
            (
                "gradcam_resnet50_masked_COVID_0.png",
                "Exemple Grad-CAM apres masquage : l'attention est davantage contrainte aux champs pulmonaires.",
            ),
        ],
    },
]


def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], fnt, fill=TEXT, max_width=78, line_gap=8) -> int:
    x, y = xy
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=max_width))
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_gap
    return y


def paste_fit(page: Image.Image, path: Path, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    max_w, max_h = x1 - x0, y1 - y0
    img = Image.open(path).convert("RGB")
    scale = min(max_w / img.width, max_h / img.height)
    new_w, new_h = int(img.width * scale), int(img.height * scale)
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    x = x0 + (max_w - new_w) // 2
    y = y0 + (max_h - new_h) // 2
    page.paste(resized, (x, y))
    return x, y, x + new_w, y + new_h


def new_page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    page = Image.new("RGB", (PAGE_W, PAGE_H), BG)
    return page, ImageDraw.Draw(page)


def add_header(draw: ImageDraw.ImageDraw, title: str, page_no: int) -> int:
    y = MARGIN_TOP
    draw.text((MARGIN_X, y), title, font=F_H1, fill=ACCENT)
    draw.text((PAGE_W - MARGIN_X - 220, y + 14), f"Annexe {page_no}", font=F_SMALL, fill=MUTED)
    y += 78
    draw.line((MARGIN_X, y, PAGE_W - MARGIN_X, y), fill=RULE, width=3)
    return y + 58


def cover_page() -> Image.Image:
    page, draw = new_page()
    y = MARGIN_TOP + 350
    draw.text((MARGIN_X, y), "Annexe", font=F_TITLE, fill=ACCENT)
    y += 95
    draw.text((MARGIN_X, y), "Graphiques des modeles", font=F_TITLE, fill=TEXT)
    y += 110
    y = draw_wrapped(
        draw,
        "Figures ajoutees pour illustrer les performances, la comparaison et l'interpretabilite des modeles de classification de radiographies pulmonaires.",
        (MARGIN_X, y),
        F_BODY,
        fill=MUTED,
        max_width=86,
        line_gap=12,
    )
    y += 95
    draw.line((MARGIN_X, y, PAGE_W - MARGIN_X, y), fill=RULE, width=3)
    y += 70
    bullets = [
        "Comparaison des performances entre modeles",
        "Courbes d'apprentissage et matrices de confusion",
        "Courbes ROC / Precision-Recall",
        "Grad-CAM, SHAP et verification apres masquage pulmonaire",
    ]
    for item in bullets:
        draw.ellipse((MARGIN_X, y + 12, MARGIN_X + 18, y + 30), fill=ACCENT)
        y = draw_wrapped(draw, item, (MARGIN_X + 42, y), F_BODY, max_width=80)
        y += 18
    return page


def section_pages(section: dict, appendix_no_start: int) -> list[Image.Image]:
    pages: list[Image.Image] = []
    title = section["title"]
    items = [(FIG_DIR / fname, caption) for fname, caption in section["items"] if (FIG_DIR / fname).exists()]
    if not items:
        return pages

    page, draw = new_page()
    y = add_header(draw, title, appendix_no_start)
    slots = 2 if len(items) == 2 else 1

    if len(items) == 1:
        fig, caption = items[0]
        paste_fit(page, fig, (MARGIN_X, y, PAGE_W - MARGIN_X, PAGE_H - MARGIN_BOTTOM - 235))
        draw_wrapped(draw, caption, (MARGIN_X, PAGE_H - MARGIN_BOTTOM - 175), F_CAP, fill=TEXT, max_width=94)
        pages.append(page)
        return pages

    if len(items) == 2:
        slot_h = (PAGE_H - y - MARGIN_BOTTOM - 210) // 2
        for i, (fig, caption) in enumerate(items):
            top = y + i * (slot_h + 195)
            paste_fit(page, fig, (MARGIN_X, top, PAGE_W - MARGIN_X, top + slot_h))
            draw_wrapped(draw, caption, (MARGIN_X, top + slot_h + 28), F_CAP, fill=TEXT, max_width=94)
        pages.append(page)
        return pages

    # Three figures: one large figure on the first page, two compact figures on the next page.
    first_fig, first_caption = items[0]
    paste_fit(page, first_fig, (MARGIN_X, y, PAGE_W - MARGIN_X, PAGE_H - MARGIN_BOTTOM - 235))
    draw_wrapped(draw, first_caption, (MARGIN_X, PAGE_H - MARGIN_BOTTOM - 175), F_CAP, fill=TEXT, max_width=94)
    pages.append(page)

    page, draw = new_page()
    y = add_header(draw, f"{title} - suite", appendix_no_start + 1)
    slot_h = (PAGE_H - y - MARGIN_BOTTOM - 210) // 2
    for i, (fig, caption) in enumerate(items[1:]):
        top = y + i * (slot_h + 195)
        paste_fit(page, fig, (MARGIN_X, top, PAGE_W - MARGIN_X, top + slot_h))
        draw_wrapped(draw, caption, (MARGIN_X, top + slot_h + 28), F_CAP, fill=TEXT, max_width=94)
    pages.append(page)
    return pages


def build_appendix() -> None:
    pages = [cover_page()]
    appendix_no = 1
    for section in FIGURES:
        generated = section_pages(section, appendix_no)
        pages.extend(generated)
        appendix_no += len(generated)
    TMP_APPENDIX.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(TMP_APPENDIX, save_all=True, append_images=pages[1:], resolution=300.0)


def merge() -> None:
    writer = PdfWriter()
    for pdf in (SOURCE_PDF, TMP_APPENDIX):
        reader = PdfReader(str(pdf))
        for page in reader.pages:
            writer.add_page(page)
    with OUT_PDF.open("wb") as f:
        writer.write(f)


def main() -> None:
    if not SOURCE_PDF.exists():
        raise FileNotFoundError(SOURCE_PDF)
    build_appendix()
    merge()
    print(OUT_PDF)


if __name__ == "__main__":
    main()
