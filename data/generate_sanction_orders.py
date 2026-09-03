"""
Generates a sample of synthetic MPLADS "Sanction Order" document images —
real rendered text (project ID, sanctioned amount, district authority,
date), not placeholder graphics — so the OCR pipeline in
sentinel/engines/document_ocr.py has genuine document images to extract
text from. A subset of documents are deliberately generated with a
MISMATCHED amount (the printed figure differs from what's in the
projects table) to validate that OCR-based cross-checking actually
catches a doctored/incorrect sanction order — a real MPLADS fraud
pattern.

Run: python data/generate_sanction_orders.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sentinel.config import DATA_DIR

np.random.seed(11)

DOC_DIR = DATA_DIR.parent / "sample_documents"
N_DOCUMENTS = 60
MISMATCH_RATE = 0.25  # fraction with a deliberately altered printed amount

DOC_SIZE = (900, 500)


def _load_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_sanction_order(project_id, printed_amount, district, implementing_agency, sanction_date):
    img = Image.new("RGB", DOC_SIZE, (250, 248, 240))
    draw = ImageDraw.Draw(img)

    title_font = _load_font(28)
    label_font = _load_font(20)
    body_font = _load_font(22)

    draw.rectangle([20, 20, DOC_SIZE[0] - 20, DOC_SIZE[1] - 20], outline=(0, 0, 0), width=2)
    draw.text((DOC_SIZE[0] // 2 - 160, 45), "SANCTION ORDER", font=title_font, fill=(0, 0, 0))
    draw.text((DOC_SIZE[0] // 2 - 210, 85), "Members of Parliament Local Area Development Scheme",
              font=label_font, fill=(40, 40, 40))
    draw.line([(40, 120), (DOC_SIZE[0] - 40, 120)], fill=(0, 0, 0), width=1)

    lines = [
        f"Project ID: {project_id}",
        f"District: {district}",
        f"Implementing Agency: {implementing_agency}",
        f"Date of Sanction: {sanction_date}",
        "",
        f"Sanctioned Amount: Rs. {printed_amount:,.0f}/-",
        "",
        "This work is hereby sanctioned under the MPLADS Scheme subject to",
        "completion within the guideline timeline and submission of the",
        "requisite Utilisation Certificate upon completion.",
    ]
    y = 160
    for line in lines:
        font = body_font if "Sanctioned Amount" in line else label_font
        draw.text((50, y), line, font=font, fill=(10, 10, 10))
        y += 32

    return img


def main():
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    projects = pd.read_csv(DATA_DIR / "projects.csv")
    sample = projects.sample(min(N_DOCUMENTS, len(projects)), random_state=5).reset_index(drop=True)

    n_mismatch = int(len(sample) * MISMATCH_RATE)
    mismatch_idx = set(np.random.choice(sample.index, n_mismatch, replace=False))

    rows = []
    for i, row in sample.iterrows():
        true_amount = row["sanctioned_amount"]
        is_mismatched = i in mismatch_idx
        if is_mismatched:
            # A doctored sanction order — printed figure inflated relative
            # to what's actually recorded in the system, the real-world
            # fraud pattern this check exists to catch.
            printed_amount = true_amount * np.random.uniform(1.15, 1.6)
        else:
            printed_amount = true_amount

        img = render_sanction_order(
            row["project_id"], printed_amount, row["district"],
            row.get("implementing_agency", "District Implementing Agency"),
            str(row["sanction_date"])[:10],
        )
        fname = f"SANCTION_{row['project_id']}.png"
        img.save(DOC_DIR / fname)

        rows.append({
            "document_id": f"DOC_{row['project_id']}",
            "project_id": row["project_id"],
            "filename": fname,
            "printed_amount": round(printed_amount, 2),
            "true_sanctioned_amount": round(true_amount, 2),
            "is_seeded_mismatch": is_mismatched,
        })

    doc_df = pd.DataFrame(rows)
    doc_df.to_csv(DATA_DIR / "sanction_order_documents.csv", index=False)
    print(f"Rendered {len(doc_df)} sanction order documents into {DOC_DIR}")
    print(f"  {doc_df['is_seeded_mismatch'].sum()} deliberately seeded with a mismatched printed amount")


if __name__ == "__main__":
    main()
