"""Select held-out contracts exactly as pre-registered in docs/heldout_preregistration.md."""

import csv
import random
import re
import shutil
from pathlib import Path

from contractguard.ingestion.pdf import PDFIngestionError
from contractguard.pipeline import load_document

CUAD = Path.home() / "Downloads" / "CUAD_v1"
MASTER = CUAD / "master_clauses.csv"
OURS = Path("data/contracts")
OUT_DIR = Path("data/heldout")
OUT_CSV = Path("evals/heldout_selection.csv")
SEED = 20261103
N_MAIN, N_NONE = 10, 2
MAX_PAGES = 200


def key(name: str) -> str:
    """Compare names by letters and digits only (CUAD spells some names differently)."""
    stem = name[:-4] if name.lower().endswith(".pdf") else name
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def empty(value: str) -> bool:
    return value.strip() in ("", "[]")


def preflight(path: Path) -> str:
    try:
        document = load_document(path)
    except PDFIngestionError:
        return "unreadable"
    if document.page_count == 0:
        return "empty"
    if document.page_count > MAX_PAGES:
        return "too_many_pages"
    if all(page.needs_ocr for page in document.pages):
        return "scanned"
    return "ok"


def pick(order, n, set_name, pdfs, out):
    chosen = 0
    for position, k in enumerate(order):
        if chosen == n:
            break
        src = pdfs.get(k)
        status = "missing_pdf" if src is None else preflight(src)
        out.append({"set": set_name, "order": position,
                    "contract": src.name if src else k, "status": status})
        if status == "ok":
            chosen += 1
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, OUT_DIR / src.name)
    assert chosen == n, f"only {chosen} usable contracts for {set_name}"


def main() -> None:
    assert not OUT_CSV.exists(), "Selection already done; it must run only once."

    master_rows = list(csv.DictReader(open(MASTER, encoding="utf-8")))
    rows = {key(r["Filename"]): r for r in master_rows}
    assert len(rows) == len(master_rows) == 510, "file-name keys collide in CSV"

    pdf_files = [p for p in CUAD.glob("full_contract_pdf/**/*")
                 if p.suffix.lower() == ".pdf"]
    pdfs = {key(p.name): p for p in pdf_files}
    assert len(pdfs) == len(pdf_files) == 510, "file-name keys collide in PDFs"
    assert set(pdfs) == set(rows), "CSV and PDF folder names do not match"

    ours = {key(p.name) for p in OURS.iterdir() if p.suffix.lower() == ".pdf"}
    assert len(ours) == 30, f"expected 30 original contracts, found {len(ours)}"
    not_found = ours - rows.keys()
    assert not not_found, f"originals not found in CUAD: {not_found}"

    main_pool = sorted(k for k, r in rows.items() if k not in ours
                       and not empty(r["Renewal Term"])
                       and not empty(r["Cap On Liability"]))
    none_pool = sorted(k for k, r in rows.items() if k not in ours
                       and empty(r["Renewal Term"])
                       and empty(r["Cap On Liability"])
                       and empty(r["Notice Period To Terminate Renewal"]))
    rng = random.Random(SEED)
    rng.shuffle(main_pool)
    rng.shuffle(none_pool)
    print(f"pool sizes: main={len(main_pool)} no_answer={len(none_pool)}")

    out: list[dict] = []
    pick(main_pool, N_MAIN, "main", pdfs, out)
    pick(none_pool, N_NONE, "no_answer", pdfs, out)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["set", "order", "contract", "status"])
        writer.writeheader()
        writer.writerows(out)
    for row in out:
        print(row["set"], row["order"], row["status"], row["contract"])


if __name__ == "__main__":
    main()