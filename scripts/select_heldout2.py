"""Select the second held-out set as pre-registered in docs/heldout2_preregistration.md.

Copy of scripts/select_heldout.py. Only changes: SEED, output paths, and the
contracts of the first held-out set (data/heldout) are excluded as well.
"""

import csv
import random
import re
import shutil
import unicodedata
from pathlib import Path

from contractguard.ingestion.pdf import PDFIngestionError
from contractguard.pipeline import load_document

CUAD = Path.home() / "Downloads" / "CUAD_v1"
MASTER = CUAD / "master_clauses.csv"
DESIGN = Path("data/contracts")
HELDOUT = Path("data/heldout")
OUT_DIR = Path("data/heldout2")
OUT_CSV = Path("evals/heldout2_selection.csv")
SEED = 20261114
N_MAIN, N_NONE = 10, 2
MAX_PAGES = 200

# CUAD's CSV spells these contracts differently from its PDF folder (checked by hand).
CSV_TO_PDF_NAME = {
    "HarpoonTherapeuticsInc_20200312_10-K_EX-10.18_12051356_EX-10.18_Development Agreement.PDF":  # noqa: E501
        "HarpoonTherapeuticsInc_20200312_10-K_EX-10.18_12051356_EX-10.18_Development Agreement_Option Agreement.pdf",  # noqa: E501
    "KALLOINC_11_03_2011-EX-10.1-STRATEGIC ALLIANCE AGREEMENT.PDF'":
        "KALLOINC_11_03_2011-EX-10.1-STRATEGIC ALLIANCE AGREEMENT.PDF",
}


def key(name: str) -> str:
    """Compare names by plain letters and digits only.

    NFKD turns accented letters stored in different Unicode forms into the same
    plain letters (e.g. both forms of "É" become "E").
    """
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    stem = plain[:-4] if plain.lower().endswith(".pdf") else plain
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


def pdf_keys(folder: Path) -> set[str]:
    return {key(p.name) for p in folder.iterdir() if p.suffix.lower() == ".pdf"}


def main() -> None:
    assert not OUT_CSV.exists(), "Selection already done; it must run only once."
    assert not OUT_DIR.exists(), f"{OUT_DIR} already exists; it must run only once."

    master_rows = list(csv.DictReader(open(MASTER, encoding="utf-8")))
    rows = {key(CSV_TO_PDF_NAME.get(r["Filename"], r["Filename"])): r
            for r in master_rows}
    assert len(rows) == len(master_rows) == 510, "file-name keys collide in CSV"

    pdf_files = [p for p in CUAD.glob("full_contract_pdf/**/*")
                 if p.suffix.lower() == ".pdf"]
    pdfs = {key(p.name): p for p in pdf_files}
    assert len(pdfs) == len(pdf_files) == 510, "file-name keys collide in PDFs"
    assert set(pdfs) == set(rows), "CSV and PDF folder names do not match"

    design = pdf_keys(DESIGN)
    heldout = pdf_keys(HELDOUT)
    assert len(design) == 30, f"expected 30 design contracts, found {len(design)}"
    assert len(heldout) == 12, f"expected 12 held-out contracts, found {len(heldout)}"
    assert not design & heldout, "design and held-out sets overlap"
    seen = design | heldout
    not_found = seen - rows.keys()
    assert not not_found, f"used contracts not found in CUAD: {not_found}"

    main_pool = sorted(k for k, r in rows.items() if k not in seen
                       and not empty(r["Renewal Term"])
                       and not empty(r["Cap On Liability"]))
    none_pool = sorted(k for k, r in rows.items() if k not in seen
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
