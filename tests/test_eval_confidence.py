from scripts.eval_confidence import write_report


def test_write_report_refuses_existing_file(tmp_path, capsys):
    path = tmp_path / "report.txt"
    path.write_text("saved result\n", encoding="utf-8")

    assert write_report(path, "replacement\n") == 1

    assert path.read_text(encoding="utf-8") == "saved result\n"
    assert capsys.readouterr().out == f"Refusing to overwrite {path}; use --out\n"


def test_write_report_creates_new_file(tmp_path):
    path = tmp_path / "report.txt"
    assert write_report(path, "new result\n") == 0
    assert path.read_text(encoding="utf-8") == "new result\n"
