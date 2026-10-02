from pathlib import Path


def test_rehearsal_script_uses_lf_line_endings() -> None:
    script = Path(__file__).resolve().parents[1] / "release" / "rehearse.sh"

    assert b"\r\n" not in script.read_bytes()


def test_shell_scripts_are_pinned_to_lf_in_git_attributes() -> None:
    attributes = (Path(__file__).resolve().parents[1] / ".gitattributes").read_text(encoding="utf-8")

    assert "*.sh text eol=lf" in attributes
