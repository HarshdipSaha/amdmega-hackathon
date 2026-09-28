from silentpath.cli import build_parser


def test_run_command_parses_matrix_output_and_budget():
    args = build_parser().parse_args([
        "run", "--matrix", "m.yaml", "--out", "runs/x", "--cap-hours", "1.5"
    ])
    assert args.command == "run"
    assert args.producer == "sdpa"
    assert args.cap_hours == 1.5


def test_fake_is_explicit_shortcut():
    args = build_parser().parse_args([
        "run", "--matrix", "m.yaml", "--out", "runs/x", "--cap-hours", "1", "--fake"
    ])
    assert args.fake is True


def test_producer_option_accepts_fake_or_sdpa():
    args = build_parser().parse_args([
        "run", "--matrix", "m.yaml", "--out", "runs/x", "--cap-hours", "1",
        "--producer", "fake"
    ])
    assert args.producer == "fake"


def test_report_command_parses():
    assert build_parser().parse_args(["report", "--out", "runs/x"]).command == "report"
