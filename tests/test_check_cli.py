from contractguard.check import build_parser


def test_followup_version_defaults_to_v2():
    args = build_parser().parse_args(["demo.pdf", "--policy", "policy.json"])
    assert args.followup_version == "v2"
