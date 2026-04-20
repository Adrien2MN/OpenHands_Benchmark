"""Tests for shared argument parser defaults."""

from pathlib import Path

from benchmarks.utils.args_parser import get_default_output_dir, get_parser


def test_default_output_dir_points_to_workspace_root() -> None:
    expected = Path(__file__).resolve().parents[3] / "OpenHands_output"
    assert get_default_output_dir() == str(expected)


def test_parser_uses_workspace_output_dir_by_default() -> None:
    parser = get_parser(add_llm_config=False)
    args = parser.parse_args([])

    expected = Path(__file__).resolve().parents[3] / "OpenHands_output"
    assert args.output_dir == str(expected)