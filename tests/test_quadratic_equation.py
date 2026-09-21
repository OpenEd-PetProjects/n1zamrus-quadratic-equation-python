import json
from pathlib import Path
import subprocess
import sys

import pytest

from quadratic_equation import (
    discriminant,
    integer_to_decimal,
    main,
    parse_integer,
    solve,
)

CLI_PATH = Path(__file__).parents[1].joinpath("quadratic_equation.py")


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI_PATH), *arguments],
        capture_output=True,
        check=False,
        text=True,
    )


def test_two_rational_roots_are_exact() -> None:
    result = solve("1", "-3", "2")

    assert result.status == "two_real"
    assert [root.expression() for root in result.roots] == ["2", "1"]


def test_negative_leading_coefficient_is_supported() -> None:
    result = solve(-1, 3, -2)

    assert result.status == "two_real"
    assert len(result.roots) == 2


def test_double_root() -> None:
    result = solve(1, -2, 1)

    assert result.status == "double_real"
    assert result.roots[0].expression() == "1"


def test_irrational_roots_keep_radical_form() -> None:
    result = solve(1, 0, -2)

    assert result.status == "two_real"
    assert [root.expression() for root in result.roots] == ["sqrt(2)", "-sqrt(2)"]


def test_complex_roots_are_exact() -> None:
    result = solve(1, 2, 5)

    assert result.status == "two_complex"
    assert [root.expression() for root in result.roots] == ["-1 + 2i", "-1 - 2i"]


def test_pure_imaginary_roots_are_compact_and_exact() -> None:
    result = solve(1, 0, 1)

    assert result.status == "two_complex"
    assert [root.expression() for root in result.roots] == ["i", "-i"]


def test_linear_equation() -> None:
    result = solve(0, 2, 4)

    assert result.status == "one_real"
    assert result.roots[0].expression() == "-2"


@pytest.mark.parametrize(
    ("coefficients", "status"),
    [((0, 0, 1), "no_solution"), ((0, 0, 0), "all_reals")],
)
def test_constant_equations(coefficients: tuple[int, int, int], status: str) -> None:
    assert solve(*coefficients).status == status


def test_ten_thousand_digit_coefficients_round_trip_without_float() -> None:
    text = "1" + "0" * 9_999
    value = parse_integer(text)

    assert integer_to_decimal(value) == text
    result = solve("1", "0", "-" + text)
    assert result.status == "two_real"
    root = result.roots[0]
    assert root.radicand == 10
    assert root.radical_coefficient**2 * root.radicand == value
    assert len(integer_to_decimal(root.radical_coefficient.numerator)) == 5_000


def test_large_perfect_square_has_exact_integer_roots() -> None:
    base = parse_integer("9" * 4_999)
    coefficient = integer_to_decimal(-(base * base))

    result = solve("1", "0", coefficient)

    assert result.status == "two_real"
    assert result.roots[0].radical_coefficient == 0
    assert result.roots[0].rational == base
    assert result.roots[1].rational == -base


def test_invalid_coefficients_are_rejected() -> None:
    for value in ("", "+", "-", "1.0", "1_000"):
        with pytest.raises(ValueError):
            parse_integer(value)

    with pytest.raises(TypeError):
        parse_integer(1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="10,000"):
        parse_integer("1" + "0" * 10_000)


def test_signed_and_padded_decimal_strings_are_supported() -> None:
    assert parse_integer(" +00042 ") == 42
    assert parse_integer("-00042") == -42


def test_discriminant_uses_integer_arithmetic() -> None:
    source = Path(__file__).parents[1].joinpath("quadratic_equation.py").read_text()

    assert discriminant("1", "-3", "2") == 1
    assert "float(" not in source


def test_cli_human_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["1", "-3", "2"]) == 0

    assert capsys.readouterr().out.splitlines() == [
        "status: two_real",
        "roots:",
        "- 2",
        "- 1",
    ]


def test_cli_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["1", "2", "5", "--json"]) == 0

    assert capsys.readouterr().out.strip() == '{"status": "two_complex", "roots": ["-1 + 2i", "-1 - 2i"]}'


def test_cli_json_keeps_ten_thousand_digit_values_as_strings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    coefficient = "1" + "0" * 9_999

    assert main(["1", "0", "-" + coefficient, "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "two_real"
    assert all(isinstance(root, str) for root in payload["roots"])


def test_cli_reports_invalid_input(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["1.5", "2", "3"]) == 2

    assert capsys.readouterr().err == "error: coefficient must be a decimal integer\n"


@pytest.mark.parametrize(
    ("arguments", "status"),
    [
        (("1", "-3", "2"), "two_real"),
        (("1", "-2", "1"), "double_real"),
        (("1", "2", "5"), "two_complex"),
        (("0", "2", "4"), "one_real"),
        (("0", "0", "1"), "no_solution"),
        (("0", "0", "0"), "all_reals"),
    ],
)
def test_cli_subprocess_reports_every_status(
    arguments: tuple[str, str, str], status: str
) -> None:
    completed = run_cli(*arguments, "--json")

    assert completed.returncode == 0
    assert json.loads(completed.stdout)["status"] == status


def test_cli_subprocess_reports_invalid_input_and_missing_arguments() -> None:
    invalid = run_cli("1.5", "2", "3")
    missing = run_cli("1", "2")

    assert invalid.returncode == 2
    assert invalid.stdout == ""
    assert invalid.stderr == "error: coefficient must be a decimal integer\n"
    assert missing.returncode == 2
    assert "usage:" in missing.stderr


def test_cli_subprocess_preserves_irrational_root_format() -> None:
    completed = run_cli("1", "0", "-2")

    assert completed.returncode == 0
    assert "- sqrt(2)" in completed.stdout
