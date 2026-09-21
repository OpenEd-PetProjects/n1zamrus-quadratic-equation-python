"""Exact quadratic-equation solver for arbitrarily large integer coefficients."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from math import isqrt
from typing import Literal, Sequence

_CHUNK_BASE = 1_000_000_000
_DIGITS_PER_CHUNK = 9
MAX_COEFFICIENT_DIGITS = 10_000
_SMALL_PRIMES = (
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
    101,
    103,
    107,
    109,
    113,
    127,
    131,
    137,
    139,
    149,
    151,
    157,
    163,
    167,
    173,
    179,
    181,
    191,
    193,
    197,
    199,
)
Status = Literal[
    "two_real",
    "double_real",
    "two_complex",
    "one_real",
    "no_solution",
    "all_reals",
]


def parse_integer(value: str) -> int:
    """Parse a signed decimal integer without Python's digit-count limit."""

    if not isinstance(value, str):
        raise TypeError("coefficients must be decimal strings or integers")
    token = value.strip()
    if not token:
        raise ValueError("coefficient cannot be empty")

    sign = 1
    if token[0] in "+-":
        sign = -1 if token[0] == "-" else 1
        token = token[1:]
    if not token or any(character < "0" or character > "9" for character in token):
        raise ValueError("coefficient must be a decimal integer")
    if len(token) > MAX_COEFFICIENT_DIGITS:
        raise ValueError(
            f"coefficient cannot contain more than {MAX_COEFFICIENT_DIGITS:,} digits"
        )

    first_chunk_length = len(token) % _DIGITS_PER_CHUNK or _DIGITS_PER_CHUNK
    result = int(token[:first_chunk_length])
    for start in range(first_chunk_length, len(token), _DIGITS_PER_CHUNK):
        chunk = token[start : start + _DIGITS_PER_CHUNK]
        result = result * _CHUNK_BASE + int(chunk)
    return sign * result


def integer_to_decimal(value: int) -> str:
    """Format a potentially huge integer without Python's digit-count limit."""

    if not isinstance(value, int):
        raise TypeError("value must be an integer")
    if value == 0:
        return "0"

    sign = "-" if value < 0 else ""
    remainder = abs(value)
    chunks: list[str] = []
    while remainder:
        remainder, chunk = divmod(remainder, _CHUNK_BASE)
        chunks.append(f"{chunk:0{_DIGITS_PER_CHUNK}d}")
    return sign + chunks[-1].lstrip("0") + "".join(reversed(chunks[:-1]))


def _as_integer(value: str | int) -> int:
    if isinstance(value, bool):
        raise TypeError("coefficients must be decimal strings or integers")
    if isinstance(value, int):
        return value
    return parse_integer(value)


def _fraction_text(value: Fraction) -> str:
    numerator = integer_to_decimal(value.numerator)
    denominator = integer_to_decimal(value.denominator)
    return numerator if denominator == "1" else f"{numerator}/{denominator}"


def _normalize_radical(coefficient: Fraction, radicand: int) -> tuple[Fraction, int]:
    """Remove readily detectable square factors while preserving exactness."""

    factor = 1
    remaining = radicand
    for prime in _SMALL_PRIMES:
        square = prime * prime
        while remaining % square == 0:
            remaining //= square
            factor *= prime
    return coefficient * factor, remaining


@dataclass(frozen=True)
class ExactRoot:
    """An exact root represented as rational + optional radical term.

    If ``imaginary`` is true, the radical term is multiplied by ``i``. A
    radicand of 1 represents a rational imaginary coefficient.
    """

    rational: Fraction
    radical_coefficient: Fraction = Fraction(0)
    radicand: int | None = None
    imaginary: bool = False

    def expression(self) -> str:
        if self.radical_coefficient == 0 or self.radicand is None:
            return _fraction_text(self.rational)

        coefficient = abs(self.radical_coefficient)
        if self.radicand == 1:
            radical = "i" if coefficient == 1 else f"{_fraction_text(coefficient)}i"
        else:
            coefficient_text = "" if coefficient == 1 else f"{_fraction_text(coefficient)}*"
            radical = f"{coefficient_text}sqrt({integer_to_decimal(self.radicand)})"
            if self.imaginary:
                radical += "i"

        if self.rational == 0:
            prefix = "-" if self.radical_coefficient < 0 else ""
            return prefix + radical

        sign = "+" if self.radical_coefficient > 0 else "-"
        return f"{_fraction_text(self.rational)} {sign} {radical}"

    def __str__(self) -> str:
        return self.expression()


@dataclass(frozen=True)
class Solution:
    """Exact result and classification for an equation."""

    status: Status
    roots: tuple[ExactRoot, ...] = ()

    @property
    def is_identity(self) -> bool:
        return self.status == "all_reals"

    @property
    def is_contradiction(self) -> bool:
        return self.status == "no_solution"


def discriminant(a: str | int, b: str | int, c: str | int) -> int:
    """Return b² - 4ac using integer arithmetic only."""

    a_value, b_value, c_value = map(_as_integer, (a, b, c))
    return b_value * b_value - 4 * a_value * c_value


def _quadratic_roots(a: int, b: int, value: int) -> Solution:
    denominator = 2 * abs(a)
    rational = Fraction(-b, denominator)
    if value == 0:
        return Solution("double_real", (ExactRoot(rational),))

    if value > 0:
        root = isqrt(value)
        if root * root == value:
            return Solution(
                "two_real",
                (
                    ExactRoot(Fraction(-b + root, denominator)),
                    ExactRoot(Fraction(-b - root, denominator)),
                ),
            )
        coefficient, radicand = _normalize_radical(Fraction(1, denominator), value)
        return Solution(
            "two_real",
            (
                ExactRoot(rational, coefficient, radicand),
                ExactRoot(rational, -coefficient, radicand),
            ),
        )

    radicand = -value
    root = isqrt(radicand)
    if root * root == radicand:
        radical_coefficient = Fraction(root, denominator)
        radicand = 1
    else:
        radical_coefficient, radicand = _normalize_radical(Fraction(1, denominator), radicand)
    return Solution(
        "two_complex",
        (
            ExactRoot(rational, radical_coefficient, radicand, imaginary=True),
            ExactRoot(rational, -radical_coefficient, radicand, imaginary=True),
        ),
    )


def solve(a: str | int, b: str | int, c: str | int) -> Solution:
    """Solve ``a*x² + b*x + c = 0`` exactly for integer coefficients."""

    a_value, b_value, c_value = map(_as_integer, (a, b, c))
    if a_value == 0:
        if b_value == 0:
            return Solution("all_reals" if c_value == 0 else "no_solution")
        return Solution("one_real", (ExactRoot(Fraction(-c_value, b_value)),))

    return _quadratic_roots(a_value, b_value, b_value * b_value - 4 * a_value * c_value)


def _solution_payload(result: Solution) -> dict[str, object]:
    return {
        "status": result.status,
        "roots": [root.expression() for root in result.roots],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process exit code."""

    parser = argparse.ArgumentParser(description="Solve an integer quadratic equation exactly")
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("c")
    parser.add_argument(
        "--json",
        action="store_true",
        help="print a machine-readable JSON object instead of human-readable lines",
    )
    arguments = parser.parse_args(argv)

    try:
        result = solve(arguments.a, arguments.b, arguments.c)
    except (TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if arguments.json:
        print(json.dumps(_solution_payload(result), ensure_ascii=False))
    else:
        print(f"status: {result.status}")
        if result.roots:
            print("roots:")
            for root in result.roots:
                print(f"- {root.expression()}")
        else:
            print("roots: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
