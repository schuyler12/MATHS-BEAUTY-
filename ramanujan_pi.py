"""
ramanujan_pi.py
================

Computes pi to a user-specified number of decimal places using
Ramanujan's 1914 series for 1/pi:

                    2*sqrt(2)     inf   (4k)! * (1103 + 26390k)
        1 / pi  =  -----------  * SUM  --------------------------
                       9801       k=0      (k!)^4 * 396^(4k)
                       
        a_(k+1) = a_k * (4k+1)(4k+2)(4k+3)(4k+4) / ((k+1)^4 * 396^4)
"""

from decimal import Decimal, getcontext, localcontext
import time
import sys


def compute_pi(decimal_places: int, guard_digits: int = 50) -> Decimal:
    """
    Compute pi to `decimal_places` digits after the decimal point
    using Ramanujan's series.

    Parameters
    ----------
    decimal_places : int
        Number of correct digits required after "3.".
    guard_digits : int
        Extra internal precision carried to protect against
        accumulated rounding error. 50 is generous for any
        practical digit count.

    Returns
    -------
    Decimal
        pi, rounded to `decimal_places` digits after the point.
    """
    if decimal_places <= 0:
        raise ValueError("decimal_places must be a positive integer")

    working_precision = decimal_places + guard_digits

    with localcontext() as ctx:
        ctx.prec = working_precision

        # Each term supplies ~8 correct digits -> terms needed, plus margin.
        terms = working_precision // 8 + 10

        C396_4 = Decimal(396) ** 4      # 396^4, computed once
        total = Decimal(0)
        ratio = Decimal(1)              # a_0 = (0)!/(0!)^4/396^0 = 1

        for k in range(terms):
            term_coeff = Decimal(1103 + 26390 * k)
            total += ratio * term_coeff

            # Advance ratio: a_k -> a_(k+1)
            num = Decimal((4 * k + 1) * (4 * k + 2) * (4 * k + 3) * (4 * k + 4))
            den = Decimal(k + 1) ** 4 * C396_4
            ratio = (ratio * num) / den

        sqrt2 = Decimal(2).sqrt()
        inv_pi = (2 * sqrt2 / Decimal(9801)) * total
        pi = 1 / inv_pi

    # Round the (still guard-padded) result down to exactly what was asked for.
    with localcontext() as ctx:
        ctx.prec = decimal_places + 1  # +1 for the leading "3"
        pi = +pi  # unary plus re-applies rounding under the new context

    return pi, terms


def format_pi(pi: Decimal, decimal_places: int) -> str:
    """Format pi as '3.xxxxxxx' with exactly `decimal_places` digits."""
    s = f"{pi:.{decimal_places}f}"
    return s


def main():
    print("=" * 60)
    print(" Ramanujan Series pi Calculator (arbitrary precision)")
    print("=" * 60)

    try:
        raw = input("Enter number of decimal places of pi to compute: ").strip()
        decimal_places = int(raw)
        if decimal_places <= 0:
            raise ValueError
    except ValueError:
        print("Error: please enter a positive integer.", file=sys.stderr)
        sys.exit(1)

    if decimal_places > 200_000:
        print(f"Note: {decimal_places} digits is a lot of work for a pure-Python "
              f"Decimal implementation -- this may take a while.")

    start = time.perf_counter()
    try:
        pi_value, terms_used = compute_pi(decimal_places)
    except Exception as exc:  # keep this narrow in spirit: report, don't hide
        print(f"Error while computing pi: {exc}", file=sys.stderr)
        sys.exit(1)
    elapsed = time.perf_counter() - start

    result = format_pi(pi_value, decimal_places)

    print(f"\nTerms used        : {terms_used}")
    print(f"Time taken         : {elapsed:.4f} seconds")
    print(f"\npi = {result}")


if __name__ == "__main__":
    main()
