/*
 * ramanujan_pi.cpp
 * =================
 *
 * Computes pi to a user-specified number of decimal places using
 * Ramanujan's 1914 series for 1/pi:
 *
 *                     2*sqrt(2)     inf   (4k)! * (1103 + 26390k)
 *         1 / pi  =  -----------  * SUM  --------------------------
 *                        9801       k=0      (k!)^4 * 396^(4k)
 *
 * Why not plain `double`?
 * ------------------------
 * `double` is IEEE-754 binary64: ~15-17 significant decimal digits,
 * a hard limit baked into the hardware format -- no algorithm can
 * push a double past that. To go beyond it we use GMP's `mpf_t`
 * arbitrary-precision floating-point type, whose precision (in bits)
 * we set explicitly based on how many digits the user asks for.
 *
 * Performance / accuracy design
 * -------------------------------
 * 1. Each successive term of Ramanujan's series contributes roughly
 *    8 additional correct decimal digits (256/396^4 ~ 1.04e-8 shrink
 *    per term), so terms_needed ~= digits/8, plus a safety margin.
 *
 * 2. Instead of recomputing (4k)!, (k!)^4 and 396^(4k) from scratch
 *    every iteration, the ratio a_k = (4k)!/(k!)^4/396^(4k) is
 *    updated incrementally:
 *
 *        a_(k+1) = a_k * (4k+1)(4k+2)(4k+3)(4k+4) / ((k+1)^4 * 396^4)
 *
 *    keeping the per-term cost roughly constant instead of growing,
 *    so total cost stays close to O(terms) rather than O(terms^2).
 *
 * 3. A block of "guard bits" beyond the requested precision is
 *    carried through the whole computation and only the requested
 *    number of digits is printed at the end, absorbing accumulated
 *    round-off from repeated division.
 *
 * Dependencies
 * ------------
 * Requires the GNU Multiple Precision Arithmetic Library (GMP) and
 * its C++ bindings (gmpxx).
 *
 *   Ubuntu/Debian : sudo apt-get install libgmp-dev
 *   Fedora        : sudo dnf install gmp-devel
 *   macOS (brew)  : brew install gmp
 *   Windows       : use MSYS2/vcpkg to install gmp, or WSL + apt above
 *
 * Compile
 * -------
 *   g++ -O2 -std=c++17 ramanujan_pi.cpp -o ramanujan_pi -lgmpxx -lgmp
 *
 * Run
 * ---
 *   ./ramanujan_pi
 */

#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <gmpxx.h>

struct PiResult {
    mpf_class value;
    long terms_used;
};

// Computes pi to `decimal_places` digits after the decimal point.
// Throws std::invalid_argument on bad input.
PiResult compute_pi(long decimal_places, long guard_digits = 50) {
    if (decimal_places <= 0) {
        throw std::invalid_argument("decimal_places must be a positive integer");
    }

    const long total_digits = decimal_places + guard_digits;

    // bits ~= digits * log2(10) (~3.32193), plus a little headroom.
    const mp_bitcnt_t bits =
        static_cast<mp_bitcnt_t>(total_digits * 3.3219280948873623) + 64;
    mpf_set_default_prec(bits);

    // Each term yields ~8 correct decimal digits; add a safety margin.
    const long terms = total_digits / 8 + 10;

    mpz_class c396_4;
    mpz_ui_pow_ui(c396_4.get_mpz_t(), 396, 4); // 396^4, exact integer

    mpf_class total(0);
    mpf_class ratio(1); // a_0 = (0)! / (0!)^4 / 396^0 = 1

    for (long k = 0; k < terms; ++k) {
        mpz_class term_coeff_z = mpz_class(1103) + mpz_class(26390) * k;
        mpf_class term_coeff(term_coeff_z);
        total += ratio * term_coeff;

        // Advance ratio: a_k -> a_(k+1), all as exact integers first,
        // then a single division at working float precision.
        mpz_class num = mpz_class(4 * k + 1) * mpz_class(4 * k + 2) *
                         mpz_class(4 * k + 3) * mpz_class(4 * k + 4);
        mpz_class kp1 = k + 1;
        mpz_class den = kp1 * kp1 * kp1 * kp1 * c396_4;

        ratio *= mpf_class(num);
        ratio /= mpf_class(den);
    }

    mpf_class sqrt2;
    mpf_sqrt(sqrt2.get_mpf_t(), mpf_class(2).get_mpf_t());

    mpf_class inv_pi = (mpf_class(2) * sqrt2 / mpf_class(9801)) * total;
    mpf_class pi = mpf_class(1) / inv_pi;

    return {pi, terms};
}

// Format an mpf_class as a fixed-point string "3.xxxxxxx" with exactly
// `decimal_places` digits after the point, using GMP's own correctly
// rounded base-10 conversion (mpf_get_str), not stream formatting.
std::string format_pi(const mpf_class& pi, long decimal_places) {
    mp_exp_t exp;
    // n_digits = decimal_places + 1 significant digits (leading "3" + fractional part)
    std::unique_ptr<char, void(*)(void*)> raw(
        mpf_get_str(nullptr, &exp, 10, decimal_places + 1, pi.get_mpf_t()),
        [](void* p) { free(p); }
    );
    std::string digits(raw.get());

    // mpf_get_str can return fewer digits than asked if trailing digits
    // are exactly zero; pad on the right just in case.
    while (static_cast<long>(digits.size()) < decimal_places + 1) {
        digits.push_back('0');
    }

    std::string result;
    if (exp <= 0) {
        // value < 1 (shouldn't happen for pi, but handle defensively)
        result = "0." + std::string(-exp, '0') + digits;
    } else {
        result = digits.substr(0, exp) + "." + digits.substr(exp);
    }
    return result;
}

int main() {
    std::cout << "============================================================\n";
    std::cout << " Ramanujan Series pi Calculator (arbitrary precision, GMP)\n";
    std::cout << "============================================================\n";
    std::cout << "Enter number of decimal places of pi to compute: ";

    long decimal_places;
    if (!(std::cin >> decimal_places)) {
        std::cerr << "Error: please enter a positive integer.\n";
        return 1;
    }
    if (decimal_places <= 0) {
        std::cerr << "Error: please enter a positive integer.\n";
        return 1;
    }

    if (decimal_places > 200000) {
        std::cout << "Note: " << decimal_places
                  << " digits is a lot of work -- this may take a while.\n";
    }

    try {
        auto start = std::chrono::high_resolution_clock::now();
        PiResult result = compute_pi(decimal_places);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        std::string pi_str = format_pi(result.value, decimal_places);

        std::cout << "\nTerms used         : " << result.terms_used << "\n";
        std::cout << "Time taken         : " << std::fixed
                  << std::setprecision(4) << elapsed << " seconds\n";
        std::cout << "\npi = " << pi_str << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error while computing pi: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
