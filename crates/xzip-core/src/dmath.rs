//! Deterministic transcendental functions.
//!
//! `f64::exp` / `f64::ln` call the platform libm, whose last-bit results differ
//! between glibc and MSVCRT. Anything that feeds the coder must be bit-identical
//! on every machine, so we implement them from IEEE-754 basic operations only
//! (+, -, *, /, round), which are exactly rounded everywhere. Rust never
//! contracts a*b+c into FMA implicitly, so these are reproducible.

const LN2: f64 = core::f64::consts::LN_2;

/// 1/i! for i = 0..=16 (exactly the f64 nearest each value).
const INV_FACT: [f64; 17] = [
    1.0,
    1.0,
    0.5,
    0.16666666666666666,
    0.041666666666666664,
    0.008333333333333333,
    0.001388888888888889,
    0.0001984126984126984,
    2.48015873015873e-05,
    2.7557319223985893e-06,
    2.755731922398589e-07,
    2.505210838544172e-08,
    2.08767569878681e-09,
    1.6059043836821613e-10,
    1.1470745597729725e-11,
    7.647163731819816e-13,
    4.779477332387385e-14,
];

/// 1/k for odd k = 3..=25.
const INV_ODD: [f64; 12] = [
    1.0 / 3.0,
    1.0 / 5.0,
    1.0 / 7.0,
    1.0 / 9.0,
    1.0 / 11.0,
    1.0 / 13.0,
    1.0 / 15.0,
    1.0 / 17.0,
    1.0 / 19.0,
    1.0 / 21.0,
    1.0 / 23.0,
    1.0 / 25.0,
];

/// e^x, deterministic. Accurate to ~1 ulp on [-700, 700].
pub fn exp(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x > 700.0 {
        return f64::INFINITY;
    }
    if x < -700.0 {
        return 0.0;
    }
    let k = (x / LN2).round();
    let r = x - k * LN2; // |r| <= ln2/2
    // Taylor series to r^16 (error < 0.347^17/17! ~ 1e-22), Horner form with
    // constant 1/i! coefficients: no divisions, fixed operation order.
    let mut sum = INV_FACT[16];
    let mut i = 15;
    loop {
        sum = sum * r + INV_FACT[i];
        if i == 0 {
            break;
        }
        i -= 1;
    }
    let ki = k as i64;
    sum * f64::from_bits(((ki + 1023) as u64) << 52)
}

/// Natural log for finite x > 0 (normal range), deterministic.
pub fn ln(x: f64) -> f64 {
    debug_assert!(x > 0.0 && x.is_finite());
    let bits = x.to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000); // [1,2)
    if m > core::f64::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    // ln(m) = 2 atanh(s), s = (m-1)/(m+1), |s| <= 0.1716
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let mut term = s;
    let mut sum = s;
    for &inv in INV_ODD.iter() {
        term *= s2;
        sum += term * inv;
    }
    (e as f64) * LN2 + 2.0 * sum
}

/// Logistic function 1/(1+e^-x).
#[inline]
pub fn squash(x: f64) -> f64 {
    1.0 / (1.0 + exp(-x))
}

/// Inverse logistic ln(p/(1-p)).
#[inline]
pub fn stretch(p: f64) -> f64 {
    ln(p / (1.0 - p))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inverse_factorials_exact() {
        let mut f = 1.0f64;
        for (i, &c) in INV_FACT.iter().enumerate() {
            if i > 0 {
                f *= i as f64;
            }
            assert_eq!(c, 1.0 / f, "1/{i}!");
        }
    }
    #[test]
    fn matches_libm_closely() {
        let mut x = -20.0;
        while x < 20.0 {
            let a = exp(x);
            let b = x.exp();
            assert!(((a - b) / b).abs() < 1e-14, "exp({x}) {a} vs {b}");
            x += 0.0137;
        }
        let mut p = 1e-9;
        while p < 1e9 {
            let a = ln(p);
            let b = p.ln();
            assert!((a - b).abs() < 1e-13, "ln({p}) {a} vs {b}");
            p *= 1.37;
        }
    }
}
