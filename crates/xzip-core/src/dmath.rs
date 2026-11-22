//! Deterministic transcendental functions.
//!
//! `f64::exp` / `f64::ln` call the platform libm, whose last-bit results differ
//! between glibc and MSVCRT. Anything that feeds the coder must be bit-identical
//! on every machine, so we implement them from IEEE-754 basic operations only
//! (+, -, *, /, round), which are exactly rounded everywhere. Rust never
//! contracts a*b+c into FMA implicitly, so these are reproducible.

const LN2: f64 = core::f64::consts::LN_2;

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
    // Taylor series to r^16: error < (0.347^17)/17! ~ 1e-22
    let mut term = 1.0f64;
    let mut sum = 1.0f64;
    let mut i = 1.0f64;
    while i <= 16.0 {
        term = term * r / i;
        sum += term;
        i += 1.0;
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
    let mut k = 3.0f64;
    while k <= 25.0 {
        term *= s2;
        sum += term / k;
        k += 2.0;
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
