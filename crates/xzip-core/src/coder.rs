//! Binary arithmetic coder (carryless, 32-bit state) driven by 24-bit
//! probabilities. P(bit = 1) = p24 / 2^24, p24 in [1, 2^24 - 1].
//!
//! 24-bit probabilities matter at enwik9 scale: with the 12-bit paq
//! convention, clamping alone can cost ~350 KB (see ALGORITHMS.md §A3).

pub const PBITS: u32 = 24;
pub const PMAX: u32 = (1 << PBITS) - 1;

#[inline]
fn split(x1: u32, x2: u32, p24: u32) -> u32 {
    debug_assert!(p24 >= 1 && p24 <= PMAX);
    let range = (x2 - x1) as u64;
    x1 + ((range * p24 as u64) >> PBITS) as u32
}

pub struct Encoder {
    x1: u32,
    x2: u32,
    out: Vec<u8>,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder {
    pub fn new() -> Self {
        Encoder { x1: 0, x2: u32::MAX, out: Vec::new() }
    }

    #[inline]
    pub fn encode(&mut self, bit: u32, p24: u32) {
        let xmid = split(self.x1, self.x2, p24);
        if bit != 0 {
            self.x2 = xmid;
        } else {
            self.x1 = xmid + 1;
        }
        while (self.x1 ^ self.x2) & 0xff00_0000 == 0 {
            self.out.push((self.x2 >> 24) as u8);
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 255;
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        // Any value in [x1, x2] decodes correctly; emit x1 in full.
        self.out.extend_from_slice(&self.x1.to_be_bytes());
        self.out
    }
}

pub struct Decoder<'a> {
    x1: u32,
    x2: u32,
    x: u32,
    data: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut d = Decoder { x1: 0, x2: u32::MAX, x: 0, data, pos: 0 };
        for _ in 0..4 {
            d.x = (d.x << 8) | d.next_byte();
        }
        d
    }

    #[inline]
    fn next_byte(&mut self) -> u32 {
        let b = self.data.get(self.pos).copied().unwrap_or(0) as u32;
        self.pos += 1;
        b
    }

    #[inline]
    pub fn decode(&mut self, p24: u32) -> u32 {
        let xmid = split(self.x1, self.x2, p24);
        let bit = if self.x <= xmid {
            self.x2 = xmid;
            1
        } else {
            self.x1 = xmid + 1;
            0
        };
        while (self.x1 ^ self.x2) & 0xff00_0000 == 0 {
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 255;
            self.x = (self.x << 8) | self.next_byte();
        }
        bit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s >> 33
    }

    #[test]
    fn roundtrip_random_probs() {
        let mut s = 12345u64;
        let n = 200_000;
        let mut bits = Vec::with_capacity(n);
        let mut probs = Vec::with_capacity(n);
        for _ in 0..n {
            let p = (lcg(&mut s) as u32 % PMAX).max(1);
            // draw bit roughly according to p, sometimes adversarially
            let r = (lcg(&mut s) as u32) & PMAX;
            let bit = if lcg(&mut s) % 50 == 0 { (r & 1) as u32 } else { (r < p) as u32 };
            bits.push(bit);
            probs.push(p);
        }
        let mut e = Encoder::new();
        for i in 0..n {
            e.encode(bits[i], probs[i]);
        }
        let buf = e.finish();
        let mut d = Decoder::new(&buf);
        for i in 0..n {
            assert_eq!(d.decode(probs[i]), bits[i], "bit {i}");
        }
    }

    #[test]
    fn extreme_probs() {
        let mut e = Encoder::new();
        let pat: Vec<(u32, u32)> =
            (0..10_000).map(|i| ((i % 7 == 0) as u32, if i % 2 == 0 { 1 } else { PMAX })).collect();
        for &(b, p) in &pat {
            e.encode(b, p);
        }
        let buf = e.finish();
        let mut d = Decoder::new(&buf);
        for &(b, p) in &pat {
            assert_eq!(d.decode(p), b);
        }
    }
}
