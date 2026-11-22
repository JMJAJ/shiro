//! xzip CM model v0: hashed order-N + word contexts, match model,
//! 2-layer gated logistic mixer, APM/SSE chain. Bitwise, MSB first.
//!
//! Every operation here must be deterministic across machines: integer
//! counters, f32 mixer arithmetic in a fixed order, and `dmath` for exp/ln.

use crate::coder::{PBITS, PMAX};
use crate::dmath;

// ---------- helpers ----------

#[inline]
fn hash2(a: u64, b: u64) -> u64 {
    let h = (a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15)).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    h ^ (h >> 29)
}

/// Stretch table over 12-bit quantised probabilities (inputs only; exactness
/// of inputs does not matter, determinism does).
struct Tables {
    stretch12: Vec<f32>,
    rate: [u32; 256],
}

impl Tables {
    fn new() -> Self {
        let stretch12 = (0..4096)
            .map(|i| dmath::stretch((i as f64 + 0.5) / 4096.0) as f32)
            .collect();
        let mut rate = [0u32; 256];
        for (n, r) in rate.iter_mut().enumerate() {
            *r = (65536.0 / (n as f64 + 1.6)) as u32;
        }
        Tables { stretch12, rate }
    }
    #[inline]
    fn st16(&self, p16: u16) -> f32 {
        self.stretch12[(p16 >> 4) as usize]
    }
}

#[inline]
fn squash32(x: f32) -> f32 {
    dmath::squash(x as f64) as f32
}

// ---------- adaptive probability slot ----------

#[derive(Clone, Copy)]
struct Slot {
    p: u16, // P(1) * 65536
    n: u16, // hit count (adaptation-rate state)
}

const SLOT0: Slot = Slot { p: 32768, n: 0 };

#[inline]
fn slot_update(s: &mut Slot, bit: u32, t: &Tables, limit: u16) {
    let target: i32 = if bit != 0 { 65535 } else { 0 };
    let p = s.p as i32;
    let r = t.rate[s.n as usize] as i64;
    let np = p as i64 + (((target - p) as i64 * r) >> 16);
    s.p = np.clamp(32, 65503) as u16;
    if s.n < limit {
        s.n += 1;
    }
}

// ---------- mixer ----------

struct Mixer {
    n: usize,
    w: Vec<f32>,
    sel: usize,
    dot: f32,
    pr: f32,
}

impl Mixer {
    fn new(n: usize, sets: usize, w0: f32) -> Self {
        Mixer { n, w: vec![w0; n * sets], sel: 0, dot: 0.0, pr: 0.5 }
    }
    #[inline]
    fn predict(&mut self, x: &[f32], sel: usize) -> f32 {
        self.sel = sel * self.n;
        let w = &self.w[self.sel..self.sel + self.n];
        let mut d = 0.0f32;
        for i in 0..self.n {
            d += w[i] * x[i];
        }
        self.dot = d.clamp(-15.0, 15.0);
        self.pr = squash32(self.dot);
        self.dot
    }
    #[inline]
    fn update(&mut self, x: &[f32], bit: u32, lr: f32) {
        let err = (bit as f32 - self.pr) * lr;
        let w = &mut self.w[self.sel..self.sel + self.n];
        for i in 0..self.n {
            w[i] += err * x[i];
        }
    }
}

// ---------- APM / SSE ----------

struct Apm {
    t: Vec<f32>, // 33 buckets per context
    idx: usize,
    f: f32,
    rate: f32,
}

impl Apm {
    fn new(ctxs: usize, rate: f32) -> Self {
        let mut t = Vec::with_capacity(ctxs * 33);
        for _ in 0..ctxs {
            for j in 0..33 {
                t.push(squash32((j as f32 - 16.0) * 0.5));
            }
        }
        Apm { t, idx: 0, f: 0.0, rate }
    }
    #[inline]
    fn predict(&mut self, st: f32, ctx: usize) -> f32 {
        let pos = (st.clamp(-7.999, 7.999) + 8.0) * 2.0;
        let j = pos as usize; // floor, pos >= 0
        self.f = pos - j as f32;
        self.idx = ctx * 33 + j;
        self.t[self.idx] * (1.0 - self.f) + self.t[self.idx + 1] * self.f
    }
    #[inline]
    fn update(&mut self, bit: u32) {
        let y = bit as f32;
        let (a, b) = (self.idx, self.idx + 1);
        self.t[a] += self.rate * (1.0 - self.f) * (y - self.t[a]);
        self.t[b] += self.rate * self.f * (y - self.t[b]);
    }
}

// ---------- model ----------

/// Context orders (bytes) for hashed order-N models.
const ORDERS: [u32; 7] = [1, 2, 3, 4, 5, 6, 8];
const N_CTX: usize = 1 + ORDERS.len() + 2; // o0, orders, word, word-bigram
const N_IN: usize = N_CTX + 2 + 1; // + 2 match inputs + bias
const MATCH_MIN: usize = 6;
const MATCH_BITS: u32 = 22;

pub struct ModelConfig {
    pub table_bits: u32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig { table_bits: 22 }
    }
}

pub struct Model {
    t: Tables,
    buf: Vec<u8>,
    c0: u32,     // partial byte with leading 1
    bitpos: u32, // 0..7
    c8: u64,     // last 8 bytes
    word: u64,
    prevword: u64,
    ctx: [u64; N_CTX],
    tables: Vec<Vec<Slot>>,
    tbits: Vec<u32>,
    idx: [usize; N_CTX],
    // match model
    mtab: Vec<u32>,
    mptr: usize,
    mlen: usize,
    malive: bool,
    mexp: u32,
    mbucket: usize,
    mcnt: Vec<Slot>,
    // mixing
    x: [f32; N_IN],
    m1: [Mixer; 3],
    x2: [f32; 4],
    m2: Mixer,
    a1: Apm,
    a2: Apm,
    st_final: f32,
    bits_seen: u64,
}

impl Model {
    pub fn new(cfg: &ModelConfig) -> Self {
        let mut tables = Vec::with_capacity(N_CTX);
        let mut tbits = Vec::with_capacity(N_CTX);
        for i in 0..N_CTX {
            let b = if i == 0 { 16 } else if i <= 2 { 22.min(cfg.table_bits) } else { cfg.table_bits };
            tables.push(vec![SLOT0; 1usize << b]);
            tbits.push(b);
        }
        let mut m = Model {
            t: Tables::new(),
            buf: Vec::new(),
            c0: 1,
            bitpos: 0,
            c8: 0,
            word: 0,
            prevword: 0,
            ctx: [0; N_CTX],
            tables,
            tbits,
            idx: [0; N_CTX],
            mtab: vec![0; 1 << MATCH_BITS],
            mptr: 0,
            mlen: 0,
            malive: false,
            mexp: 0,
            mbucket: 0,
            mcnt: vec![SLOT0; 64],
            x: [0.0; N_IN],
            m1: [Mixer::new(N_IN, 256, 0.3), Mixer::new(N_IN, 64, 0.3), Mixer::new(N_IN, 256, 0.3)],
            x2: [0.0; 4],
            m2: Mixer::new(4, 8, 0.33),
            a1: Apm::new(256, 0.02),
            a2: Apm::new(65536, 0.02),
            st_final: 0.0,
            bits_seen: 0,
        };
        m.byte_contexts();
        m
    }

    #[inline]
    fn c1(&self) -> u32 {
        (self.c8 & 0xff) as u32
    }

    fn byte_contexts(&mut self) {
        self.ctx[0] = 0;
        for (k, &o) in ORDERS.iter().enumerate() {
            let mask = if o >= 8 { u64::MAX } else { (1u64 << (8 * o)) - 1 };
            self.ctx[k + 1] = hash2(self.c8 & mask, o as u64 + 1);
        }
        self.ctx[N_CTX - 2] = hash2(self.word, 101);
        self.ctx[N_CTX - 1] = hash2(self.word ^ self.prevword.wrapping_mul(0x2545_F491_4F6C_DD1D), 102);
    }

    fn match_bucket(len: usize) -> usize {
        if len < 16 {
            len
        } else {
            (16 + (len - 16) / 8).min(31)
        }
    }

    /// Probability that the next bit is 1, as a 24-bit integer.
    pub fn predict(&mut self) -> u32 {
        let c0 = self.c0 as u64;
        let h0 = c0.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for i in 0..N_CTX {
            let b = self.tbits[i];
            let ix = if i == 0 {
                self.c0 as usize
            } else {
                ((self.ctx[i] ^ h0).wrapping_mul(0xD6E8_FEB8_6659_FD93) >> (64 - b)) as usize
            };
            self.idx[i] = ix;
            self.x[i] = self.t.st16(self.tables[i][ix].p);
        }

        // match model
        self.malive = false;
        if self.mlen > 0 {
            let pb = self.buf[self.mptr] as u32;
            if ((pb | 256) >> (8 - self.bitpos)) == self.c0 {
                self.malive = true;
                self.mexp = (pb >> (7 - self.bitpos)) & 1;
                self.mbucket = Self::match_bucket(self.mlen);
            }
        }
        let msel;
        if self.malive {
            let pc = self.mcnt[self.mbucket].p;
            let s = self.t.st16(pc);
            let sign = if self.mexp != 0 { 1.0 } else { -1.0 };
            self.x[N_CTX] = s * sign;
            self.x[N_CTX + 1] = sign * self.mbucket as f32 * 0.0625;
            msel = 1 + self.mbucket * 2 + self.mexp as usize;
        } else {
            self.x[N_CTX] = 0.0;
            self.x[N_CTX + 1] = 0.0;
            msel = 0;
        }
        self.x[N_CTX + 2] = 0.5;

        let c1 = self.c1() as usize;
        let d0 = self.m1[0].predict(&self.x, self.c0 as usize);
        let d1 = self.m1[1].predict(&self.x, msel.min(63));
        let d2 = self.m1[2].predict(&self.x, c1);
        self.x2 = [d0, d1, d2, 0.5];
        let d = self.m2.predict(&self.x2, self.bitpos as usize);
        self.st_final = d;
        let pm = self.m2.pr;
        let pa1 = self.a1.predict(d, self.c0 as usize);
        let pa2 = self.a2.predict(d, (self.c0 as usize) | (c1 << 8));
        let p = (pm + pa1 + 2.0 * pa2) * 0.25;
        let p24 = (p as f64 * (1u64 << PBITS) as f64) as i64;
        p24.clamp(1, PMAX as i64) as u32
    }

    pub fn update(&mut self, bit: u32) {
        self.bits_seen += 1;
        for i in 0..N_CTX {
            let limit = if i < 3 { 250 } else { 127 };
            slot_update(&mut self.tables[i][self.idx[i]], bit, &self.t, limit);
        }
        if self.malive {
            slot_update(&mut self.mcnt[self.mbucket], (bit == self.mexp) as u32, &self.t, 255);
        }
        let lr = 0.002 + 0.02 / (1.0 + self.bits_seen as f32 * (1.0 / 1_048_576.0));
        for m in self.m1.iter_mut() {
            m.update(&self.x, bit, lr);
        }
        self.m2.update(&self.x2, bit, lr * 0.5);
        self.a1.update(bit);
        self.a2.update(bit);

        self.c0 = (self.c0 << 1) | bit;
        self.bitpos += 1;
        if self.bitpos == 8 {
            let b = (self.c0 & 0xff) as u8;
            self.end_byte(b);
            self.c0 = 1;
            self.bitpos = 0;
        }
    }

    fn end_byte(&mut self, b: u8) {
        self.buf.push(b);
        let pos = self.buf.len();
        self.c8 = (self.c8 << 8) | b as u64;

        let is_letter = b.is_ascii_alphabetic() || b >= 128;
        if is_letter {
            self.word = hash2(self.word, b.to_ascii_lowercase() as u64 + 1);
        } else if self.word != 0 {
            self.prevword = self.word;
            self.word = 0;
        }

        // match model maintenance
        if self.mlen > 0 {
            if self.buf[self.mptr] == b {
                self.mlen += 1;
                self.mptr += 1;
            } else {
                self.mlen = 0;
            }
        }
        if pos >= MATCH_MIN {
            let h = (hash2(self.c8 & 0xffff_ffff_ffff, 7) >> (64 - MATCH_BITS)) as usize;
            if self.mlen == 0 {
                let cand = self.mtab[h] as usize;
                if cand > 0 {
                    let mut l = 0;
                    while l < 400 && cand > l && self.buf[cand - 1 - l] == self.buf[pos - 1 - l] {
                        l += 1;
                    }
                    if l >= MATCH_MIN {
                        self.mlen = l;
                        self.mptr = cand;
                    }
                }
            }
            self.mtab[h] = pos as u32;
        }
        self.byte_contexts();
    }
}
