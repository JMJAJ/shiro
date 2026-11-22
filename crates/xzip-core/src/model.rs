//! xzip CM model v1.
//!
//! - Hashed contexts live in 64-byte checksummed buckets, one bucket per
//!   (context, nibble): 15 bit-slots for the nibble's binary tree, plus a
//!   per-context byte "run" record (last byte seen in this context, repeat count).
//! - New high-order buckets inherit their parent order's slot probabilities
//!   (PPM information inheritance, ALGORITHMS.md §D6).
//! - Inputs: slot stretch, run-model prediction per context, match model, bias.
//! - Mixing: 4 context-selected logistic mixers -> final mixer -> APM chain.
//!
//! Determinism: integer counters, f32 arithmetic in fixed order, `dmath` exp/ln.

use crate::coder::{PBITS, PMAX};
use crate::dmath;

// ---------- helpers ----------

#[inline]
fn hash2(a: u64, b: u64) -> u64 {
    let h = (a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15)).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    h ^ (h >> 29)
}

struct Tables {
    stretch12: Vec<f32>,
    rate: [u32; 256],
}

impl Tables {
    fn new() -> Self {
        let stretch12 = (0..4096).map(|i| dmath::stretch((i as f64 + 0.5) / 4096.0) as f32).collect();
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

// ---------- hashed bucket table ----------

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Bucket {
    chk: u16,
    last: u8, // run model: last byte seen in this context (nibble-0 bucket only)
    run: u8,  // run model: times `last` repeated (0 = no record)
    s: [Slot; 15],
}

const BUCKET0: Bucket = Bucket { chk: 0, last: 0, run: 0, s: [SLOT0; 15] };

/// Find or allocate the bucket for hash `h`. Returns (index, newly_allocated).
#[inline]
fn lookup(tab: &mut [Bucket], bits: u32, h: u64) -> (usize, bool) {
    let chk = ((h >> 16) as u16) | 1;
    let mask = tab.len() - 1;
    let base = (h >> (64 - bits)) as usize;
    let mut victim = base & mask;
    let mut vp = u16::MAX;
    for k in 0..3usize {
        let ix = (base ^ k) & mask;
        let b = &tab[ix];
        if b.chk == chk {
            return (ix, false);
        }
        let pr = if b.chk == 0 { 0 } else { b.s[0].n + 1 };
        if pr < vp {
            vp = pr;
            victim = ix;
        }
    }
    tab[victim] = BUCKET0;
    tab[victim].chk = chk;
    (victim, true)
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
    t: Vec<f32>,
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
        let j = pos as usize;
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

/// Context layout:
///   0          order 0
///   1..=7      orders 1,2,3,4,5,6,8   (each inherits from the previous)
///   8, 9       word, word bigram
///   10, 11     sparse: byte 2; bytes 2..3
///   12         column: byte above in previous line + c1
const ORDERS: [u32; 7] = [1, 2, 3, 4, 5, 6, 8];
const N_CTX: usize = 13;
const N_IN: usize = 2 * N_CTX + 2 + 1;
const MATCH_MIN: usize = 6;
const MATCH_BITS: u32 = 22;
const INHERIT_N: u16 = 2;

pub struct ModelConfig {
    /// log2 of buckets per hashed table (64 bytes each).
    pub table_bits: u32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig { table_bits: 20 }
    }
}

pub struct Model {
    t: Tables,
    buf: Vec<u8>,
    c0: u32,
    bitpos: u32,
    c8: u64,
    word: u64,
    prevword: u64,
    line_start: usize,
    prev_line_start: usize,
    ctx: [u64; N_CTX],
    tables: Vec<Vec<Bucket>>,
    tbits: [u32; N_CTX],
    cur: [usize; N_CTX],
    b0: [usize; N_CTX],
    b0chk: [u16; N_CTX],
    hits: usize,
    node: usize,
    // run model
    rlast: [u8; N_CTX],
    rrun: [u8; N_CTX],
    ralive: [bool; N_CTX],
    rexp: [u32; N_CTX],
    ridx: [usize; N_CTX],
    runmap: Vec<Slot>,
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
    m1: [Mixer; 4],
    x2: [f32; 5],
    m2: Mixer,
    a1: Apm,
    a2: Apm,
    a3: Apm,
    a4: Apm,
    a3ctx: usize,
    bits_seen: u64,
}

impl Model {
    pub fn new(cfg: &ModelConfig) -> Self {
        let mut tables = Vec::with_capacity(N_CTX);
        let mut tbits = [0u32; N_CTX];
        for (i, tb) in tbits.iter_mut().enumerate() {
            let b = match i {
                0 => 10,
                1 => 16.min(cfg.table_bits),
                2 => 20.min(cfg.table_bits),
                _ => cfg.table_bits,
            };
            tables.push(vec![BUCKET0; 1usize << b]);
            *tb = b;
        }
        let mut m = Model {
            t: Tables::new(),
            buf: Vec::new(),
            c0: 1,
            bitpos: 0,
            c8: 0,
            word: 0,
            prevword: 0,
            line_start: 0,
            prev_line_start: 0,
            ctx: [0; N_CTX],
            tables,
            tbits,
            cur: [0; N_CTX],
            b0: [0; N_CTX],
            b0chk: [0; N_CTX],
            hits: 0,
            node: 1,
            rlast: [0; N_CTX],
            rrun: [0; N_CTX],
            ralive: [false; N_CTX],
            rexp: [0; N_CTX],
            ridx: [0; N_CTX],
            runmap: vec![SLOT0; N_CTX * 16],
            mtab: vec![0; 1 << MATCH_BITS],
            mptr: 0,
            mlen: 0,
            malive: false,
            mexp: 0,
            mbucket: 0,
            mcnt: vec![SLOT0; 64],
            x: [0.0; N_IN],
            m1: [
                Mixer::new(N_IN, 256, 0.2),
                Mixer::new(N_IN, 64, 0.2),
                Mixer::new(N_IN, 256, 0.2),
                Mixer::new(N_IN, 64, 0.2),
            ],
            x2: [0.0; 5],
            m2: Mixer::new(5, 8, 0.25),
            a1: Apm::new(256, 0.02),
            a2: Apm::new(65536, 0.02),
            a3: Apm::new(65536, 0.02),
            a4: Apm::new(65 * 256, 0.02),
            a3ctx: 0,
            bits_seen: 0,
        };
        m.byte_contexts();
        m.nibble_lookup(true);
        m
    }

    #[inline]
    fn c1(&self) -> u32 {
        (self.c8 & 0xff) as u32
    }

    fn byte_contexts(&mut self) {
        self.ctx[0] = hash2(0, 1);
        for (k, &o) in ORDERS.iter().enumerate() {
            let mask = if o >= 8 { u64::MAX } else { (1u64 << (8 * o)) - 1 };
            self.ctx[k + 1] = hash2(self.c8 & mask, o as u64 + 1);
        }
        self.ctx[8] = hash2(self.word, 101);
        self.ctx[9] = hash2(self.word ^ self.prevword.wrapping_mul(0x2545_F491_4F6C_DD1D), 102);
        self.ctx[10] = hash2(self.c8 & 0xff00, 103);
        self.ctx[11] = hash2(self.c8 & 0xff_ff00, 104);
        let pos = self.buf.len();
        let col = pos - self.line_start;
        let above = if self.prev_line_start + col < self.line_start {
            self.buf[self.prev_line_start + col] as u64
        } else {
            0
        };
        self.ctx[12] = hash2((above << 8) | (self.c8 & 0xff), 105);
        self.a3ctx = (hash2(self.c8 & 0xffff, 106) >> 56) as usize; // 8 bits of (c1,c2) hash
    }

    /// Look up (or allocate) the bucket of every context for the current nibble.
    fn nibble_lookup(&mut self, first: bool) {
        let salt = if first { 0 } else { self.c0 as u64 + 0x100 };
        let mut hits = 0;
        for i in 0..N_CTX {
            let h = if first { self.ctx[i] } else { hash2(self.ctx[i], salt) };
            let b = self.tbits[i];
            let (ix, new) = lookup(&mut self.tables[i], b, h);
            if new && (2..=7).contains(&i) {
                // inherit from the parent order's bucket for this nibble
                let parent = self.tables[i - 1][self.cur[i - 1]].s;
                let bk = &mut self.tables[i][ix];
                for j in 0..15 {
                    bk.s[j] = Slot { p: parent[j].p, n: parent[j].n.min(INHERIT_N) };
                }
            }
            self.cur[i] = ix;
            if first {
                self.b0[i] = ix;
                let bk = &self.tables[i][ix];
                self.b0chk[i] = bk.chk;
                self.rlast[i] = bk.last;
                self.rrun[i] = if new { 0 } else { bk.run };
                if !new && (1..=7).contains(&i) {
                    hits += 1;
                }
            }
        }
        if first {
            self.hits = hits;
        }
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
        self.node = if self.bitpos < 4 {
            self.c0 as usize
        } else {
            let k = self.bitpos - 4;
            (1usize << k) | (self.c0 as usize & ((1 << k) - 1))
        };
        let sn = self.node - 1;
        for i in 0..N_CTX {
            let s = self.tables[i][self.cur[i]].s[sn];
            self.x[i] = self.t.st16(s.p);
            // run model
            let run = self.rrun[i];
            let last = self.rlast[i] as u32;
            if run > 0 && ((last | 256) >> (8 - self.bitpos)) == self.c0 {
                let e = (last >> (7 - self.bitpos)) & 1;
                let ri = i * 16 + (run as usize).min(15);
                self.ralive[i] = true;
                self.rexp[i] = e;
                self.ridx[i] = ri;
                let st = self.t.st16(self.runmap[ri].p);
                self.x[N_CTX + i] = if e != 0 { st } else { -st };
            } else {
                self.ralive[i] = false;
                self.x[N_CTX + i] = 0.0;
            }
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
        let mi = 2 * N_CTX;
        let msel;
        if self.malive {
            let s = self.t.st16(self.mcnt[self.mbucket].p);
            let sign = if self.mexp != 0 { 1.0 } else { -1.0 };
            self.x[mi] = s * sign;
            self.x[mi + 1] = sign * self.mbucket as f32 * 0.0625;
            msel = 1 + self.mbucket * 2 + self.mexp as usize;
        } else {
            self.x[mi] = 0.0;
            self.x[mi + 1] = 0.0;
            msel = 0;
        }
        self.x[mi + 2] = 0.5;

        let c1 = self.c1() as usize;
        let c0 = self.c0 as usize;
        let d0 = self.m1[0].predict(&self.x, c0);
        let d1 = self.m1[1].predict(&self.x, msel.min(63));
        let d2 = self.m1[2].predict(&self.x, c1);
        let d3 = self.m1[3].predict(&self.x, self.hits * 8 + self.bitpos as usize);
        self.x2 = [d0, d1, d2, d3, 0.5];
        let d = self.m2.predict(&self.x2, self.bitpos as usize);
        let pm = self.m2.pr;
        let pa1 = self.a1.predict(d, c0);
        let pa2 = self.a2.predict(d, c0 | (c1 << 8));
        let pa3 = self.a3.predict(d, c0 | (self.a3ctx << 8));
        let pa4 = self.a4.predict(d, c0 | (msel.min(64) << 8));
        let p = 0.25 * pm + 0.15 * pa1 + 0.2 * pa2 + 0.25 * pa3 + 0.15 * pa4;
        let p24 = (p as f64 * (1u64 << PBITS) as f64) as i64;
        p24.clamp(1, PMAX as i64) as u32
    }

    pub fn update(&mut self, bit: u32) {
        self.bits_seen += 1;
        let sn = self.node - 1;
        for i in 0..N_CTX {
            let limit = if i <= 2 { 250 } else { 127 };
            let ix = self.cur[i];
            slot_update(&mut self.tables[i][ix].s[sn], bit, &self.t, limit);
            if self.ralive[i] {
                slot_update(&mut self.runmap[self.ridx[i]], (bit == self.rexp[i]) as u32, &self.t, 255);
            }
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
        self.a3.update(bit);
        self.a4.update(bit);

        self.c0 = (self.c0 << 1) | bit;
        self.bitpos += 1;
        if self.bitpos == 8 {
            let b = (self.c0 & 0xff) as u8;
            self.end_byte(b);
            self.c0 = 1;
            self.bitpos = 0;
            self.nibble_lookup(true);
        } else if self.bitpos == 4 {
            self.nibble_lookup(false);
        }
    }

    fn end_byte(&mut self, b: u8) {
        // run records for every context (nibble-0 buckets)
        for i in 0..N_CTX {
            let bk = &mut self.tables[i][self.b0[i]];
            if bk.chk != self.b0chk[i] {
                continue; // evicted meanwhile
            }
            if bk.run > 0 && bk.last == b {
                bk.run = bk.run.saturating_add(1);
            } else {
                bk.last = b;
                bk.run = 1;
            }
        }

        self.buf.push(b);
        let pos = self.buf.len();
        self.c8 = (self.c8 << 8) | b as u64;
        if b == b'\n' {
            self.prev_line_start = self.line_start;
            self.line_start = pos;
        }

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
