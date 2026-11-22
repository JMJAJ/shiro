//! xzip CM model v4.
//!
//! - Hashed contexts live in 64-byte checksummed buckets, one bucket per
//!   (context, nibble): 15 bit-slots for the nibble's binary tree, plus a
//!   per-context byte "run" record (last byte seen in this context, repeat count).
//! - New high-order buckets inherit their parent order's slot probabilities
//!   (PPM information inheritance, ALGORITHMS.md §D6).
//! - Inputs per context: slot stretch, indirect (bit-history) map, run model.
//!   Plus two match models (short / long minimum length) and a bias.
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
    n: u8,  // hit count (adaptation-rate state)
    h: u8,  // bit history in this slot, marker-prefixed (1 = empty), last <=7 bits
}

const SLOT0: Slot = Slot { p: 32768, n: 0, h: 1 };

#[inline]
fn slot_update(s: &mut Slot, bit: u32, t: &Tables, limit: u8) {
    let target: i32 = if bit != 0 { 65535 } else { 0 };
    let p = s.p as i32;
    let r = t.rate[s.n as usize] as i64;
    let np = p as i64 + (((target - p) as i64 * r) >> 16);
    s.p = np.clamp(32, 65503) as u16;
    if s.n < limit {
        s.n += 1;
    }
    let h = ((s.h as u32) << 1) | bit;
    s.h = if h >= 256 { ((h & 0x7f) | 0x80) as u8 } else { h as u8 };
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
        let pr = if b.chk == 0 { 0 } else { b.s[0].n as u16 + 1 };
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
        let row: Vec<f32> = (0..33).map(|j| squash32((j as f32 - 16.0) * 0.5)).collect();
        let mut t = Vec::with_capacity(ctxs * 33);
        for _ in 0..ctxs {
            t.extend_from_slice(&row);
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

// ---------- match model ----------

struct MatchModel {
    minlen: usize,
    bits: u32,
    tab: Vec<u32>,
    ptr: usize,
    len: usize,
    alive: bool,
    exp: u32,
    bucket: usize,
    cnt: Vec<Slot>,
}

impl MatchModel {
    fn new(minlen: usize, bits: u32) -> Self {
        MatchModel {
            minlen,
            bits,
            tab: vec![0; 1 << bits],
            ptr: 0,
            len: 0,
            alive: false,
            exp: 0,
            bucket: 0,
            cnt: vec![SLOT0; 64],
        }
    }

    fn len_bucket(len: usize) -> usize {
        if len < 16 {
            len
        } else {
            (16 + (len - 16) / 8).min(31)
        }
    }

    /// Writes 2 inputs; returns mixer-select value (0 = no match, else 1..=63).
    #[inline]
    fn predict(&mut self, buf: &[u8], c0: u32, bitpos: u32, t: &Tables, x: &mut [f32]) -> usize {
        self.alive = false;
        if self.len > 0 {
            let pb = buf[self.ptr] as u32;
            if ((pb | 256) >> (8 - bitpos)) == c0 {
                self.alive = true;
                self.exp = (pb >> (7 - bitpos)) & 1;
                self.bucket = Self::len_bucket(self.len);
            }
        }
        if self.alive {
            let s = t.st16(self.cnt[self.bucket].p);
            let sign = if self.exp != 0 { 1.0 } else { -1.0 };
            x[0] = s * sign;
            x[1] = sign * self.bucket as f32 * 0.0625;
            1 + self.bucket * 2 + self.exp as usize
        } else {
            x[0] = 0.0;
            x[1] = 0.0;
            0
        }
    }

    #[inline]
    fn update(&mut self, bit: u32, t: &Tables) {
        if self.alive {
            slot_update(&mut self.cnt[self.bucket], (bit == self.exp) as u32, t, 255);
        }
    }

    fn end_byte(&mut self, buf: &[u8]) {
        let pos = buf.len();
        let b = buf[pos - 1];
        if self.len > 0 {
            if buf[self.ptr] == b {
                self.len += 1;
                self.ptr += 1;
            } else {
                self.len = 0;
            }
        }
        if pos >= self.minlen {
            let mut h = self.minlen as u64;
            for &c in &buf[pos - self.minlen..pos] {
                h = hash2(h, c as u64 + 1);
            }
            let hi = (h >> (64 - self.bits)) as usize;
            if self.len == 0 {
                let cand = self.tab[hi] as usize;
                if cand > 0 {
                    let mut l = 0;
                    while l < 400 && cand > l && buf[cand - 1 - l] == buf[pos - 1 - l] {
                        l += 1;
                    }
                    if l >= self.minlen {
                        self.len = l;
                        self.ptr = cand;
                    }
                }
            }
            self.tab[hi] = pos as u32;
        }
    }
}

// ---------- tuning knobs ----------

/// Research knobs. Archives always use `Tune::default()`; `eval` may override
/// via `XZIP_TUNE="key=val,key=val"` for parameter sweeps.
#[derive(Clone, Debug)]
pub struct Tune {
    pub lr_hi: f32,
    pub lr_lo: f32,
    pub lr_t: f32,
    pub m2_scale: f32,
    pub apm_rate: f32,
    pub inherit_n: u8,
    pub limit_hi: u8,
    pub limit_lo: u8,
    pub w0: f32,
    pub pmix: f32,
    /// final mixer weight-set selection: 0 = bitpos, 1 = c0
    pub m2sel: u8,
    /// hash bits of the order-2 APM context (<= 12)
    pub a3bits: u8,
    /// enable the word-length-selected mixer
    pub mw: u8,
    /// learned final blend of mixer + APMs: 0 off, 1 replace, 2 average with fixed blend
    pub fm: u8,
    /// ablation: bit k turns context 20+k into a constant
    pub xoff: u32,
    /// bit-history map index includes a confidence bucket of the slot count
    pub ih: u8,
    /// order-3 APM in the learned final blend
    pub a5: u8,
    /// mixer selected by the previous two bytes
    pub m7: u8,
    /// side (neural) prediction also enters the final blend: raw + APM(c0,c1)
    pub sf: u8,
}

impl Default for Tune {
    fn default() -> Self {
        Tune {
            lr_hi: 0.01,
            lr_lo: 0.0015,
            lr_t: 1_048_576.0,
            m2_scale: 0.2,
            apm_rate: 0.02,
            inherit_n: 0,
            limit_hi: 127,
            limit_lo: 250,
            w0: 0.2,
            pmix: 0.25,
            m2sel: 1,
            a3bits: 12,
            mw: 1,
            fm: 2,
            xoff: 0,
            ih: 1,
            a5: 1,
            m7: 1,
            sf: 0,
        }
    }
}

impl Tune {
    pub fn parse(spec: &str) -> Result<Tune, String> {
        let mut t = Tune::default();
        for kv in spec.split(',').filter(|s| !s.trim().is_empty()) {
            let (k, v) = kv.split_once('=').ok_or(format!("bad tune item {kv}"))?;
            let f: f64 = v.trim().parse().map_err(|_| format!("bad value {v}"))?;
            match k.trim() {
                "lr_hi" => t.lr_hi = f as f32,
                "lr_lo" => t.lr_lo = f as f32,
                "lr_t" => t.lr_t = f as f32,
                "m2_scale" => t.m2_scale = f as f32,
                "apm_rate" => t.apm_rate = f as f32,
                "inherit_n" => t.inherit_n = f as u8,
                "limit_hi" => t.limit_hi = f as u8,
                "limit_lo" => t.limit_lo = f as u8,
                "w0" => t.w0 = f as f32,
                "pmix" => t.pmix = f as f32,
                "m2sel" => t.m2sel = f as u8,
                "a3bits" => t.a3bits = (f as u8).min(12),
                "mw" => t.mw = f as u8,
                "fm" => t.fm = f as u8,
                "xoff" => t.xoff = f as u32,
                "ih" => t.ih = f as u8,
                "a5" => t.a5 = f as u8,
                "m7" => t.m7 = f as u8,
                "sf" => t.sf = f as u8,
                _ => return Err(format!("unknown tune key {k}")),
            }
        }
        Ok(t)
    }
}

// ---------- model ----------

/// Context layout:
///   0          order 0
///   1..=7      orders 1,2,3,4,5,6,8   (each inherits from the previous)
///   8, 9, 10   word, word bigram, word trigram
///   11, 12     sparse: byte 2; bytes 2..3
///   13         column: byte above in previous line + c1
///   14         word skip: word0 + word2
///   15         bracket: innermost open bracket type + depth + word0
///   16         line head: first two bytes of the line + c1
///   17, 18     orders 12, 16
///   19         indirect: the two bytes that followed the last two occurrences of c2c1, + c1
///   20         XML tag: name of the last opened tag + order 3
///   21         number: digit-run length + 2 bytes before the run + c1
///   22         sparse: bytes 1 and 3
///   23         markup segment: opener ([[ | = etc.) + distance bucket + word0
const ORDERS: [u32; 7] = [1, 2, 3, 4, 5, 6, 8];
const N_CTX: usize = 24;
/// per context: slot, indirect, run, confidence; + 2x2 match + bias
const N_IN: usize = 4 * N_CTX + 4 + 1 + 2; // + side (neural) input, present flag

/// Coarse character class used for mixer selection.
#[inline]
fn cclass(b: u8) -> usize {
    match b {
        b'a'..=b'z' => 1,
        b'A'..=b'Z' => 2,
        b'0'..=b'9' => 3,
        b'.' | b',' | b';' | b':' | b'!' | b'?' => 4,
        b'[' | b']' | b'{' | b'}' | b'<' | b'>' | b'|' | b'=' => 5,
        128..=255 => 7,
        0..=32 => 0,
        _ => 6,
    }
}

pub struct ModelConfig {
    /// log2 of buckets per hashed table (64 bytes each).
    pub table_bits: u32,
    pub tune: Tune,
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig { table_bits: 20, tune: Tune::default() }
    }
}

pub struct Model {
    t: Tables,
    tune: Tune,
    buf: Vec<u8>,
    c0: u32,
    bitpos: u32,
    c8: u64,
    word: u64,
    prevword: u64,
    prevword2: u64,
    line_start: usize,
    prev_line_start: usize,
    brk: [u8; 16],
    brk_n: usize,
    ind2: Vec<u16>,
    ctx: [u64; N_CTX],
    tables: Vec<Vec<Bucket>>,
    tbits: [u32; N_CTX],
    cur: [usize; N_CTX],
    b0: [usize; N_CTX],
    b0chk: [u16; N_CTX],
    hits: usize,
    node: usize,
    // indirect (bit-history) maps
    hidx: [usize; N_CTX],
    imap: Vec<Slot>,
    // run model
    rlast: [u8; N_CTX],
    rrun: [u8; N_CTX],
    ralive: [bool; N_CTX],
    rexp: [u32; N_CTX],
    ridx: [usize; N_CTX],
    runmap: Vec<Slot>,
    // match models
    mm: [MatchModel; 2],
    // mixing
    x: [f32; N_IN],
    m1: [Mixer; 8],
    x2: [f32; 10],
    wlen: usize,
    tag: u64,
    in_tag: bool,
    tag_done: bool,
    drun: u64,
    dpre: u64,
    seg: u64,
    seg_pos: usize,
    side: u16,
    side_st: f32,
    m2: Mixer,
    fmx: Mixer,
    xf: [f32; 9],
    a6: Apm,
    a6on: bool,
    a1: Apm,
    a2: Apm,
    a3: Apm,
    a4: Apm,
    a5: Apm,
    a5ctx: usize,
    a3ctx: usize,
    bits_seen: u64,
}

impl Model {
    pub fn new(cfg: &ModelConfig) -> Self {
        let tune = cfg.tune.clone();
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
        let w0 = tune.w0;
        let ar = tune.apm_rate;
        let mut m = Model {
            t: Tables::new(),
            buf: Vec::new(),
            c0: 1,
            bitpos: 0,
            c8: 0,
            word: 0,
            prevword: 0,
            prevword2: 0,
            line_start: 0,
            prev_line_start: 0,
            brk: [0; 16],
            brk_n: 0,
            ind2: vec![0; 65536],
            ctx: [0; N_CTX],
            tables,
            tbits,
            cur: [0; N_CTX],
            b0: [0; N_CTX],
            b0chk: [0; N_CTX],
            hits: 0,
            node: 1,
            hidx: [0; N_CTX],
            imap: vec![SLOT0; N_CTX * 1024],
            rlast: [0; N_CTX],
            rrun: [0; N_CTX],
            ralive: [false; N_CTX],
            rexp: [0; N_CTX],
            ridx: [0; N_CTX],
            runmap: vec![SLOT0; N_CTX * 16],
            mm: [MatchModel::new(5, 22), MatchModel::new(16, 22)],
            x: [0.0; N_IN],
            m1: [
                Mixer::new(N_IN, 256, w0),
                Mixer::new(N_IN, 64, w0),
                Mixer::new(N_IN, 256, w0),
                Mixer::new(N_IN, 64, w0),
                Mixer::new(N_IN, 512, w0),
                Mixer::new(N_IN, 33 * 8, w0),
                Mixer::new(N_IN, 16 * 8 * 8, w0),
                Mixer::new(N_IN, 65536, w0),
            ],
            x2: [0.0; 10],
            wlen: 0,
            tag: 0,
            in_tag: false,
            tag_done: true,
            drun: 0,
            dpre: 0,
            seg: 0,
            seg_pos: 0,
            side: 0,
            side_st: 0.0,
            m2: Mixer::new(10, 256, 0.15),
            fmx: Mixer::new(9, 512, 0.2),
            xf: [0.0; 9],
            a6: Apm::new(65536, ar),
            a6on: false,
            a1: Apm::new(256, ar),
            a2: Apm::new(65536, ar),
            a3: Apm::new(1 << 20, ar),
            a4: Apm::new(65 * 256, ar),
            a5: Apm::new(1 << 20, ar),
            a5ctx: 0,
            a3ctx: 0,
            bits_seen: 0,
            tune,
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
        let w1 = self.prevword.wrapping_mul(0x2545_F491_4F6C_DD1D);
        // between words, condition on the previous word and the delimiter
        self.ctx[8] = if self.word != 0 {
            hash2(self.word, 101)
        } else {
            hash2(self.prevword ^ (self.c8 & 0xff), 108)
        };
        self.ctx[9] = hash2(self.word ^ w1, 102);
        self.ctx[10] = hash2(self.word ^ w1 ^ self.prevword2.wrapping_mul(0x9FB2_1C65_1E98_DF25), 107);
        self.ctx[11] = hash2(self.c8 & 0xff00, 103);
        self.ctx[12] = hash2(self.c8 & 0xff_ff00, 104);
        let pos = self.buf.len();
        let col = pos - self.line_start;
        let above = if self.prev_line_start + col < self.line_start {
            self.buf[self.prev_line_start + col] as u64
        } else {
            0
        };
        self.ctx[13] = hash2((above << 8) | (self.c8 & 0xff), 105);
        self.ctx[14] = hash2(self.word ^ self.prevword2.wrapping_mul(0x9FB2_1C65_1E98_DF25), 109);
        let top = if self.brk_n > 0 { self.brk[(self.brk_n - 1).min(15)] as u64 } else { 0 };
        self.ctx[15] = hash2(self.word ^ (top << 56) ^ ((self.brk_n.min(7) as u64) << 48), 110);
        let head = if col >= 2 {
            ((self.buf[self.line_start] as u64) << 8) | self.buf[self.line_start + 1] as u64
        } else if col == 1 {
            self.buf[self.line_start] as u64
        } else {
            0
        };
        self.ctx[16] = hash2((head << 8) | (self.c8 & 0xff), 111);
        let n = self.buf.len();
        let mut h12 = 0u64;
        for &c in &self.buf[n.saturating_sub(12)..] {
            h12 = h12.wrapping_mul(0x100_0000_01B3).wrapping_add(c as u64 + 1);
        }
        let mut h16 = h12;
        for &c in &self.buf[n.saturating_sub(16)..n.saturating_sub(12)] {
            h16 = h16.wrapping_mul(0x100_0000_01B3).wrapping_add(c as u64 + 7);
        }
        self.ctx[17] = hash2(h12, 112);
        self.ctx[18] = hash2(h16, 113);
        let iv = self.ind2[(self.c8 & 0xffff) as usize] as u64;
        self.ctx[19] = hash2((iv << 8) | (self.c8 & 0xff), 114);
        self.ctx[20] = hash2(self.tag ^ ((self.c8 & 0xff_ffff) << 1) ^ (self.in_tag as u64), 115);
        self.ctx[21] = hash2((self.drun.min(15) << 32) | (self.dpre << 8) | (self.c8 & 0xff), 116);
        self.ctx[22] = hash2(self.c8 & 0xff_00ff, 117);
        let sd = (n - self.seg_pos).min(15) as u64;
        self.ctx[23] = hash2(self.word ^ (self.seg << 56) ^ (sd << 48), 118);
        let xo = self.tune.xoff;
        for k in 0..4 {
            if xo >> k & 1 != 0 {
                self.ctx[20 + k] = hash2(0, 200 + k as u64);
            }
        }
        self.a3ctx = (hash2(self.c8 & 0xffff, 106) >> (64 - self.tune.a3bits as u32)) as usize;
        self.a5ctx = (hash2(self.c8 & 0xff_ffff, 119) >> 52) as usize;
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
                let parent = self.tables[i - 1][self.cur[i - 1]].s;
                let inh = self.tune.inherit_n;
                let bk = &mut self.tables[i][ix];
                for j in 0..15 {
                    bk.s[j] = Slot { p: parent[j].p, n: parent[j].n.min(inh), h: 1 };
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

    /// External (neural) prediction for the next bit: P(1) * 65536, 0 = none.
    /// Must be set before every `predict` when used; it is cleared by `update`.
    #[inline]
    pub fn set_side(&mut self, p16: u16) {
        self.side = p16;
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
            let nb = if self.tune.ih != 0 { [0usize, 1, 1, 2, 2, 2, 2, 2][(s.n as usize).min(7)] + (s.n >= 30) as usize } else { 0 };
            let hi = (i * 4 + nb) * 256 + s.h as usize;
            self.hidx[i] = hi;
            self.x[N_CTX + i] = self.t.st16(self.imap[hi].p);
            let run = self.rrun[i];
            let last = self.rlast[i] as u32;
            if run > 0 && ((last | 256) >> (8 - self.bitpos)) == self.c0 {
                let e = (last >> (7 - self.bitpos)) & 1;
                let ri = i * 16 + (run as usize).min(15);
                self.ralive[i] = true;
                self.rexp[i] = e;
                self.ridx[i] = ri;
                let st = self.t.st16(self.runmap[ri].p);
                self.x[2 * N_CTX + i] = if e != 0 { st } else { -st };
            } else {
                self.ralive[i] = false;
                self.x[2 * N_CTX + i] = 0.0;
            }
            self.x[3 * N_CTX + i] = (s.p as f32 - 32768.0) * (1.0 / 16384.0);
        }

        let mi = 4 * N_CTX;
        let (xa, xb) = self.x[mi..mi + 4].split_at_mut(2);
        let s0 = self.mm[0].predict(&self.buf, self.c0, self.bitpos, &self.t, xa);
        let s1 = self.mm[1].predict(&self.buf, self.c0, self.bitpos, &self.t, xb);
        self.x[mi + 4] = 0.5;
        let present = self.side != 0;
        self.side_st = if present { self.t.st16(self.side) } else { 0.0 };
        self.x[mi + 5] = self.side_st;
        self.x[mi + 6] = if present { 0.5 } else { 0.0 };
        let ssel = if present { 1 + ((self.side_st + 8.0) * 2.0).clamp(0.0, 31.0) as usize } else { 0 };
        let msel = if s1 > 0 { s1 } else { s0 };

        let c1 = self.c1() as usize;
        let c0 = self.c0 as usize;
        let d0 = self.m1[0].predict(&self.x, c0);
        let d1 = self.m1[1].predict(&self.x, msel.min(63));
        let d2 = self.m1[2].predict(&self.x, c1);
        let d3 = self.m1[3].predict(&self.x, self.hits * 8 + self.bitpos as usize);
        let cls = cclass(c1 as u8) * 8 + cclass((self.c8 >> 8) as u8);
        let d4 = self.m1[4].predict(&self.x, cls * 8 + self.bitpos as usize);
        let d5 = self.m1[5].predict(&self.x, ssel * 8 + self.bitpos as usize);
        let d6 = if self.tune.mw != 0 {
            let ws = (self.wlen.min(15) * 8 + cclass(c1 as u8)) * 8 + self.bitpos as usize;
            self.m1[6].predict(&self.x, ws)
        } else {
            0.0
        };
        let d7 = if self.tune.m7 != 0 { self.m1[7].predict(&self.x, (self.c8 & 0xffff) as usize) } else { 0.0 };
        self.x2 = [d0, d1, d2, d3, d4, d5, d6, d7, self.side_st, 0.5];
        let s2 = if self.tune.m2sel == 1 { c0 } else { self.bitpos as usize };
        let d = self.m2.predict(&self.x2, s2);
        let pm = self.m2.pr;
        let pa1 = self.a1.predict(d, c0);
        let pa2 = self.a2.predict(d, c0 | (c1 << 8));
        let pa3 = self.a3.predict(d, c0 | (self.a3ctx << 8));
        let pa4 = self.a4.predict(d, c0 | (msel.min(64) << 8));
        let pmx = self.tune.pmix;
        let r = 1.0 - pmx;
        let mut p = pmx * pm + r * (0.2 * pa1 + 0.27 * pa2 + 0.33 * pa3 + 0.2 * pa4);
        if self.tune.fm != 0 {
            let st = |q: f32| dmath::stretch(q.clamp(1e-6, 1.0 - 1e-6) as f64) as f32;
            let pa5 = if self.tune.a5 != 0 { self.a5.predict(d, c0 | (self.a5ctx << 8)) } else { 0.5 };
            self.a6on = self.tune.sf != 0 && present;
            let (xs, xa6) = if self.a6on {
                let q6 = self.a6.predict(self.side_st, c0 | (c1 << 8));
                (self.side_st, st(q6))
            } else {
                (0.0, 0.0)
            };
            self.xf = [st(pm), st(pa1), st(pa2), st(pa3), st(pa4), st(pa5), xs, xa6, 0.5];
            self.fmx.predict(&self.xf, c0 | ((present as usize) << 8));
            let q = self.fmx.pr;
            p = if self.tune.fm == 1 { q } else { 0.5 * (p + q) };
        }
        let p24 = (p as f64 * (1u64 << PBITS) as f64) as i64;
        p24.clamp(1, PMAX as i64) as u32
    }

    pub fn update(&mut self, bit: u32) {
        self.bits_seen += 1;
        self.side = 0;
        let sn = self.node - 1;
        let (lim_lo, lim_hi) = (self.tune.limit_lo, self.tune.limit_hi);
        for i in 0..N_CTX {
            let limit = if i <= 2 { lim_lo } else { lim_hi };
            let ix = self.cur[i];
            slot_update(&mut self.tables[i][ix].s[sn], bit, &self.t, limit);
            slot_update(&mut self.imap[self.hidx[i]], bit, &self.t, 255);
            if self.ralive[i] {
                slot_update(&mut self.runmap[self.ridx[i]], (bit == self.rexp[i]) as u32, &self.t, 255);
            }
        }
        for m in self.mm.iter_mut() {
            m.update(bit, &self.t);
        }
        let tn = &self.tune;
        let lr = tn.lr_lo + tn.lr_hi / (1.0 + self.bits_seen as f32 / tn.lr_t);
        let m2lr = lr * tn.m2_scale;
        let nm = if self.tune.mw != 0 { 7 } else { 6 };
        for m in self.m1[..nm].iter_mut() {
            m.update(&self.x, bit, lr);
        }
        if self.tune.m7 != 0 {
            self.m1[7].update(&self.x, bit, lr);
        }
        self.m2.update(&self.x2, bit, m2lr);
        if self.tune.fm != 0 {
            self.fmx.update(&self.xf, bit, m2lr * 0.5);
            if self.a6on {
                self.a6.update(bit);
            }
        }
        self.a1.update(bit);
        self.a2.update(bit);
        self.a3.update(bit);
        self.a4.update(bit);
        if self.tune.a5 != 0 && self.tune.fm != 0 {
            self.a5.update(bit);
        }

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
        for i in 0..N_CTX {
            let bk = &mut self.tables[i][self.b0[i]];
            if bk.chk != self.b0chk[i] {
                continue;
            }
            if bk.run > 0 && bk.last == b {
                bk.run = bk.run.saturating_add(1);
            } else {
                bk.last = b;
                bk.run = 1;
            }
        }

        let k2 = (self.c8 & 0xffff) as usize;
        self.ind2[k2] = (self.ind2[k2] << 8) | b as u16;
        self.buf.push(b);
        let pos = self.buf.len();
        self.c8 = (self.c8 << 8) | b as u64;
        if b == b'\n' {
            self.prev_line_start = self.line_start;
            self.line_start = pos;
        }
        match b {
            b'[' | b'{' | b'<' | b'(' => {
                if self.brk_n < 16 {
                    self.brk[self.brk_n] = b;
                }
                self.brk_n += 1;
            }
            b']' | b'}' | b'>' | b')' => {
                self.brk_n = self.brk_n.saturating_sub(1);
            }
            b'\n' if self.brk_n > 0 && self.brk_n <= 16 && self.brk[self.brk_n - 1] == b'(' => {
                self.brk_n -= 1;
            }
            _ => {}
        }
        if self.brk_n > 64 {
            self.brk_n = 0;
        }

        // XML tag name: bytes after '<' up to space / '>' / '/'
        if b == b'<' {
            self.in_tag = true;
            self.tag_done = false;
            self.tag = 0;
        } else if self.in_tag && !self.tag_done {
            if b == b' ' || b == b'>' || (b == b'/' && self.tag != 0) {
                self.tag_done = true;
            } else {
                self.tag = hash2(self.tag, b as u64);
            }
        }
        if b == b'>' {
            self.in_tag = false;
        }
        if b.is_ascii_digit() {
            if self.drun == 0 {
                self.dpre = (self.c8 >> 8) & 0xffff;
            }
            self.drun += 1;
        } else {
            self.drun = 0;
        }
        match b {
            b'[' | b'|' | b'=' | b'{' | 10 | b'*' | b'#' | 39 => {
                self.seg = b as u64;
                self.seg_pos = pos;
            }
            _ => {}
        }
        let is_letter = b.is_ascii_alphabetic() || b >= 128;
        if is_letter {
            self.word = hash2(self.word, b.to_ascii_lowercase() as u64 + 1);
            self.wlen += 1;
        } else if self.word != 0 {
            self.wlen = 0;
            self.prevword2 = self.prevword;
            self.prevword = self.word;
            self.word = 0;
        }

        for m in self.mm.iter_mut() {
            m.end_byte(&self.buf);
        }
        self.byte_contexts();
    }
}
