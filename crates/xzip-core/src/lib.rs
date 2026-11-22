pub mod coder;
pub mod dmath;
pub mod model;
pub mod regions;

use coder::{Decoder, Encoder, PBITS};
use model::{Model, ModelConfig};

pub const MAGIC: &[u8; 4] = b"XZP0";

#[derive(Debug)]
pub enum Error {
    BadMagic,
    Truncated,
}

pub fn compress(data: &[u8], cfg: &ModelConfig) -> Vec<u8> {
    let mut m = Model::new(cfg);
    let mut e = Encoder::new();
    for &byte in data {
        for i in (0..8).rev() {
            let bit = ((byte >> i) & 1) as u32;
            let p = m.predict();
            e.encode(bit, p);
            m.update(bit);
        }
    }
    let body = e.finish();
    let mut out = Vec::with_capacity(body.len() + 13);
    out.extend_from_slice(MAGIC);
    out.push(cfg.table_bits as u8);
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    out.extend_from_slice(&body);
    out
}

pub fn decompress(archive: &[u8]) -> Result<Vec<u8>, Error> {
    if archive.len() < 13 {
        return Err(Error::Truncated);
    }
    if &archive[..4] != MAGIC {
        return Err(Error::BadMagic);
    }
    let cfg = ModelConfig { table_bits: archive[4] as u32, ..Default::default() };
    let n = u64::from_le_bytes(archive[5..13].try_into().unwrap()) as usize;
    let mut m = Model::new(&cfg);
    let mut d = Decoder::new(&archive[13..]);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut c = 0u32;
        for _ in 0..8 {
            let p = m.predict();
            let bit = d.decode(p);
            m.update(bit);
            c = (c << 1) | bit;
        }
        out.push(c as u8);
    }
    Ok(out)
}

/// Ideal code length (bits) of each byte under the model, without coding.
/// Exactly the quantity the arithmetic coder realises to within a few bytes.
pub struct EvalReport {
    pub bytes: u64,
    pub total_bits: f64,
    pub region_bits: [f64; regions::N_REGIONS],
    pub region_bytes: [u64; regions::N_REGIONS],
}

/// `side`: optional external per-bit predictions, 8 per byte (P(1)*65536, 0 = none).
pub fn eval(
    data: &[u8],
    cfg: &ModelConfig,
    side: Option<&[u16]>,
    progress: impl FnMut(usize, f64),
) -> EvalReport {
    eval_streams(data, cfg, side, &[], progress)
}

/// Like `eval`, but `switches` = sorted (byte position, stream id): before coding the
/// byte at that position the model switches to that stream's context history.
pub fn eval_streams(
    data: &[u8],
    cfg: &ModelConfig,
    side: Option<&[u16]>,
    switches: &[(usize, usize)],
    mut progress: impl FnMut(usize, f64),
) -> EvalReport {
    let mut m = Model::new(cfg);
    let mut sw = switches.iter().peekable();
    let mut rc = regions::Classifier::new();
    let mut rep = EvalReport {
        bytes: data.len() as u64,
        total_bits: 0.0,
        region_bits: [0.0; regions::N_REGIONS],
        region_bytes: [0; regions::N_REGIONS],
    };
    let scale = 1.0 / (1u64 << PBITS) as f64;
    for (pos, &byte) in data.iter().enumerate() {
        while let Some(&&(p, k)) = sw.peek() {
            if p > pos {
                break;
            }
            m.switch_stream(k);
            sw.next();
        }
        let r = rc.region() as usize;
        let mut cost = 0.0f64;
        for i in (0..8).rev() {
            let bit = ((byte >> i) & 1) as u32;
            if let Some(sd) = side {
                m.set_side(sd[pos * 8 + (7 - i)]);
            }
            let p1 = m.predict() as f64 * scale;
            let p = if bit != 0 { p1 } else { 1.0 - p1 };
            cost -= p.log2();
            m.update(bit);
        }
        rc.push(byte);
        rep.total_bits += cost;
        rep.region_bits[r] += cost;
        rep.region_bytes[r] += 1;
        if pos & 0xF_FFFF == 0xF_FFFF {
            progress(pos + 1, rep.total_bits);
        }
    }
    rep
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<u8> {
        let mut v = Vec::new();
        let words = ["the ", "quick ", "brown ", "fox ", "<page>", "[[link]] ", "{{tmpl}}", "\n"];
        let mut s = 7u64;
        for _ in 0..30_000 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            v.extend_from_slice(words[(s >> 60) as usize % words.len()].as_bytes());
            if (s >> 50) & 31 == 0 {
                v.push((s >> 20) as u8);
            }
        }
        v
    }

    #[test]
    fn roundtrip_text_and_edge_cases() {
        let cfg = ModelConfig { table_bits: 14, ..Default::default() };
        for data in [Vec::new(), vec![0u8], vec![255u8; 1000], sample()] {
            let c = compress(&data, &cfg);
            let d = decompress(&c).unwrap();
            assert_eq!(d, data);
        }
    }

    #[test]
    fn eval_matches_real_size() {
        let cfg = ModelConfig { table_bits: 14, ..Default::default() };
        let data = sample();
        let real = compress(&data, &cfg).len() as f64 - 13.0;
        let ideal = eval(&data, &cfg, None, |_, _| {}).total_bits / 8.0;
        assert!((real - ideal).abs() < 16.0, "real {real} ideal {ideal}");
    }
}
