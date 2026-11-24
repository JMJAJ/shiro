use std::io::Write;
use std::time::Instant;
use shiro_core::{compress, decompress, eval_streams, model::ModelConfig, regions};

fn usage() -> ! {
    eprintln!(
        "usage:\n  shiro c <in> <out> [--bits N]\n  shiro d <in> <out>\n  shiro eval <in> [--bits N] [--log results.jsonl] [--note TEXT] [--rev SHA] [--side FILE] [--order FILE [--streams B]]"
    );
    std::process::exit(2)
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn peak_rss_mb() -> f64 {
    // Linux: VmHWM from /proc/self/status. Elsewhere: unknown (0).
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find(|l| l.starts_with("VmHWM:")).and_then(|l| {
                l.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok())
            })
        })
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        usage();
    }
    let bits = flag(&args, "--bits").map(|v| v.parse().expect("--bits")).unwrap_or(ModelConfig::default().table_bits);
    let mut cfg = ModelConfig { table_bits: bits, ..Default::default() };
    let tune_spec = std::env::var("SHIRO_TUNE").unwrap_or_default();
    if args[1] == "eval" && !tune_spec.is_empty() {
        cfg.tune = shiro_core::model::Tune::parse(&tune_spec).unwrap_or_else(|e| {
            eprintln!("SHIRO_TUNE: {e}");
            std::process::exit(2)
        });
    }
    let t0 = Instant::now();
    match args[1].as_str() {
        "c" => {
            if args.len() < 4 {
                usage();
            }
            let data = std::fs::read(&args[2]).expect("read input");
            let out = compress(&data, &cfg);
            std::fs::write(&args[3], &out).expect("write output");
            let s = t0.elapsed().as_secs_f64();
            eprintln!(
                "{} -> {} bytes  ({:.4} bpc)  {:.1}s  {:.0} ns/B  rss {:.0} MB",
                data.len(),
                out.len(),
                out.len() as f64 * 8.0 / data.len().max(1) as f64,
                s,
                s * 1e9 / data.len().max(1) as f64,
                peak_rss_mb()
            );
        }
        "d" => {
            if args.len() < 4 {
                usage();
            }
            let arc = std::fs::read(&args[2]).expect("read input");
            let out = decompress(&arc).expect("corrupt archive");
            std::fs::write(&args[3], &out).expect("write output");
            let s = t0.elapsed().as_secs_f64();
            eprintln!("{} -> {} bytes  {:.1}s  rss {:.0} MB", arc.len(), out.len(), s, peak_rss_mb());
        }
        "eval" => {
            let path = &args[2];
            let mut data = std::fs::read(path).expect("read input");
            let n = data.len();
            let mut side: Option<Vec<u16>> = flag(&args, "--side").map(|sp| {
                let raw = std::fs::read(&sp).expect("read side file");
                assert_eq!(raw.len(), n * 16, "side file must hold 8 u16 per input byte");
                raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
            });
            // --order FILE: u64 LE (start, end) byte ranges, in the order a decoder would
            // see them (e.g. the NN's block-interleaved stream schedule). Must cover every
            // byte exactly once.
            let mut switches: Vec<(usize, usize)> = Vec::new();
            if let Some(op) = flag(&args, "--order") {
                let raw = std::fs::read(&op).expect("read order file");
                let nstreams: usize = flag(&args, "--streams").map(|v| v.parse().expect("--streams N")).unwrap_or(0);
                let rs: Vec<(usize, usize)> = raw
                    .chunks_exact(16)
                    .map(|c| {
                        (
                            u64::from_le_bytes(c[..8].try_into().unwrap()) as usize,
                            u64::from_le_bytes(c[8..].try_into().unwrap()) as usize,
                        )
                    })
                    .collect();
                // stream b's bytes are one contiguous span; its first range is in step 0
                let mut starts: Vec<usize> = rs.iter().take(nstreams).map(|r| r.0).collect();
                starts.sort_unstable();
                let mut q = 0usize;
                for &(a, b) in &rs {
                    if nstreams > 0 && a < b {
                        let k = starts.partition_point(|&s0| s0 <= a) - 1;
                        switches.push((q, k));
                    }
                    q += b - a;
                }
                let mut seen = vec![false; n];
                let mut nd = Vec::with_capacity(n);
                let mut ns = side.as_ref().map(|_| Vec::with_capacity(n * 8));
                for c in raw.chunks_exact(16) {
                    let a = u64::from_le_bytes(c[..8].try_into().unwrap()) as usize;
                    let b = u64::from_le_bytes(c[8..].try_into().unwrap()) as usize;
                    assert!(a <= b && b <= n, "bad range {a}..{b}");
                    for (i, s) in seen[a..b].iter_mut().enumerate() {
                        assert!(!*s, "byte {} ordered twice", a + i);
                        *s = true;
                    }
                    nd.extend_from_slice(&data[a..b]);
                    if let (Some(v), Some(sd)) = (ns.as_mut(), side.as_ref()) {
                        v.extend_from_slice(&sd[a * 8..b * 8]);
                    }
                }
                assert!(seen.iter().all(|&s| s), "order file does not cover every byte");
                eprintln!("order: {} ranges", raw.len() / 16);
                data = nd;
                if ns.is_some() {
                    side = ns;
                }
            }
            let rep = eval_streams(&data, &cfg, side.as_deref(), &switches, |pos, bits| {
                let el = t0.elapsed().as_secs_f64();
                eprint!(
                    "\r  {:>6.1}%  {:.4} bpc  {:.0} ns/B   ",
                    pos as f64 * 100.0 / n as f64,
                    bits / pos as f64,
                    el * 1e9 / pos as f64
                );
            });
            eprintln!();
            let secs = t0.elapsed().as_secs_f64();
            let bytes = rep.total_bits / 8.0;
            let mut regions_json = Vec::new();
            println!("file      {path}");
            println!("bytes     {:.0}   ({:.5} bpc)", bytes, rep.total_bits / n.max(1) as f64);
            println!("time      {:.1}s  ({:.0} ns/B)   rss {:.0} MB", secs, secs * 1e9 / n.max(1) as f64, peak_rss_mb());
            println!("region       in_bytes   out_bytes    bpc   share");
            for i in 0..regions::N_REGIONS {
                let ib = rep.region_bytes[i];
                let ob = rep.region_bits[i] / 8.0;
                println!(
                    "{:<10} {:>10} {:>11.0} {:>6.3} {:>6.1}%",
                    regions::NAMES[i],
                    ib,
                    ob,
                    rep.region_bits[i] / ib.max(1) as f64,
                    ob * 100.0 / bytes.max(1.0)
                );
                regions_json.push(format!("{}:[{},{:.0}]", json_str(regions::NAMES[i]), ib, ob));
            }
            if let Some(log) = flag(&args, "--log") {
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                let line = format!(
                    "{{\"ts\":{},\"rev\":{},\"note\":{},\"file\":{},\"in\":{},\"out\":{:.0},\"bpc\":{:.5},\"secs\":{:.1},\"rss_mb\":{:.0},\"cfg\":{{\"bits\":{},\"tune\":{}}},\"regions\":{{{}}}}}\n",
                    ts,
                    json_str(&flag(&args, "--rev").unwrap_or_default()),
                    json_str(&flag(&args, "--note").unwrap_or_default()),
                    json_str(path),
                    n,
                    bytes,
                    rep.total_bits / n.max(1) as f64,
                    secs,
                    peak_rss_mb(),
                    bits,
                    json_str(&tune_spec),
                    regions_json.join(",")
                );
                let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&log).expect("open log");
                f.write_all(line.as_bytes()).expect("write log");
            }
        }
        _ => usage(),
    }
}
