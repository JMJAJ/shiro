use std::io::Write;
use std::time::Instant;
use xzip_core::{compress, decompress, eval, model::ModelConfig, regions};

fn usage() -> ! {
    eprintln!(
        "usage:\n  xzip c <in> <out> [--bits N]\n  xzip d <in> <out>\n  xzip eval <in> [--bits N] [--log results.jsonl] [--note TEXT] [--rev SHA]"
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
    let cfg = ModelConfig { table_bits: bits };
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
            let data = std::fs::read(path).expect("read input");
            let n = data.len();
            let rep = eval(&data, &cfg, |pos, bits| {
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
                    "{{\"ts\":{},\"rev\":{},\"note\":{},\"file\":{},\"in\":{},\"out\":{:.0},\"bpc\":{:.5},\"secs\":{:.1},\"rss_mb\":{:.0},\"cfg\":{{\"bits\":{}}},\"regions\":{{{}}}}}\n",
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
                    regions_json.join(",")
                );
                let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&log).expect("open log");
                f.write_all(line.as_bytes()).expect("write log");
            }
        }
        _ => usage(),
    }
}
