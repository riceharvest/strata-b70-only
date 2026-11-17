//! Drawing JSON results and progress events in the terminal: aligned keys, tables for lists of records, sizes and
//! times in words, colour only on a terminal.

use std::io::{self, IsTerminal, Write};

use serde_json::{Map, Value};

use crate::fmt;

static PLAIN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static WIDTH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn color() -> bool {
    !PLAIN.load(std::sync::atomic::Ordering::Relaxed) && std::env::var_os("NO_COLOR").is_none() && io::stdout().is_terminal()
}

/// The Python command line names itself `strata` in its messages; here the commands are intelinference's.
pub fn rename(t: &str) -> String {
    if !t.contains("strata ") {
        return t.to_string();
    }
    let mut out = t.replace("strata logs", "intelinference activity");
    for (path, _, _) in crate::bridge::TREE {
        let top = path.split(' ').next().unwrap_or(path);
        out = out.replace(&format!("strata {top}"), &format!("intelinference {top}"));
    }
    for native in ["status", "monitor", "models"] {
        out = out.replace(&format!("strata {native}"), &format!("intelinference {native}"));
    }
    out
}

/// The result as plain lines (no colour), for the live screen.
pub fn plain_lines(title: &str, v: &Value, width: usize) -> Vec<String> {
    PLAIN.store(true, std::sync::atomic::Ordering::Relaxed);
    WIDTH.store(width, std::sync::atomic::Ordering::Relaxed);
    let mut out = vec![];
    if v["ok"].as_bool() == Some(false) {
        let msg = v["error"]["message"].as_str().or_else(|| v["error"].as_str()).unwrap_or("failed");
        out.push(format!("✗ {}", rename(msg)));
    } else {
        out.push(format!("✓ {title}"));
    }
    if let Some(o) = v.as_object() {
        let mut o = o.clone();
        o.remove("error");
        object(&o, 2, &mut out);
    }
    PLAIN.store(false, std::sync::atomic::Ordering::Relaxed);
    WIDTH.store(0, std::sync::atomic::Ordering::Relaxed);
    out.into_iter().map(|l| l.trim_end().to_string()).collect()
}

fn paint(code: &str, t: &str) -> String {
    if color() { format!("\x1b[{code}m{t}\x1b[0m") } else { t.to_string() }
}
pub fn bold(t: &str) -> String {
    paint("1", t)
}
pub fn dim(t: &str) -> String {
    paint("2", t)
}
pub fn cyan(t: &str) -> String {
    paint("38;2;0;199;253", t)
}
pub fn green(t: &str) -> String {
    paint("32", t)
}
pub fn red(t: &str) -> String {
    paint("31", t)
}
pub fn amber(t: &str) -> String {
    paint("38;2;255;186;72", t)
}

fn width(t: &str) -> usize {
    // visible width: ANSI codes do not count
    let mut n = 0;
    let mut esc = false;
    for c in t.chars() {
        if esc {
            if c == 'm' {
                esc = false;
            }
        } else if c == '\x1b' {
            esc = true;
        } else {
            n += 1;
        }
    }
    n
}

fn pad(t: &str, w: usize) -> String {
    format!("{t}{}", " ".repeat(w.saturating_sub(width(t))))
}

fn words(key: &str) -> String {
    // environment variable names stay as they are
    if key.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_') {
        return key.to_string();
    }
    key.replace('_', " ")
}

/// A scalar in words, by what its key says it is.
pub fn scalar(key: &str, v: &Value) -> String {
    let k = key.to_ascii_lowercase();
    match v {
        Value::Null => dim("—"),
        Value::Bool(true) => green("yes"),
        Value::Bool(false) => dim("no"),
        Value::Number(n) => {
            let x = n.as_f64().unwrap_or(0.0);
            if k.ends_with("bytes") || k == "size" || k.ends_with("_size") {
                if x >= (1u64 << 30) as f64 {
                    format!("{:.2} GB", x / 1e9)
                } else {
                    format!("{:.1} MB", x / 1e6)
                }
            } else if k.ends_with("_mib") {
                fmt::mib(x)
            } else if k.ends_with("_gb") || k == "gb" {
                format!("{x:.1} GB")
            } else if (k.ends_with("_s") || k.ends_with("_seconds")) && !k.ends_with("_tok_s") && !k.ends_with("per_s") {
                fmt::secs(x)
            } else if k.ends_with("_ms") {
                fmt::secs(x / 1000.0)
            } else if (k == "at" || k.ends_with("_at") || k == "time" || k == "started" || k == "since") && x > 1.5e9 {
                fmt::clock(x)
            } else if n.is_f64() {
                if x.abs() >= 100.0 { format!("{x:.0}") } else { format!("{x:.2}").trim_end_matches('0').trim_end_matches('.').to_string() }
            } else {
                n.to_string()
            }
        }
        Value::String(s) => {
            if k == "status" || k == "outcome" || k == "state" || k == "verdict" || k == "severity" {
                match s.as_str() {
                    "ok" | "passed" | "pass" | "ready" | "installed" | "applied" | "current" | "up-to-date" | "valid" | "running" | "accepted" | "keep" => green(s),
                    "error" | "failed" | "fail" | "missing" | "broken" | "invalid" | "rejected" | "stuck" => red(s),
                    "warning" | "warn" | "optional_missing" | "install" | "available" | "update" | "stale" | "pending" => amber(s),
                    _ => s.clone(),
                }
            } else if s.is_empty() {
                dim("—")
            } else {
                rename(s)
            }
        }
        Value::Array(a) if a.is_empty() => dim("none"),
        Value::Array(a) => a.iter().map(|x| strip(&scalar(key, x))).collect::<Vec<_>>().join(if a.iter().any(|x| x.as_str().is_some_and(|s| s.len() > 40)) { "\n" } else { ", " }),
        Value::Object(_) => v.to_string(),
    }
}

fn strip(t: &str) -> String {
    let mut out = String::new();
    let mut esc = false;
    for c in t.chars() {
        if esc {
            if c == 'm' {
                esc = false;
            }
        } else if c == '\x1b' {
            esc = true;
        } else {
            out.push(c);
        }
    }
    out
}

fn is_scalar(v: &Value) -> bool {
    !v.is_object() && !(v.is_array() && v.as_array().is_some_and(|a| a.iter().any(|x| x.is_object() || x.is_array())))
}

/// Keys that hold prose: printed as a paragraph, first.
const PROSE: &[&str] = &["text", "message", "line", "summary", "note", "reason", "hint", "next", "advice"];
/// Keys left out: envelope and noise.
const SKIP: &[&str] = &["schema_version", "ok", "event"];

fn wrap(text: &str, indent: usize, w: usize) -> Vec<String> {
    let w = w.saturating_sub(indent).max(30);
    let mut out = vec![];
    for para in text.split('\n') {
        let mut line = String::new();
        for word in para.split(' ') {
            if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > w {
                out.push(std::mem::take(&mut line));
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
        out.push(line);
    }
    out
}

fn term_width() -> usize {
    let w = WIDTH.load(std::sync::atomic::Ordering::Relaxed);
    if w > 0 {
        return w;
    }
    let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
    if unsafe { libc::ioctl(1, libc::TIOCGWINSZ, &mut ws) } == 0 && ws.ws_col > 20 { ws.ws_col as usize } else { 100 }
}

/// A list of records as a table: the scalar columns that fit.
fn table(rows: &[Value], indent: usize, out: &mut Vec<String>) {
    let mut cols: Vec<String> = vec![];
    for r in rows {
        if let Some(o) = r.as_object() {
            for (k, v) in o {
                if is_scalar(v) && k != "schema_version" && !cols.contains(k) {
                    cols.push(k.clone());
                }
            }
        }
    }
    let cells: Vec<Vec<String>> = rows
        .iter()
        .map(|r| cols.iter().map(|c| { let s = scalar(c, &r[c]); if strip(&s).chars().count() > 48 { format!("{}…", strip(&s).chars().take(47).collect::<String>()) } else { s } }).collect())
        .collect();
    let mut widths: Vec<usize> = cols.iter().map(|c| words(c).len()).collect();
    for row in &cells {
        for (i, c) in row.iter().enumerate() {
            widths[i] = widths[i].max(width(c));
        }
    }
    // drop columns from the right until it fits
    let tw = term_width();
    let mut keep = cols.len();
    while keep > 1 && indent + widths[..keep].iter().map(|w| w + 2).sum::<usize>() > tw {
        keep -= 1;
    }
    let ind = " ".repeat(indent);
    out.push(format!("{ind}{}", cols[..keep].iter().enumerate().map(|(i, c)| pad(&dim(&words(c)), widths[i])).collect::<Vec<_>>().join("  ")));
    for row in &cells {
        out.push(format!("{ind}{}", row[..keep].iter().enumerate().map(|(i, c)| pad(c, widths[i])).collect::<Vec<_>>().join("  ")));
    }
    if keep < cols.len() {
        out.push(format!("{ind}{}", dim(&format!("(+{} more columns: --json shows all)", cols.len() - keep))));
    }
}

fn object(o: &Map<String, Value>, indent: usize, out: &mut Vec<String>) {
    let ind = " ".repeat(indent);
    let tw = term_width();
    for k in PROSE {
        if let Some(Value::String(s)) = o.get(*k)
            && !s.is_empty()
            && !s.contains('\n')
        {
            for l in wrap(&rename(s), indent, tw) {
                out.push(format!("{ind}{l}"));
            }
        }
    }
    let keyw = o.keys().filter(|k| is_scalar(&o[*k])).map(|k| words(k).len()).max().unwrap_or(0).min(26);
    for (k, v) in o {
        if SKIP.contains(&k.as_str()) || (PROSE.contains(&k.as_str()) && v.is_string()) || !is_scalar(v) {
            continue;
        }
        if v.as_str().is_some_and(|t| t.contains('\n')) {
            continue; // a block of text: below, as it is
        }
        let val = scalar(k, v);
        let first = indent + keyw + 2;
        let lines = if (width(&val) + first > tw || val.contains('\n')) && (v.is_string() || v.is_array()) { wrap(&val, first, tw) } else { vec![val] };
        for (i, l) in lines.iter().enumerate() {
            if i == 0 {
                out.push(format!("{ind}{}  {l}", pad(&dim(&words(k)), keyw)));
            } else {
                out.push(format!("{ind}{}  {l}", " ".repeat(keyw)));
            }
        }
    }
    // blocks of text (file contents, logs): verbatim, indentation kept
    for (k, v) in o {
        if let Some(t) = v.as_str().filter(|t| t.contains('\n')) {
            out.push(format!("{ind}{}", dim(&words(k))));
            for l in t.trim_end().lines() {
                out.push(format!("{ind}  {}", paint("38;2;170;235;255", l)));
            }
        }
    }
    for (k, v) in o {
        if SKIP.contains(&k.as_str()) || is_scalar(v) {
            continue;
        }
        match v {
            Value::Object(inner) if inner.is_empty() => {}
            Value::Object(inner) => {
                out.push(String::new());
                out.push(format!("{ind}{}", bold(&words(k))));
                object(inner, indent + 2, out);
            }
            Value::Array(a) => {
                out.push(String::new());
                out.push(format!("{ind}{} {}", bold(&words(k)), dim(&format!("({})", a.len()))));
                if a.iter().all(Value::is_object) {
                    // a table when the records are flat and no value runs over several lines
                    let flat = a.iter().all(|r| {
                        r.as_object().is_some_and(|o| o.values().filter(|v| !is_scalar(v)).count() <= 1 && !o.values().any(|v| v.as_str().is_some_and(|s| s.contains('\n'))))
                    });
                    if flat && a.len() > 1 {
                        table(a, indent + 2, out);
                    } else {
                        for (i, r) in a.iter().enumerate() {
                            if i > 0 {
                                out.push(format!("{ind}  {}", dim("·")));
                            }
                            object(r.as_object().expect("object"), indent + 2, out);
                        }
                    }
                } else {
                    for x in a {
                        out.push(format!("{ind}  • {}", if is_scalar(x) { scalar(k, x) } else { x.to_string() }));
                    }
                }
            }
            _ => {}
        }
    }
}

/// Draw a command's result; returns the exit code it implies (1 when `ok` is false).
pub fn result(title: &str, v: &Value) -> i32 {
    let ok = v["ok"].as_bool().unwrap_or(true);
    let mut out = vec![];
    if !ok {
        let msg = v["error"]["message"].as_str().or_else(|| v["error"].as_str()).or_else(|| v["message"].as_str()).unwrap_or("failed");
        eprintln!("{} {}", red("✗"), rename(msg));
        let code = v["error"]["code"].as_i64().unwrap_or(1) as i32;
        if let Some(o) = v.as_object() {
            let mut rest = o.clone();
            rest.remove("error");
            let mut lines = vec![];
            object(&rest, 2, &mut lines);
            for l in lines {
                eprintln!("{l}");
            }
        }
        return code.max(1);
    }
    if let Some(o) = v.as_object() {
        // the one interesting list or object alone: no envelope keys around it
        let payload: Vec<(&String, &Value)> = o.iter().filter(|(k, _)| !SKIP.contains(&k.as_str())).collect();
        if payload.len() == 1
            && let Value::Array(a) = payload[0].1
        {
            if a.is_empty() {
                println!("{} {}", dim("○"), dim(&format!("no {}", words(payload[0].0))));
                return 0;
            }
            println!("{} {}", cyan("◆"), bold(&format!("{} {}", words(payload[0].0), dim(&format!("({})", a.len())))));
        } else {
            println!("{} {}", green("✓"), bold(title));
        }
        object(o, 2, &mut out);
    } else {
        out.push(v.to_string());
    }
    let mut so = io::stdout().lock();
    for l in out {
        writeln!(so, "{l}").ok();
    }
    0
}

// ---- progress events ----

thread_local! {
    static OPEN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn bar(frac: f64, w: usize) -> String {
    let full = (frac.clamp(0.0, 1.0) * w as f64 * 8.0) as usize;
    let parts = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"];
    let mut s = "█".repeat(full / 8);
    if full / 8 < w {
        s.push_str(parts[full % 8]);
    }
    let n = s.chars().count();
    format!("{}{}", cyan(&s), dim(&"·".repeat(w.saturating_sub(n))))
}

/// One streamed event on stderr: progress redraws one line, everything else is a line of its own.
pub fn event_line(ev: &Value) {
    let tty = io::stderr().is_terminal();
    let kind = ev["event"].as_str().unwrap_or("");
    let mut e = io::stderr().lock();
    if kind == "progress" {
        let done = ev["done"].as_f64().unwrap_or(0.0);
        let total = ev["total"].as_f64().unwrap_or(0.0);
        let frac = ev["percent"].as_f64().map(|p| p / 100.0).unwrap_or(if total > 0.0 { done / total } else { 0.0 });
        let label = ev["label"].as_str().or(ev["step"].as_str()).unwrap_or("working");
        let amount = if total > 1e6 { format!("{:.0} / {:.0} MB", done / 1e6, total / 1e6) } else if total > 0.0 { format!("{done:.0} / {total:.0}") } else { String::new() };
        if tty {
            write!(e, "\r\x1b[2K  {} {} {:>3.0}%  {}", bar(frac, 28), label, frac * 100.0, dim(&amount)).ok();
            OPEN.with(|o| o.set(true));
        } else if (frac * 100.0) as i64 % 25 == 0 {
            writeln!(e, "  {label} {:.0}%", frac * 100.0).ok();
        }
        e.flush().ok();
        return;
    }
    if OPEN.with(|o| o.replace(false)) {
        writeln!(e).ok();
    }
    let msg = ev["message"].as_str().map(rename).unwrap_or_else(|| {
        // a record (activity): its scalar fields in one line
        ev.as_object()
            .map(|o| o.iter().filter(|(k, v)| !SKIP.contains(&k.as_str()) && is_scalar(v)).map(|(k, v)| format!("{} {}", dim(&words(k)), strip(&scalar(k, v)))).collect::<Vec<_>>().join("  "))
            .unwrap_or_default()
    });
    let mark = match kind {
        "phase" => cyan("›"),
        "waiting" => amber("…"),
        "failed" | "error" => red("✗"),
        "text" => " ".into(),
        _ => dim("·"),
    };
    writeln!(e, "{mark} {msg}").ok();
}

pub fn finish_progress() {
    if OPEN.with(|o| o.replace(false)) {
        eprintln!();
    }
}
