//! Long commands from the live screen (setup, updates, downloads, benchmarks, tuning): run in the background, their
//! progress and output streamed into a panel, the result drawn when they end. One at a time.

use std::{
    io::{BufReader, Read},
    process::Stdio,
    sync::mpsc::Sender,
    thread,
    time::Instant,
};

use serde_json::Value;

use super::Msg;
use crate::{
    bridge::{self, Out},
    render,
    settings::Settings,
};

#[derive(Clone, Debug)]
pub struct Spec {
    pub title: String,
    pub path: Vec<String>,
    pub args: Vec<String>,
    pub out: Out,
}

impl Spec {
    pub fn new(title: &str, path: &[&str], args: &[&str], out: Out) -> Self {
        Self { title: title.into(), path: path.iter().map(|s| s.to_string()).collect(), args: args.iter().map(|s| s.to_string()).collect(), out }
    }

    /// The same command for a terminal.
    pub fn command_line(&self) -> String {
        let mut p = self.path.clone();
        if p.first().map(String::as_str) == Some("logs") {
            p[0] = "activity".into();
        }
        format!("intelinference {} {}", p.join(" "), self.args.join(" ")).trim().to_string()
    }
}

pub struct Job {
    pub spec: Spec,
    pub lines: Vec<String>,
    pub progress: Option<(f64, String)>,
    pub result: Option<Vec<String>>,
    pub code: Option<i32>,
    pub started: Instant,
    pub ended: Option<Instant>,
    pub pid: Option<u32>,
}

impl Job {
    pub fn new(spec: Spec) -> Self {
        Self { spec, lines: vec![], progress: None, result: None, code: None, started: Instant::now(), ended: None, pid: None }
    }
    pub fn running(&self) -> bool {
        self.code.is_none()
    }
    pub fn push(&mut self, text: String, replace: bool) {
        if replace && !self.lines.is_empty() {
            *self.lines.last_mut().expect("non-empty") = text;
        } else {
            self.lines.push(text);
        }
        if self.lines.len() > 2000 {
            self.lines.drain(..500);
        }
    }
}

/// Text from a pipe, split at newlines; a segment ended by `\r` replaces the line before it (download progress).
fn pump(r: impl Read, mut each: impl FnMut(String, bool)) {
    let mut r = BufReader::new(r);
    let mut buf = Vec::new();
    let mut byte = [0u8; 4096];
    let mut replace_next = false;
    loop {
        let n = match r.read(&mut byte) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        for &b in &byte[..n] {
            if b == b'\n' || b == b'\r' {
                let line = String::from_utf8_lossy(&buf).trim_end().to_string();
                buf.clear();
                if !line.is_empty() {
                    each(line, replace_next);
                }
                replace_next = b == b'\r';
            } else {
                buf.push(b);
            }
        }
    }
    if !buf.is_empty() {
        each(String::from_utf8_lossy(&buf).to_string(), replace_next);
    }
}

fn strip_ansi(t: &str) -> String {
    let mut out = String::new();
    let mut esc = false;
    for c in t.chars() {
        if esc {
            if c.is_ascii_alphabetic() {
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

pub fn start(s: &Settings, spec: Spec, width: usize, tx: Sender<Msg>) {
    let mut c = match bridge::base(s) {
        Ok(c) => c,
        Err(e) => {
            tx.send(Msg::JobLine { text: format!("{e:#}"), replace: false }).ok();
            tx.send(Msg::JobDone { code: 1, result: None }).ok();
            return;
        }
    };
    c.args(&spec.path).args(&spec.args);
    match spec.out {
        Out::Jsonl => {
            c.args(["--output", "jsonl"]);
        }
        Out::Json => {
            c.args(["--output", "json"]);
        }
        Out::Tty => {}
    }
    c.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).env("NO_COLOR", "1").env("PYTHONUNBUFFERED", "1");
    let mut child = match c.spawn() {
        Ok(ch) => ch,
        Err(e) => {
            tx.send(Msg::JobLine { text: format!("cannot run {}: {e}", s.python), replace: false }).ok();
            tx.send(Msg::JobDone { code: 1, result: None }).ok();
            return;
        }
    };
    tx.send(Msg::JobPid(child.id())).ok();
    let err = child.stderr.take().expect("piped");
    let out = child.stdout.take().expect("piped");
    let tx_err = tx.clone();
    let err_thread = thread::spawn(move || {
        pump(err, |line, replace| {
            tx_err.send(Msg::JobLine { text: render::rename(&strip_ansi(&line)), replace }).ok();
        })
    });
    let title = spec.title.clone();
    thread::spawn(move || {
        let mut result: Option<Value> = None;
        let mut all = vec![];
        pump(out, |line, replace| {
            all.push(line.clone());
            match serde_json::from_str::<Value>(line.trim()) {
                Ok(v) if v.get("event").is_some() && v.get("ok").is_none() => {
                    if v["event"].as_str() == Some("progress") {
                        let done = v["done"].as_f64().unwrap_or(0.0);
                        let total = v["total"].as_f64().unwrap_or(0.0);
                        let frac = v["percent"].as_f64().map(|p| p / 100.0).unwrap_or(if total > 0.0 { done / total } else { 0.0 });
                        let label = v["label"].as_str().or(v["step"].as_str()).unwrap_or("working").to_string();
                        tx.send(Msg::JobProgress { frac, label }).ok();
                    } else {
                        let text = v["message"].as_str().map(render::rename).unwrap_or_else(|| v.to_string());
                        tx.send(Msg::JobLine { text: format!("› {text}"), replace: false }).ok();
                    }
                }
                Ok(v) if v.is_object() => result = Some(v),
                _ => {
                    tx.send(Msg::JobLine { text: render::rename(&strip_ansi(&line)), replace }).ok();
                }
            }
        });
        err_thread.join().ok();
        let code = child.wait().ok().and_then(|s| s.code()).unwrap_or(1);
        if result.is_none() {
            result = serde_json::from_str::<Value>(&all.join("\n")).ok().filter(Value::is_object);
        }
        let lines = result.as_ref().map(|v| render::plain_lines(&title, v, width));
        tx.send(Msg::JobDone { code, result: lines }).ok();
    });
}

/// Ask a running job to stop (SIGINT first, as Ctrl+C in a terminal would).
pub fn interrupt(pid: u32) {
    unsafe { libc::kill(pid as i32, libc::SIGINT) };
}
