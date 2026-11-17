//! Numbers in words people read at a glance, and the /metrics reply as one flat snapshot.

use serde_json::Value;

use crate::api::{f, s};

pub fn tokens(n: f64) -> String {
    let a = n.abs();
    if a >= 1e9 {
        format!("{:.2}B", n / 1e9)
    } else if a >= 1e6 {
        format!("{:.1}M", n / 1e6)
    } else if a >= 10e3 {
        format!("{:.0}K", n / 1e3)
    } else if a >= 1e3 {
        format!("{:.1}K", n / 1e3)
    } else {
        format!("{n:.0}")
    }
}

pub fn gib(bytes: f64) -> String {
    format!("{:.1} GiB", bytes / (1u64 << 30) as f64)
}

pub fn mib(m: f64) -> String {
    if m >= 1024.0 { format!("{:.1} GiB", m / 1024.0) } else { format!("{m:.0} MiB") }
}

pub fn secs(t: f64) -> String {
    if t < 1.0 {
        format!("{:.0} ms", t * 1000.0)
    } else if t < 60.0 {
        format!("{t:.1} s")
    } else if t < 3600.0 {
        format!("{}m{:02}s", (t / 60.0) as u64, (t % 60.0) as u64)
    } else {
        format!("{}h{:02}m", (t / 3600.0) as u64, ((t % 3600.0) / 60.0) as u64)
    }
}

pub fn pct(part: f64, whole: f64) -> Option<f64> {
    (whole > 0.0).then(|| 100.0 * part / whole)
}

/// Local clock time of a unix timestamp, HH:MM:SS.
pub fn clock(ts: f64) -> String {
    let t = ts as libc::time_t;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    unsafe { libc::localtime_r(&t, &mut tm) };
    format!("{:02}:{:02}:{:02}", tm.tm_hour, tm.tm_min, tm.tm_sec)
}

#[derive(Debug, Clone, Default)]
pub struct Request {
    pub time: f64,
    pub prompt: f64,
    pub reused: f64,
    pub read: f64,
    pub first_token_s: Option<f64>,
    pub output: f64,
    pub decode_tok_s: Option<f64>,
    pub prompt_ms: Option<f64>,
    pub drafts_offered: f64,
    pub drafts_accepted: f64,
    pub finish: String,
    pub duration_s: f64,
}

#[derive(Debug, Clone, Default)]
pub struct Thread {
    pub key: String,
    pub state: String,
    pub tokens: f64,
    pub turns: f64,
    pub pinned: bool,
}

#[derive(Debug, Clone, Default)]
pub struct Day {
    pub date: String,
    pub output: f64,
}

/// /metrics, flattened. Missing keys stay None / 0 / empty.
#[derive(Debug, Clone, Default)]
pub struct Snap {
    pub model: String,
    pub version: String,
    pub context: Option<f64>,
    pub kv: Option<String>,
    pub drafting: Option<String>,
    pub spec: Option<f64>,
    pub vram_free_mib: Option<f64>,
    pub expert_slots: Option<f64>,

    pub state: String,
    pub phase: Option<String>,
    pub queued: f64,
    pub prompt_tokens: Option<f64>,
    pub prompt_read: Option<f64>,
    pub prompt_total: Option<f64>,
    pub generated: Option<f64>,
    pub max_tokens: Option<f64>,
    pub elapsed_s: Option<f64>,
    pub tok_s: Option<f64>,
    pub parallel: Option<f64>,
    pub running: Option<f64>,

    pub cpu: Option<f64>,
    pub ram_used: Option<f64>,
    pub ram_total: Option<f64>,
    pub gpu_name: Option<String>,
    pub hist_tok_s: Vec<f64>,
    pub hist_cpu: Vec<f64>,

    pub requests: Vec<Request>,
    pub t_requests: f64,
    pub t_prompt: f64,
    pub t_reused: f64,
    pub t_output: f64,
    pub t_prompt_ms: f64,
    pub t_decode_ms: f64,
    pub t_offered: f64,
    pub t_accepted: f64,
    pub t_since: Option<f64>,

    pub life_requests: f64,
    pub life_output: f64,
    pub days: Vec<Day>,

    pub cc_parks: f64,
    pub cc_restores: f64,
    pub cc_ssd_files: Option<f64>,
    pub cc_ssd_gib: Option<f64>,
    pub cc_ssd_restores: Option<f64>,
    pub threads: Vec<Thread>,
    pub return_gap_p50: Option<f64>,
    pub time: f64,
}

fn series(v: &Value) -> Vec<f64> {
    v.as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect()).unwrap_or_default()
}

impl Snap {
    pub fn from(m: &Value) -> Self {
        let e = &m["engine"];
        let l = &m["live"];
        let h = &m["hardware"];
        let t = &m["totals"];
        let lt = &m["lifetime"];
        let cc = &m["conversation_cache"];
        let mut days: Vec<Day> = lt["days"]
            .as_object()
            .map(|o| {
                o.iter()
                    .map(|(k, d)| Day { date: k.clone(), output: f(&d["output_tokens"]).unwrap_or(0.0) })
                    .collect()
            })
            .unwrap_or_default();
        days.sort_by(|a, b| a.date.cmp(&b.date));
        Snap {
            model: s(&e["model"]).unwrap_or("?").to_string(),
            version: s(&e["engine"]).or(s(&e["version"])).unwrap_or("").to_string(),
            context: f(&e["context"]).or(f(&e["max_context"])),
            kv: s(&e["kv"]).map(str::to_string),
            drafting: s(&e["drafting"]).map(str::to_string),
            spec: f(&e["spec"]),
            vram_free_mib: f(&e["vram_free_mib"]),
            expert_slots: f(&e["expert_slots"]),
            state: s(&l["state"]).unwrap_or("?").to_string(),
            phase: s(&l["phase"]).map(str::to_string),
            queued: f(&l["queued"]).unwrap_or(0.0),
            prompt_tokens: f(&l["prompt_tokens"]),
            prompt_read: f(&l["prompt_read"]),
            prompt_total: f(&l["prompt_total"]),
            generated: f(&l["generated"]),
            max_tokens: f(&l["max_tokens"]),
            elapsed_s: f(&l["elapsed_s"]),
            tok_s: f(&l["tok_s"]),
            parallel: f(&l["parallel"]),
            running: f(&l["running"]),
            cpu: f(&h["cpu"]),
            ram_used: f(&h["ram_used"]),
            ram_total: f(&h["ram_total"]),
            gpu_name: s(&m["hardware_static"]["gpu_name"]).map(str::to_string),
            hist_tok_s: series(&m["history"]["tok_s"]),
            hist_cpu: series(&m["history"]["cpu"]),
            requests: m["requests"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|r| Request {
                            time: f(&r["time"]).unwrap_or(0.0),
                            prompt: f(&r["prompt_tokens"]).unwrap_or(0.0),
                            reused: f(&r["reused"]).unwrap_or(0.0),
                            read: f(&r["prompt_read"]).unwrap_or(0.0),
                            first_token_s: f(&r["first_token_s"]),
                            output: f(&r["output_tokens"]).unwrap_or(0.0),
                            decode_tok_s: f(&r["decode_tok_s"]),
                            prompt_ms: f(&r["prompt_ms"]),
                            drafts_offered: f(&r["drafts_offered"]).unwrap_or(0.0),
                            drafts_accepted: f(&r["drafts_accepted"]).unwrap_or(0.0),
                            finish: s(&r["finish"]).unwrap_or("").to_string(),
                            duration_s: f(&r["duration_s"]).unwrap_or(0.0),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            t_requests: f(&t["requests"]).unwrap_or(0.0),
            t_prompt: f(&t["prompt_tokens"]).unwrap_or(0.0),
            t_reused: f(&t["reused"]).unwrap_or(0.0),
            t_output: f(&t["output_tokens"]).unwrap_or(0.0),
            t_prompt_ms: f(&t["prompt_ms"]).unwrap_or(0.0),
            t_decode_ms: f(&t["decode_ms"]).unwrap_or(0.0),
            t_offered: f(&t["drafts_offered"]).unwrap_or(0.0),
            t_accepted: f(&t["drafts_accepted"]).unwrap_or(0.0),
            t_since: f(&t["since"]),
            life_requests: f(&lt["requests"]).unwrap_or(0.0),
            life_output: f(&lt["output_tokens"]).unwrap_or(0.0),
            days,
            cc_parks: f(&cc["parks"]).unwrap_or(0.0),
            cc_restores: f(&cc["restores"]).unwrap_or(0.0),
            cc_ssd_files: f(&cc["ssd_files"]),
            cc_ssd_gib: f(&cc["ssd_gib"]),
            cc_ssd_restores: f(&cc["ssd_restores"]),
            threads: m["threads"]["threads"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|x| Thread {
                            key: s(&x["key"]).unwrap_or("").to_string(),
                            state: s(&x["state"]).unwrap_or("").to_string(),
                            tokens: f(&x["tokens"]).unwrap_or(0.0),
                            turns: f(&x["turns"]).unwrap_or(0.0),
                            pinned: x["pinned"].as_bool().unwrap_or(false),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            return_gap_p50: f(&m["threads"]["return_gap"]["p50_s"]),
            time: f(&m["time"]).unwrap_or(0.0),
        }
    }

    /// Mean decode speed of this server's run (output tokens over decode time).
    pub fn decode_mean(&self) -> Option<f64> {
        (self.t_decode_ms > 0.0).then(|| self.t_output / self.t_decode_ms * 1000.0)
    }

    /// Mean prompt reading speed (tokens actually read, not reused, over read time).
    pub fn read_mean(&self) -> Option<f64> {
        (self.t_prompt_ms > 0.0).then(|| (self.t_prompt - self.t_reused).max(0.0) / self.t_prompt_ms * 1000.0)
    }

    pub fn reuse_pct(&self) -> Option<f64> {
        pct(self.t_reused, self.t_prompt)
    }

    pub fn draft_pct(&self) -> Option<f64> {
        pct(self.t_accepted, self.t_offered)
    }

    pub fn busy(&self) -> bool {
        matches!(self.state.as_str(), "reading" | "generating")
    }
}
