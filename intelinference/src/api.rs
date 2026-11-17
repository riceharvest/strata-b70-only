//! The server's HTTP API: /health, /metrics, /config and the control posts (load, unload, restart).
//!
//! Replies are read as loose JSON: an older or newer server with fewer or more keys still shows what it has.

use std::{io::Read, time::Duration};

use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use ureq::Agent;

use crate::settings::Settings;

#[derive(Clone)]
pub struct Api {
    base: String,
    key: Option<String>,
    agent: Agent,
    long: Agent,
}

fn agent(timeout: Option<Duration>) -> Agent {
    Agent::config_builder()
        .timeout_global(timeout)
        .timeout_connect(Some(Duration::from_secs(2)))
        .http_status_as_error(false)
        .build()
        .into()
}

impl Api {
    pub fn new(s: &Settings) -> Self {
        Self {
            base: s.base_url(),
            key: s.api_key.clone(),
            agent: agent(Some(Duration::from_secs(4))),
            long: agent(None),
        }
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    fn auth<B>(&self, r: ureq::RequestBuilder<B>) -> ureq::RequestBuilder<B> {
        match &self.key {
            Some(k) => r.header("Authorization", format!("Bearer {k}")),
            None => r,
        }
    }

    pub fn get(&self, path: &str) -> Result<Value> {
        let mut resp = self.auth(self.agent.get(format!("{}{path}", self.base))).call()?;
        let status = resp.status().as_u16();
        let body: Value = resp.body_mut().with_config().limit(64 << 20).read_json().unwrap_or(Value::Null);
        if status >= 400 {
            return Err(anyhow!("{path}: HTTP {status}: {}", error_text(&body)));
        }
        Ok(body)
    }

    /// POST JSON; returns the status and the reply (errors are replies too, for the caller to word).
    pub fn post(&self, path: &str, body: &Value, long: bool) -> Result<(u16, Value)> {
        let a = if long { &self.long } else { &self.agent };
        let mut resp = self
            .auth(a.post(format!("{}{path}", self.base)))
            .header("Content-Type", "application/json")
            .send_json(body)?;
        let status = resp.status().as_u16();
        let v: Value = resp.body_mut().read_json().unwrap_or(Value::Null);
        Ok((status, v))
    }

    pub fn health(&self) -> Result<Value> {
        self.get("/health")
    }

    pub fn metrics(&self) -> Result<Value> {
        self.get("/metrics")
    }

    pub fn unload(&self) -> Result<String> {
        let (st, v) = self.post("/unload", &json!({}), true)?;
        match st {
            200 => Ok(v["status"].as_str().unwrap_or("unloaded").to_string()),
            409 => Err(anyhow!("a request is running or queued; try again when it ends")),
            _ => Err(anyhow!("unload: HTTP {st}: {}", error_text(&v))),
        }
    }

    pub fn load(&self) -> Result<()> {
        let (st, v) = self.post("/load", &json!({}), true)?;
        match st {
            200 => Ok(()),
            _ => Err(anyhow!("load: HTTP {st}: {}", error_text(&v))),
        }
    }

    /// Restart the server; with `config`, as that run config (a file name in the current config's folder).
    pub fn restart(&self, config: Option<&str>, force: bool) -> Result<()> {
        let path = if force { "/v1/restart?force=1" } else { "/v1/restart" };
        let body = match config {
            Some(c) => json!({ "config": c }),
            None => json!({}),
        };
        let (st, v) = self.post(path, &body, false)?;
        match st {
            200 => Ok(()),
            409 => Err(anyhow!("{} (add --force to end it)", error_text(&v))),
            _ => Err(anyhow!("restart: HTTP {st}: {}", error_text(&v))),
        }
    }

    /// The run config the server runs and the configs it can switch to (GET /config).
    pub fn config(&self) -> Result<Value> {
        self.get("/config")
    }

    /// Stream a chat answer: `on_text(kind, text)` per piece, kind "reasoning" or "content".
    pub fn chat_stream(&self, messages: &Value, max_tokens: Option<u32>, mut on_text: impl FnMut(&str, &str)) -> Result<Value> {
        let mut body = json!({ "messages": messages, "stream": true, "stream_options": { "include_usage": true } });
        if let Some(m) = max_tokens {
            body["max_tokens"] = json!(m);
        }
        let resp = self
            .auth(self.long.post(format!("{}/v1/chat/completions", self.base)))
            .header("Content-Type", "application/json")
            .send_json(&body)?;
        let status = resp.status().as_u16();
        let mut reader = resp.into_body().into_reader();
        if status >= 400 {
            let mut s = String::new();
            reader.read_to_string(&mut s).ok();
            let v: Value = serde_json::from_str(&s).unwrap_or(Value::String(s));
            return Err(anyhow!("HTTP {status}: {}", error_text(&v)));
        }
        let mut usage = Value::Null;
        let mut pending = String::new();
        let mut buf = [0u8; 8192];
        loop {
            let n = reader.read(&mut buf).context("reading the answer")?;
            if n == 0 {
                break;
            }
            pending.push_str(&String::from_utf8_lossy(&buf[..n]));
            while let Some(i) = pending.find('\n') {
                let line = pending[..i].trim_end_matches('\r').to_string();
                pending.drain(..=i);
                let Some(data) = line.strip_prefix("data:") else { continue };
                let data = data.trim();
                if data == "[DONE]" {
                    return Ok(usage);
                }
                let Ok(ev) = serde_json::from_str::<Value>(data) else { continue };
                if !ev["usage"].is_null() {
                    usage = ev["usage"].clone();
                }
                let d = &ev["choices"][0]["delta"];
                for (key, kind) in [("reasoning_content", "reasoning"), ("reasoning", "reasoning"), ("content", "content")] {
                    if let Some(t) = d[key].as_str().filter(|t| !t.is_empty()) {
                        on_text(kind, t);
                    }
                }
            }
        }
        Ok(usage)
    }
}

/// The message of an error reply in any of the shapes the server uses.
pub fn error_text(v: &Value) -> String {
    v["error"]["message"]
        .as_str()
        .or_else(|| v["error"].as_str())
        .or_else(|| v["message"].as_str())
        .or_else(|| v.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| if v.is_null() { "no reply body".into() } else { v.to_string() })
}

// ---- small readers for loose JSON ----

pub fn f(v: &Value) -> Option<f64> {
    v.as_f64()
}

pub fn s(v: &Value) -> Option<&str> {
    v.as_str()
}
