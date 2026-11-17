//! The live screen: one loop that draws every frame, effects on top, data from background threads.

mod draw;
mod fx;
mod jobs;
mod theme;

pub use jobs::{Job, Spec};

use std::{
    sync::mpsc::{self, Receiver, Sender},
    thread,
    time::{Duration, Instant},
};

use anyhow::Result;
use ratatui::{
    DefaultTerminal,
    crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    layout::Rect,
};
use serde_json::{Value, json};

use crate::{
    api::Api,
    fmt::Snap,
    server::{self, RunConfig},
    settings::Settings,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tab {
    Live,
    Requests,
    Models,
    Setup,
    Tools,
    Logs,
    Chat,
}

impl Tab {
    pub const ALL: [Tab; 7] = [Tab::Live, Tab::Requests, Tab::Models, Tab::Setup, Tab::Tools, Tab::Logs, Tab::Chat];
    pub fn title(self) -> &'static str {
        match self {
            Tab::Live => "Live",
            Tab::Requests => "Requests",
            Tab::Models => "Models",
            Tab::Setup => "Setup",
            Tab::Tools => "Tools",
            Tab::Logs => "Logs",
            Tab::Chat => "Chat",
        }
    }
    fn index(self) -> usize {
        Tab::ALL.iter().position(|t| *t == self).unwrap_or(0)
    }
}

pub enum Msg {
    Metrics(Result<Value, String>),
    Config(Option<Value>),
    Loaded(bool),
    Action { label: String, result: Result<String, String> },
    ChatText { kind: String, text: String },
    ChatDone(Result<Value, String>),
    Catalog(Result<Value, String>),
    JobPid(u32),
    JobLine { text: String, replace: bool },
    JobProgress { frac: f64, label: String },
    JobDone { code: i32, result: Option<Vec<String>> },
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ToastKind {
    Info,
    Good,
    Bad,
}

pub struct Toast {
    pub text: String,
    pub kind: ToastKind,
    pub at: Instant,
}

/// A yes/no question in a box; `action` runs on yes.
pub struct Confirm {
    pub title: String,
    pub body: String,
    pub action: Action,
}

#[derive(Clone)]
pub enum Action {
    Start(RunConfig),
    Switch(RunConfig, bool),
    Restart,
    Stop,
    Load,
    Unload,
    Job(Spec),
}

pub struct ChatMsg {
    pub role: &'static str,
    pub text: String,
    pub reasoning: String,
}

/// Screen rectangles from the last frame, for effects aimed at one part.
#[derive(Default, Clone, Copy)]
pub struct Areas {
    pub screen: Rect,
    pub header: Rect,
    pub badge: Rect,
    pub body: Rect,
    pub speed: Rect,
    pub toast: Rect,
    pub popup: Rect,
    pub job: Rect,
}

pub struct App {
    pub s: Settings,
    pub api: Api,
    pub tab: Tab,
    pub snap: Option<Snap>,
    pub online: bool,
    pub loaded: bool,
    pub error: Option<String>,
    pub seen_online: bool,
    pub configs: Vec<RunConfig>,
    pub current_config: Option<String>,
    pub switchable: Vec<String>,
    pub sel_config: usize,
    pub sel_request: usize,
    pub log_lines: Vec<String>,
    pub log_scroll: usize,
    log_read_at: Instant,
    pub toast: Option<Toast>,
    pub confirm: Option<Confirm>,
    pub pending: Option<String>,
    pending_at: Instant,
    pub help: bool,
    pub chat: Vec<ChatMsg>,
    pub input: String,
    pub chat_busy: bool,
    pub chat_scroll: u16,
    /// the shown decode speed, eased toward the latest reading
    pub shown_tok_s: f64,
    pub peak_tok_s: f64,
    pub areas: Areas,
    pub catalog: Option<Result<Value, String>>,
    pub sel_model: usize,
    pub setup: SetupForm,
    pub sel_tool: usize,
    pub job: Option<Job>,
    pub job_scroll: usize,
    pub editing_port: Option<String>,
    pub fx: fx::Fx,
    last_request_time: f64,
    last_state: String,
    tx: Sender<Msg>,
    quit: bool,
}

fn poller(api: Api, tx: Sender<Msg>) {
    thread::spawn(move || {
        let mut n = 0u64;
        loop {
            let m = api.metrics().map_err(|e| format!("{e:#}"));
            let ok = m.is_ok();
            if tx.send(Msg::Metrics(m)).is_err() {
                return;
            }
            if ok && tx.send(Msg::Loaded(api.health().ok().and_then(|h| h["loaded"].as_bool()) == Some(true))).is_err() {
                return;
            }
            if ok && n % 5 == 0 && tx.send(Msg::Config(api.config().ok())).is_err() {
                return;
            }
            n += 1;
            thread::sleep(Duration::from_millis(if ok { 1000 } else { 1500 }));
        }
    });
}

impl App {
    fn new(s: Settings, tx: Sender<Msg>) -> Self {
        let api = Api::new(&s);
        let configs = server::list_configs(&s.config_dir);
        Self {
            api,
            configs,
            s,
            tab: Tab::Live,
            snap: None,
            online: false,
            loaded: false,
            error: None,
            seen_online: false,
            current_config: None,
            switchable: vec![],
            sel_config: 0,
            sel_request: 0,
            log_lines: vec![],
            log_scroll: 0,
            log_read_at: Instant::now() - Duration::from_secs(10),
            toast: None,
            confirm: None,
            pending: None,
            pending_at: Instant::now(),
            help: false,
            chat: vec![],
            input: String::new(),
            chat_busy: false,
            chat_scroll: 0,
            shown_tok_s: 0.0,
            peak_tok_s: 0.0,
            areas: Areas::default(),
            catalog: None,
            sel_model: 0,
            setup: SetupForm::default(),
            sel_tool: 0,
            job: None,
            job_scroll: 0,
            editing_port: None,
            fx: fx::Fx::new(),
            last_request_time: 0.0,
            last_state: String::new(),
            tx,
            quit: false,
        }
    }

    pub fn state(&self) -> String {
        if self.pending.is_some() && !self.online {
            return "starting".into();
        }
        match (&self.snap, self.online) {
            (Some(s), true) => s.state.clone(),
            _ => "offline".into(),
        }
    }

    pub fn toast(&mut self, text: impl Into<String>, kind: ToastKind) {
        self.toast = Some(Toast { text: text.into(), kind, at: Instant::now() });
        self.fx.toast(self.areas.toast, kind);
    }

    fn on_msg(&mut self, m: Msg) {
        match m {
            Msg::Metrics(Ok(v)) => {
                let snap = Snap::from(&v);
                if !self.online {
                    self.fx.online(self.areas.body);
                    if self.seen_online {
                        self.toast("server is back", ToastKind::Good);
                    }
                }
                self.online = true;
                self.seen_online = true;
                self.error = None;
                // a request that finished since the last reading: flash the speed panel
                let newest = snap.requests.first().map(|r| r.time).unwrap_or(0.0);
                if self.last_request_time > 0.0 && newest > self.last_request_time {
                    self.fx.finished(self.areas.speed);
                }
                self.last_request_time = newest;
                if snap.state != self.last_state {
                    if snap.busy() {
                        self.fx.busy(self.areas.badge);
                    } else {
                        self.fx.calm();
                    }
                    if self.last_state == "unloaded" && snap.state == "idle" {
                        self.toast("model loaded", ToastKind::Good);
                    }
                    self.last_state = snap.state.clone();
                }
                if let Some(t) = snap.tok_s {
                    self.peak_tok_s = self.peak_tok_s.max(t);
                }
                self.snap = Some(snap);
            }
            Msg::Metrics(Err(e)) => {
                if self.online {
                    self.fx.offline(self.areas.body);
                    if self.pending.is_none() {
                        self.toast("server went away", ToastKind::Bad);
                    }
                }
                self.online = false;
                self.error = Some(e);
                self.last_state = "offline".into();
                self.fx.calm();
            }
            Msg::Loaded(loaded) => {
                // a start or switch is done once /health says loaded
                if loaded
                    && self.pending_at.elapsed() > Duration::from_secs(3)
                    && self.pending.as_deref().is_some_and(|p| p.starts_with("starting") || p.starts_with("switching")) {
                    let p = self.pending.take().unwrap_or_default();
                    self.toast(format!("{p}: ready"), ToastKind::Good);
                    self.fx.online(self.areas.body);
                }
                self.loaded = loaded;
            }
            Msg::Config(c) => {
                if let Some(c) = c {
                    self.current_config = c["file"].as_str().map(str::to_string);
                    self.switchable = c["models"]
                        .as_array()
                        .map(|a| a.iter().filter_map(|m| m["file"].as_str().map(str::to_string)).collect())
                        .unwrap_or_default();
                }
            }
            Msg::Action { label, result } => match result {
                Ok(t) => {
                    if !(label.starts_with("starting") || label.starts_with("switching")) {
                        self.pending = None;
                    }
                    self.toast(t, ToastKind::Good);
                }
                Err(e) => {
                    self.pending = None;
                    self.toast(format!("{label}: {e}"), ToastKind::Bad);
                }
            },
            Msg::ChatText { kind, text } => {
                if let Some(last) = self.chat.last_mut() {
                    if kind == "reasoning" {
                        last.reasoning.push_str(&text);
                    } else {
                        last.text.push_str(&text);
                    }
                }
            }
            Msg::Catalog(r) => {
                if let Ok(v) = &r {
                    let rows = v["rows"].as_array().cloned().unwrap_or_default();
                    // start on the recommended row, else the first that fits
                    self.sel_model = rows
                        .iter()
                        .position(|r| r["recommended"].as_bool() == Some(true))
                        .or_else(|| rows.iter().position(|r| r["verdict"].as_str() == Some("fits")))
                        .unwrap_or(0);
                    if let Some(ctx) = v["contexts"].as_array() {
                        self.setup.contexts = ctx.iter().filter_map(|c| c.as_u64()).collect();
                        self.setup.context = self.setup.contexts.iter().position(|c| *c == 262_144).unwrap_or(self.setup.contexts.len().saturating_sub(1));
                    }
                }
                self.catalog = Some(r);
            }
            Msg::JobPid(pid) => {
                if let Some(j) = self.job.as_mut() {
                    j.pid = Some(pid);
                }
            }
            Msg::JobLine { text, replace } => {
                if let Some(j) = self.job.as_mut() {
                    j.push(text, replace);
                }
            }
            Msg::JobProgress { frac, label } => {
                if let Some(j) = self.job.as_mut() {
                    j.progress = Some((frac, label));
                }
            }
            Msg::JobDone { code, result } => {
                if let Some(j) = self.job.as_mut() {
                    j.code = Some(code);
                    j.ended = Some(Instant::now());
                    j.result = result;
                    j.progress = None;
                    let ok = code == 0;
                    let title = j.spec.title.clone();
                    self.job_scroll = usize::MAX; // the result from its top (pgdn goes on)
                    self.toast(if ok { format!("{title}: done") } else { format!("{title}: failed (exit {code})") }, if ok { ToastKind::Good } else { ToastKind::Bad });
                    self.fx.finished(self.areas.job);
                    self.configs = server::list_configs(&self.s.config_dir);
                }
            }
            Msg::ChatDone(r) => {
                self.chat_busy = false;
                if let Err(e) = r {
                    if let Some(last) = self.chat.last_mut() {
                        last.text.push_str(&format!("\n[error: {e}]"));
                    }
                }
            }
        }
    }

    fn run_action(&mut self, a: Action) {
        let api = self.api.clone();
        let s = self.s.clone();
        let tx = self.tx.clone();
        let label = match &a {
            Action::Start(c) => format!("starting {}", c.file.trim_end_matches(".json")),
            Action::Switch(c, _) => format!("switching to {}", c.file.trim_end_matches(".json")),
            Action::Restart => "restarting".into(),
            Action::Stop => "stopping".into(),
            Action::Load => "loading".into(),
            Action::Unload => "unloading".into(),
            Action::Job(spec) => {
                self.start_job(spec.clone());
                return;
            }
        };
        self.pending = Some(label.clone());
        self.pending_at = Instant::now();
        self.fx.pending(self.areas.header);
        thread::spawn(move || {
            let result = match a {
                Action::Start(c) => {
                    std::fs::create_dir_all(&s.state_dir).ok();
                    std::fs::write(s.state_dir.join("last-config"), c.path.display().to_string()).ok();
                    server::start_detached(&s, &c).map(|pid| format!("started pid {pid}, loading {}", c.model))
                }
                Action::Switch(c, force) => api.restart(Some(&c.file), force).map(|_| format!("restarting as {}", c.model)),
                Action::Restart => api.restart(None, false).map(|_| "restarting".to_string()),
                Action::Stop => server::stop(&s, Duration::from_secs(90), false, |_| {}).map(|pid| format!("stopped pid {pid}")),
                Action::Load => api.load().map(|_| "loaded".to_string()),
                Action::Unload => api.unload().map(|r| format!("model {r}: the GPU is free")),
                Action::Job(_) => unreachable!("started above"),
            };
            tx.send(Msg::Action { label, result: result.map_err(|e| format!("{e:#}")) }).ok();
        });
    }

    pub fn start_job(&mut self, spec: Spec) {
        if self.job.as_ref().is_some_and(Job::running) {
            self.toast("a job is still running (x stops it)", ToastKind::Bad);
            return;
        }
        self.job = Some(Job::new(spec.clone()));
        self.job_scroll = 0;
        self.fx.pending(self.areas.job);
        // the output panel's text width (the panel is on the right of Setup and Tools)
        let w = (self.areas.job.width as usize).saturating_sub(4).max(40);
        jobs::start(&self.s, spec, w, self.tx.clone());
    }

    fn load_catalog(&mut self) {
        if self.catalog.is_some() {
            return;
        }
        self.catalog = Some(Err("loading the model list…".into()));
        let s = self.s.clone();
        let tx = self.tx.clone();
        thread::spawn(move || {
            tx.send(Msg::Catalog(crate::bridge::catalog(&s).map_err(|e| format!("{e:#}")))).ok();
        });
    }

    pub fn catalog_rows(&self) -> Vec<Value> {
        match &self.catalog {
            Some(Ok(v)) => v["rows"].as_array().cloned().unwrap_or_default(),
            _ => vec![],
        }
    }

    /// `launch` arguments for the picked model and the form.
    pub fn launch_args(&self) -> Option<Vec<String>> {
        let rows = self.catalog_rows();
        let r = rows.get(self.sel_model)?;
        let f = &self.setup;
        let mut a = vec![
            "--model".to_string(),
            r["model"].as_str()?.to_string(),
            "--family".into(),
            r["family"].as_str()?.to_string(),
            "--kv".into(),
            KV[f.kv].into(),
            "--vision".into(),
            VISION[f.vision].into(),
            "--drafting".into(),
            DRAFTING[f.drafting].into(),
            "--tune".into(),
            TUNE[f.tune].into(),
            "--access".into(),
            ACCESS[f.access].into(),
            "--port".into(),
            f.port.to_string(),
        ];
        if let Some(c) = f.contexts.get(f.context) {
            a.extend(["--context".into(), c.to_string()]);
        }
        Some(a)
    }

    /// The tool list: (label, what it does, the job, ask first).
    pub fn tools(&self) -> Vec<(String, &'static str, Spec, bool)> {
        use crate::bridge::Out::{Json, Jsonl};
        let url = self.api.base().to_string();
        let target = self
            .current_config
            .as_ref()
            .and_then(|f| self.configs.iter().find(|c| &c.file == f))
            .or_else(|| self.configs.get(self.sel_config))
            .map(|c| c.path.display().to_string());
        let mut t = vec![
            ("Check the install".to_string(), "engine components, what is missing", Spec::new("doctor", &["doctor"], &[], Json), false),
            ("Look for an engine update".into(), "reads the signed feed; nothing is changed", Spec::new("update check", &["update"], &["--check"], Jsonl), false),
            ("Install the engine update".into(), "download, verify, test-start, then switch", Spec::new("update", &["update"], &[], Jsonl), true),
            ("Roll the engine back".into(), "to the version the last update kept", Spec::new("engine rollback", &["components", "rollback"], &[], Jsonl), true),
            ("Engine packages here".into(), "which one this PC would start", Spec::new("engine packages", &["components", "engine"], &["--full"], Json), false),
            ("DFlash2 drafter: plan".into(), "what fetching it would download", Spec::new("drafter plan", &["drafting", "plan"], &[], Jsonl), false),
            ("DFlash2 drafter: fetch".into(), "download and pack it (about 1.2 GB)", Spec::new("drafter fetch", &["drafting", "fetch"], &["--if-needed"], Jsonl), true),
            ("Installed models".into(), "run configs setup wrote", Spec::new("installed models", &["models", "list"], &[], Json), false),
            ("Saved presets".into(), "tuning presets", Spec::new("presets", &["presets", "list"], &[], Json), false),
            ("Tuning jobs".into(), "recorded tuning runs", Spec::new("tuning jobs", &["tune", "list"], &[], Json), false),
            ("Benchmark: smoke".into(), "a quick serving check against this server", Spec::new("bench smoke", &["bench", "run"], &["--url", &url, "--suite", "serving-smoke"], Json), true),
            ("Benchmark: standard".into(), "docs/BENCHMARK_PROTOCOL.md against this server", Spec::new("bench standard", &["bench", "run"], &["--url", &url], Json), true),
            ("Request activity".into(), "the server's recent requests, redacted", Spec::new("activity", &["logs"], &["--url", &url], Json), false),
            ("Saved profiles".into(), "serving overrides outside the run config", Spec::new("profiles", &["profile", "list"], &[], Json), false),
            ("Clean up leftovers".into(), "end servers and engines whose owner is gone", Spec::new("cleanup", &["runtime", "cleanup"], &[], Json), true),
            ("Repair folder access".into(), "fix the permissions of Strata's own folders", Spec::new("repair access", &["repair-access"], &[], Json), true),
        ];
        if let Some(cfg) = target {
            let name = std::path::Path::new(&cfg).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            t.insert(10, (format!("Tune {name}"), "find the fastest settings and save a preset (~20 min, uses the GPU)", Spec::new("tune auto", &["tune", "auto"], &[&cfg], Json), true));
            t.insert(11, (format!("Preset for {name}"), "which preset it starts with, still valid?", Spec::new("preset status", &["presets", "status"], &[&cfg], Json), false));
        }
        for h in ["claude-code", "opencode", "codex", "pi", "hermes"] {
            t.push((format!("Connect {h}"), "the settings this app needs for this server", Spec::new(&format!("connect {h}"), &["connect"], &[h, "--url", &url, "--model", &self.snap.as_ref().map(|s| s.model.clone()).unwrap_or_default()], Json), false));
        }
        t
    }

    fn send_chat(&mut self) {
        let text = self.input.trim().to_string();
        if text.is_empty() || self.chat_busy {
            return;
        }
        if !self.online {
            self.toast("no server to talk to", ToastKind::Bad);
            return;
        }
        self.input.clear();
        self.chat.push(ChatMsg { role: "user", text, reasoning: String::new() });
        let messages: Vec<Value> = self.chat.iter().map(|m| json!({ "role": m.role, "content": m.text })).collect();
        self.chat.push(ChatMsg { role: "assistant", text: String::new(), reasoning: String::new() });
        self.chat_busy = true;
        self.chat_scroll = 0;
        let api = self.api.clone();
        let tx = self.tx.clone();
        thread::spawn(move || {
            let tx2 = tx.clone();
            let r = api.chat_stream(&Value::Array(messages), None, move |kind, text| {
                tx2.send(Msg::ChatText { kind: kind.to_string(), text: text.to_string() }).ok();
            });
            tx.send(Msg::ChatDone(r.map_err(|e| format!("{e:#}")))).ok();
        });
    }

    fn set_tab(&mut self, t: Tab) {
        if t != self.tab {
            let forward = t.index() > self.tab.index();
            self.tab = t;
            self.fx.tab(self.areas.body, forward);
            if t == Tab::Setup {
                self.load_catalog();
            }
        }
    }

    fn selected_config(&self) -> Option<RunConfig> {
        self.configs.get(self.sel_config).cloned()
    }

    /// Enter on a run config: start it, or switch the running server to it.
    fn choose_config(&mut self) {
        let Some(c) = self.selected_config() else { return };
        if !self.online {
            self.run_action(Action::Start(c));
            return;
        }
        if self.current_config.as_deref() == Some(c.file.as_str()) {
            self.toast(format!("{} is already running", c.file), ToastKind::Info);
            return;
        }
        if !self.switchable.iter().any(|f| f == &c.file) {
            self.toast("the server only switches within its own config folder: stop it (s) and start this one", ToastKind::Bad);
            return;
        }
        let busy = self.snap.as_ref().is_some_and(Snap::busy);
        self.ask(
            "Switch model",
            format!(
                "Restart the server as {} ({})?{}",
                c.file.trim_end_matches(".json"),
                c.model,
                if busy { "\nA request is running: it will be cut off." } else { "" }
            ),
            Action::Switch(c, busy),
        );
    }

    fn ask(&mut self, title: &str, body: String, action: Action) {
        self.confirm = Some(Confirm { title: title.into(), body, action });
        self.fx.popup(self.areas.popup);
    }

    fn key(&mut self, k: KeyEvent) {
        if k.kind != KeyEventKind::Press {
            return;
        }
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        if ctrl && matches!(k.code, KeyCode::Char('c') | KeyCode::Char('q')) {
            self.quit = true;
            return;
        }
        if let Some(c) = self.confirm.take() {
            if matches!(k.code, KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter) {
                self.run_action(c.action);
            }
            return;
        }
        if self.help {
            self.help = false;
            return;
        }
        match k.code {
            KeyCode::Tab => return self.set_tab(Tab::ALL[(self.tab.index() + 1) % Tab::ALL.len()]),
            KeyCode::BackTab => return self.set_tab(Tab::ALL[(self.tab.index() + Tab::ALL.len() - 1) % Tab::ALL.len()]),
            _ => {}
        }
        if let Some(p) = self.editing_port.as_mut() {
            match k.code {
                KeyCode::Char(c) if c.is_ascii_digit() && p.len() < 5 => p.push(c),
                KeyCode::Backspace => {
                    p.pop();
                }
                KeyCode::Enter => {
                    match p.parse::<u16>() {
                        Ok(n) if n >= 1024 => self.setup.port = n,
                        _ => self.toast("a port from 1024 to 65535", ToastKind::Bad),
                    }
                    self.editing_port = None;
                }
                KeyCode::Esc => self.editing_port = None,
                _ => {}
            }
            return;
        }
        if matches!(self.tab, Tab::Setup | Tab::Tools) {
            let running = self.job.as_ref().is_some_and(Job::running);
            match k.code {
                KeyCode::Char('x') if running => {
                    if let Some(pid) = self.job.as_ref().and_then(|j| j.pid) {
                        jobs::interrupt(pid);
                        self.toast("stopping the job", ToastKind::Info);
                    }
                    return;
                }
                KeyCode::PageUp => {
                    self.job_scroll = self.job_scroll.saturating_add(10);
                    return;
                }
                KeyCode::PageDown => {
                    self.job_scroll = self.job_scroll.saturating_sub(10);
                    return;
                }
                _ => {}
            }
        }
        if self.tab == Tab::Setup {
            let n = self.catalog_rows().len();
            let f = &mut self.setup;
            let handled = match k.code {
                KeyCode::Up | KeyCode::Char('k') => {
                    self.sel_model = self.sel_model.saturating_sub(1);
                    true
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.sel_model = (self.sel_model + 1).min(n.saturating_sub(1));
                    true
                }
                KeyCode::Char('c') => {
                    f.context = (f.context + 1) % f.contexts.len().max(1);
                    true
                }
                KeyCode::Char('C') => {
                    f.context = (f.context + f.contexts.len().max(1) - 1) % f.contexts.len().max(1);
                    true
                }
                KeyCode::Char('m') => {
                    f.kv = (f.kv + 1) % KV.len();
                    true
                }
                KeyCode::Char('v') => {
                    f.vision = (f.vision + 1) % VISION.len();
                    true
                }
                KeyCode::Char('d') => {
                    f.drafting = (f.drafting + 1) % DRAFTING.len();
                    true
                }
                KeyCode::Char('t') => {
                    f.tune = (f.tune + 1) % TUNE.len();
                    true
                }
                KeyCode::Char('a') => {
                    f.access = (f.access + 1) % ACCESS.len();
                    true
                }
                KeyCode::Char('o') => {
                    self.editing_port = Some(String::new());
                    true
                }
                KeyCode::Char('p') => {
                    if let Some(mut a) = self.launch_args() {
                        a.push("--plan".into());
                        let args: Vec<&str> = a.iter().map(String::as_str).collect();
                        self.start_job(Spec::new("plan", &["launch"], &args, crate::bridge::Out::Json));
                    }
                    true
                }
                KeyCode::Enter => {
                    if let Some(mut a) = self.launch_args() {
                        let rows = self.catalog_rows();
                        let r = &rows[self.sel_model];
                        let title = r["title"].as_str().unwrap_or("this model").to_string();
                        let gb = r["download_gb"].as_f64().unwrap_or(0.0);
                        a.extend(["--no-tui".into(), "--yes".into()]);
                        let args: Vec<&str> = a.iter().map(String::as_str).collect();
                        let spec = Spec::new(&format!("set up {title}"), &["launch"], &args, crate::bridge::Out::Json);
                        let installed = r["installed"].as_bool() == Some(true);
                        self.ask(
                            "Set up and start",
                            format!(
                                "{title}\n{}Then it is configured{} and started on port {}.\n\n{}",
                                if installed { "Already downloaded. ".to_string() } else { format!("Downloads up to {gb:.0} GB. ") },
                                if TUNE[self.setup.tune] != "skip" { ", tuned" } else { "" },
                                self.setup.port,
                                spec.command_line()
                            ),
                            Action::Job(spec),
                        );
                    }
                    true
                }
                KeyCode::Char('R') => {
                    self.catalog = None;
                    self.load_catalog();
                    true
                }
                _ => false,
            };
            if handled {
                return;
            }
        }
        if self.tab == Tab::Tools {
            let tools = self.tools();
            let handled = match k.code {
                KeyCode::Up | KeyCode::Char('k') => {
                    self.sel_tool = self.sel_tool.saturating_sub(1);
                    true
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.sel_tool = (self.sel_tool + 1).min(tools.len().saturating_sub(1));
                    true
                }
                KeyCode::Enter => {
                    if let Some((label, what, spec, ask)) = tools.get(self.sel_tool).cloned() {
                        if ask {
                            self.ask(&label, format!("{what}.\n\n{}", spec.command_line()), Action::Job(spec));
                        } else {
                            self.start_job(spec);
                        }
                    }
                    true
                }
                _ => false,
            };
            if handled {
                return;
            }
        }
        if self.tab == Tab::Chat {
            match k.code {
                KeyCode::Enter => self.send_chat(),
                KeyCode::Backspace => {
                    self.input.pop();
                }
                KeyCode::Esc => {
                    if self.input.is_empty() {
                        self.set_tab(Tab::Live)
                    } else {
                        self.input.clear()
                    }
                }
                KeyCode::Char('l') if ctrl => self.chat.clear(),
                KeyCode::PageUp | KeyCode::Up => self.chat_scroll = self.chat_scroll.saturating_add(3),
                KeyCode::PageDown | KeyCode::Down => self.chat_scroll = self.chat_scroll.saturating_sub(3),
                KeyCode::Char(ch) if !ctrl => self.input.push(ch),
                _ => {}
            }
            return;
        }
        match k.code {
            KeyCode::Char('q') | KeyCode::Esc => self.quit = true,
            KeyCode::Char('?') => self.help = true,
            KeyCode::Char(d @ '1'..='7') => self.set_tab(Tab::ALL[(d as u8 - b'1') as usize]),
            KeyCode::Char('l') if self.online => self.run_action(Action::Load),
            KeyCode::Char('u') if self.online => self.run_action(Action::Unload),
            KeyCode::Char('r') if self.online => {
                self.ask("Restart", "Restart the server with its run config read again?".into(), Action::Restart)
            }
            KeyCode::Char('s') if self.online => {
                let who = server::owner(&self.s);
                let body = match who {
                    Some(o) if o.ours => format!("Stop the server (pid {})? Conversations are parked first.", o.pid),
                    Some(o) => format!("Stop the server (pid {}, started outside intelinference)?", o.pid),
                    None => "Nothing to stop on this port.".into(),
                };
                self.ask("Stop", body, Action::Stop)
            }
            KeyCode::Up | KeyCode::Char('k') => match self.tab {
                Tab::Models | Tab::Live => self.sel_config = self.sel_config.saturating_sub(1),
                Tab::Requests => self.sel_request = self.sel_request.saturating_sub(1),
                Tab::Logs => self.log_scroll = self.log_scroll.saturating_add(1),
                Tab::Chat | Tab::Setup | Tab::Tools => {}
            },
            KeyCode::Down | KeyCode::Char('j') => match self.tab {
                Tab::Models | Tab::Live => self.sel_config = (self.sel_config + 1).min(self.configs.len().saturating_sub(1)),
                Tab::Requests => self.sel_request += 1,
                Tab::Logs => self.log_scroll = self.log_scroll.saturating_sub(1),
                Tab::Chat | Tab::Setup | Tab::Tools => {}
            },
            KeyCode::PageUp if self.tab == Tab::Logs => self.log_scroll = self.log_scroll.saturating_add(20),
            KeyCode::PageDown if self.tab == Tab::Logs => self.log_scroll = self.log_scroll.saturating_sub(20),
            KeyCode::End | KeyCode::Char('G') if self.tab == Tab::Logs => self.log_scroll = 0,
            KeyCode::Enter if matches!(self.tab, Tab::Models) || (self.tab == Tab::Live && !self.online) => self.choose_config(),
            KeyCode::Char('R') => {
                self.configs = server::list_configs(&self.s.config_dir);
                self.toast(format!("{} run configs", self.configs.len()), ToastKind::Info);
            }
            _ => {}
        }
    }

    fn tick(&mut self, dt: f64) {
        // ease the big number toward the live reading (or toward 0 when nothing decodes)
        let target = self.snap.as_ref().and_then(|s| s.tok_s).unwrap_or(0.0);
        let k = 1.0 - (-dt * 6.0).exp();
        self.shown_tok_s += (target - self.shown_tok_s) * k;
        if (self.shown_tok_s - target).abs() < 0.05 {
            self.shown_tok_s = target;
        }
        if let Some(t) = &self.toast
            && t.at.elapsed() > Duration::from_secs(5)
        {
            self.toast = None;
        }
        if matches!(self.tab, Tab::Logs | Tab::Live) && self.log_read_at.elapsed() > Duration::from_millis(700) {
            self.log_lines = read_tail(&self.s.log, 400);
            self.log_read_at = Instant::now();
        }
    }

    pub fn animating(&self) -> bool {
        self.fx.running() || self.job.as_ref().is_some_and(Job::running) || (self.shown_tok_s - self.snap.as_ref().and_then(|s| s.tok_s).unwrap_or(0.0)).abs() > 0.05 || self.chat_busy
    }
}

fn read_tail(path: &std::path::Path, n: usize) -> Vec<String> {
    use std::io::{Read, Seek, SeekFrom};
    let Ok(mut f) = std::fs::File::open(path) else { return vec![] };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    f.seek(SeekFrom::Start(len.saturating_sub(128 * 1024))).ok();
    let mut b = Vec::new();
    f.read_to_end(&mut b).ok();
    let t = String::from_utf8_lossy(&b);
    let v: Vec<String> = t.lines().map(|l| l.replace('\t', "    ")).collect();
    v[v.len().saturating_sub(n)..].to_vec()
}

fn main_loop(term: &mut DefaultTerminal, app: &mut App, rx: Receiver<Msg>) -> Result<()> {
    let mut last = Instant::now();
    while !app.quit {
        while let Ok(m) = rx.try_recv() {
            app.on_msg(m);
        }
        let now = Instant::now();
        let dt = now - last;
        last = now;
        app.tick(dt.as_secs_f64());
        term.draw(|f| {
            draw::draw(f, app);
            let area = f.area();
            app.fx.process(dt, f.buffer_mut(), area);
        })?;
        let wait = if app.animating() { Duration::from_millis(16) } else { Duration::from_millis(200) };
        if event::poll(wait)? {
            match event::read()? {
                Event::Key(k) => app.key(k),
                Event::Resize(..) => {}
                _ => {}
            }
        }
    }
    Ok(())
}

pub fn run(s: Settings, tab: Tab) -> Result<i32> {
    let (tx, rx) = mpsc::channel();
    let mut app = App::new(s, tx.clone());
    // pick the run config the server runs, else the last one started
    if let Ok(p) = std::fs::read_to_string(app.s.state_dir.join("last-config")) {
        let p = p.trim();
        if let Some(i) = app.configs.iter().position(|c| c.path.display().to_string() == p) {
            app.sel_config = i;
        }
    }
    poller(app.api.clone(), tx);
    let mut term = ratatui::init();
    let size = term.size().map(|s| Rect::new(0, 0, s.width, s.height)).unwrap_or_default();
    app.fx.intro(size);
    if tab != Tab::Live {
        app.tab = tab;
        if tab == Tab::Setup {
            app.load_catalog();
        }
    }
    let r = main_loop(&mut term, &mut app, rx);
    ratatui::restore();
    r.map(|_| 0)
}

pub const KV: [&str; 4] = ["int8", "q4_0", "k8v4", "fp16"];
pub const VISION: [&str; 4] = ["off", "on", "cpu", "gpu"];
pub const DRAFTING: [&str; 4] = ["auto", "mtp", "dflash2", "off"];
pub const TUNE: [&str; 3] = ["skip", "quick", "thorough"];
pub const ACCESS: [&str; 4] = ["local", "lan", "tailscale", "cloudflare"];

/// The setup screen's choices, as indexes into the lists above.
pub struct SetupForm {
    pub contexts: Vec<u64>,
    pub context: usize,
    pub kv: usize,
    pub vision: usize,
    pub drafting: usize,
    pub tune: usize,
    pub access: usize,
    pub port: u16,
}

impl Default for SetupForm {
    fn default() -> Self {
        Self { contexts: vec![262_144], context: 0, kv: 0, vision: 0, drafting: 0, tune: 0, access: 0, port: 8080 }
    }
}
