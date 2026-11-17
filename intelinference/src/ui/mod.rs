//! The live screen: one loop that draws every frame, effects on top, data from background threads.

mod draw;
mod fx;
mod theme;

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
    Logs,
    Chat,
}

impl Tab {
    pub const ALL: [Tab; 5] = [Tab::Live, Tab::Requests, Tab::Models, Tab::Logs, Tab::Chat];
    pub fn title(self) -> &'static str {
        match self {
            Tab::Live => "Live",
            Tab::Requests => "Requests",
            Tab::Models => "Models",
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
            };
            tx.send(Msg::Action { label, result: result.map_err(|e| format!("{e:#}")) }).ok();
        });
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
            KeyCode::Char(d @ '1'..='5') => self.set_tab(Tab::ALL[(d as u8 - b'1') as usize]),
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
                Tab::Chat => {}
            },
            KeyCode::Down | KeyCode::Char('j') => match self.tab {
                Tab::Models | Tab::Live => self.sel_config = (self.sel_config + 1).min(self.configs.len().saturating_sub(1)),
                Tab::Requests => self.sel_request += 1,
                Tab::Logs => self.log_scroll = self.log_scroll.saturating_sub(1),
                Tab::Chat => {}
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
        self.fx.running() || (self.shown_tok_s - self.snap.as_ref().and_then(|s| s.tok_s).unwrap_or(0.0)).abs() > 0.05 || self.chat_busy
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

pub fn run(s: Settings) -> Result<i32> {
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
    let r = main_loop(&mut term, &mut app, rx);
    ratatui::restore();
    r.map(|_| 0)
}
