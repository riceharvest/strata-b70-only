//! Where things are: the server to talk to, the run configs, the checkout, the logs.
//!
//! Every value comes from, in order: a command-line flag, an `INTELINFERENCE_*` environment variable,
//! `settings.json` in the config folder (`$XDG_CONFIG_HOME/intelinference/`), then a default. Nothing about
//! one machine is built in.

use std::{
    env, fs,
    path::{Path, PathBuf},
};

use serde::Deserialize;

/// What the command line can set; `None` = not given there.
#[derive(Debug, Default, Clone)]
pub struct Overrides {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub api_key: Option<String>,
    pub config_dir: Option<PathBuf>,
    pub root: Option<PathBuf>,
    pub python: Option<String>,
    pub log: Option<PathBuf>,
}

/// `settings.json`: every key optional.
#[derive(Debug, Default, Deserialize)]
struct File {
    host: Option<String>,
    port: Option<u16>,
    api_key: Option<String>,
    config_dir: Option<PathBuf>,
    root: Option<PathBuf>,
    python: Option<String>,
    log: Option<PathBuf>,
    /// extra arguments for `serve/server.py` when this tool starts it (e.g. `["--no-api-key"]`)
    #[serde(default)]
    server_args: Vec<String>,
    /// environment variables for the server this tool starts
    #[serde(default)]
    server_env: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct Settings {
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
    /// folder of run configs (`*.json` with `"engine": "strata"`)
    pub config_dir: PathBuf,
    /// the Strata checkout (holds `serve/server.py`); None = not found
    pub root: Option<PathBuf>,
    pub python: String,
    /// this tool's own state: pid records and the logs of servers it started
    pub state_dir: PathBuf,
    /// the server log the Logs view and `logs` read
    pub log: PathBuf,
    pub server_args: Vec<String>,
    pub server_env: Vec<(String, String)>,
    /// the settings file read (for `doctor`-style output)
    pub settings_file: PathBuf,
}

pub const DEFAULT_PORT: u16 = 8095; // serve/server.py's own default

fn home() -> PathBuf {
    env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."))
}

fn xdg(var: &str, fallback: &str) -> PathBuf {
    match env::var_os(var) {
        Some(v) if !v.is_empty() => PathBuf::from(v),
        _ => home().join(fallback),
    }
}

/// `~/x` -> `$HOME/x`; relative paths stay relative.
pub fn expand(p: &Path) -> PathBuf {
    match p.to_str() {
        Some(s) if s == "~" => home(),
        Some(s) if s.starts_with("~/") => home().join(&s[2..]),
        _ => p.to_path_buf(),
    }
}

fn env_str(name: &str) -> Option<String> {
    env::var(name).ok().filter(|v| !v.is_empty())
}

/// The checkout: walk up from the binary (it is built inside the checkout) looking for `serve/server.py`.
fn find_root() -> Option<PathBuf> {
    let exe = env::current_exe().ok()?.canonicalize().ok()?;
    exe.ancestors().find(|d| d.join("serve/server.py").is_file()).map(Path::to_path_buf)
}

impl Settings {
    pub fn resolve(o: &Overrides) -> anyhow::Result<Self> {
        let own_dir = xdg("XDG_CONFIG_HOME", ".config").join("intelinference");
        let settings_file = env_str("INTELINFERENCE_SETTINGS").map(PathBuf::from).unwrap_or_else(|| own_dir.join("settings.json"));
        let file: File = match fs::read_to_string(&settings_file) {
            Ok(text) => serde_json::from_str(&text)
                .map_err(|e| anyhow::anyhow!("{}: {e}", settings_file.display()))?,
            Err(_) => File::default(),
        };
        let state_dir = xdg("XDG_STATE_HOME", ".local/state").join("intelinference");

        let host = o.host.clone().or_else(|| env_str("INTELINFERENCE_HOST")).or(file.host).unwrap_or_else(|| "127.0.0.1".into());
        let port = match o.port {
            Some(p) => p,
            None => match env_str("INTELINFERENCE_PORT") {
                Some(p) => p.parse().map_err(|_| anyhow::anyhow!("INTELINFERENCE_PORT={p} is not a port number"))?,
                None => file.port.unwrap_or(DEFAULT_PORT),
            },
        };
        let api_key = o.api_key.clone().or_else(|| env_str("INTELINFERENCE_API_KEY")).or(file.api_key).filter(|k| !k.is_empty());
        let config_dir = o
            .config_dir
            .clone()
            .or_else(|| env_str("INTELINFERENCE_CONFIG_DIR").map(PathBuf::from))
            .or(file.config_dir)
            .map(|p| expand(&p))
            .unwrap_or_else(|| xdg("XDG_CONFIG_HOME", ".config").join("strata"));
        let root = o
            .root
            .clone()
            .or_else(|| env_str("INTELINFERENCE_ROOT").map(PathBuf::from))
            .or(file.root)
            .map(|p| expand(&p))
            .or_else(find_root);
        let python = o.python.clone().or_else(|| env_str("INTELINFERENCE_PYTHON")).or(file.python).unwrap_or_else(|| {
            // a distribution ships its own runtime beside the checkout; else the python3 on PATH
            root.as_ref()
                .map(|r| r.join("python/bin/python3"))
                .filter(|p| p.is_file())
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "python3".into())
        });
        let log = o
            .log
            .clone()
            .or_else(|| env_str("INTELINFERENCE_LOG").map(PathBuf::from))
            .or(file.log)
            .map(|p| expand(&p))
            .unwrap_or_else(|| state_dir.join(format!("server-{port}.log")));
        Ok(Self {
            host,
            port,
            api_key,
            config_dir,
            root,
            python,
            state_dir,
            log,
            server_args: file.server_args,
            server_env: file.server_env.into_iter().collect(),
            settings_file,
        })
    }

    pub fn base_url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }
}
