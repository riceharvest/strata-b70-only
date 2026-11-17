//! Run configs on disk, and starting and stopping `serve/server.py`.

use std::{
    fs,
    io::Write,
    os::unix::process::CommandExt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};

use crate::settings::Settings;

/// One run config: a `*.json` file with `"engine": "strata"` and an `"args"` list.
#[derive(Debug, Clone)]
pub struct RunConfig {
    pub path: PathBuf,
    pub file: String,
    pub model: String,
    pub arch: Option<String>,
    pub weights: Option<String>,
    pub context: Option<u64>,
    pub kv: Option<String>,
    pub batch: Option<u64>,
}

fn arg_after<'a>(args: &'a [Value], flag: &str) -> Option<&'a str> {
    args.iter().position(|a| a.as_str() == Some(flag)).and_then(|i| args.get(i + 1)).and_then(Value::as_str)
}

pub fn read_config(path: &Path) -> Option<RunConfig> {
    let text = fs::read_to_string(path).ok()?;
    let v: Value = serde_json::from_str(text.trim_start_matches('\u{feff}')).ok()?;
    if v["engine"].as_str() != Some("strata") {
        return None;
    }
    let args = v["args"].as_array()?;
    let mut all: Vec<Value> = args.clone();
    if let Some(extra) = v["user_args"].as_array() {
        all.extend(extra.iter().cloned());
    }
    let weights = arg_after(&all, "--native")
        .or_else(|| arg_after(&all, "--gguf"))
        .or_else(|| arg_after(&all, "--pack"))
        .map(|p| Path::new(p).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| p.to_string()));
    Some(RunConfig {
        file: path.file_name()?.to_string_lossy().into_owned(),
        path: path.to_path_buf(),
        model: v["model_name"].as_str().map(str::to_string).unwrap_or_else(|| "qwen3.8-flash-next".into()),
        arch: v["arch"].as_str().map(str::to_string),
        weights,
        context: arg_after(&all, "--max-context").and_then(|c| c.parse().ok()),
        kv: arg_after(&all, "--kv").map(str::to_string),
        batch: arg_after(&all, "--batch").and_then(|c| c.parse().ok()),
    })
}

pub fn list_configs(dir: &Path) -> Vec<RunConfig> {
    let Ok(rd) = fs::read_dir(dir) else { return vec![] };
    let mut out: Vec<RunConfig> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .filter_map(|p| read_config(&p))
        .collect();
    out.sort_by(|a, b| a.file.cmp(&b.file));
    out
}

/// A config by file name, stem or path.
pub fn find_config(s: &Settings, name: &str) -> Result<RunConfig> {
    let as_path = crate::settings::expand(Path::new(name));
    if as_path.is_file() {
        return read_config(&as_path.canonicalize()?).ok_or_else(|| anyhow!("{name}: not a Strata run config"));
    }
    let all = list_configs(&s.config_dir);
    all.iter()
        .find(|c| c.file == name || c.file.strip_suffix(".json") == Some(name) || c.model == name)
        .cloned()
        .ok_or_else(|| {
            let names: Vec<_> = all.iter().map(|c| c.file.as_str()).collect();
            anyhow!("no run config {name:?} in {} (there: {})", s.config_dir.display(), names.join(", "))
        })
}

// ---- processes ----

fn record_path(s: &Settings) -> PathBuf {
    s.state_dir.join(format!("server-{}.json", s.port))
}

pub fn alive(pid: i32) -> bool {
    pid > 0 && unsafe { libc::kill(pid, 0) } == 0
}

fn cmdline(pid: i32) -> String {
    fs::read(format!("/proc/{pid}/cmdline")).map(|b| String::from_utf8_lossy(&b).replace('\0', " ")).unwrap_or_default()
}

/// The process listening on `port` (Linux: /proc/net/tcp* -> socket inode -> /proc/*/fd).
pub fn pid_on_port(port: u16) -> Option<i32> {
    let mut inodes = vec![];
    for table in ["/proc/net/tcp", "/proc/net/tcp6"] {
        let Ok(text) = fs::read_to_string(table) else { continue };
        for line in text.lines().skip(1) {
            let cols: Vec<&str> = line.split_whitespace().collect();
            if cols.len() < 10 || cols[3] != "0A" {
                continue; // 0A = LISTEN
            }
            let Some(p) = cols[1].rsplit(':').next().and_then(|h| u16::from_str_radix(h, 16).ok()) else { continue };
            if p == port {
                inodes.push(format!("socket:[{}]", cols[9]));
            }
        }
    }
    if inodes.is_empty() {
        return None;
    }
    for e in fs::read_dir("/proc").ok()?.flatten() {
        let Some(pid) = e.file_name().to_str().and_then(|n| n.parse::<i32>().ok()) else { continue };
        let Ok(fds) = fs::read_dir(e.path().join("fd")) else { continue };
        for fd in fds.flatten() {
            if let Ok(t) = fs::read_link(fd.path())
                && inodes.iter().any(|i| t.as_os_str() == i.as_str())
            {
                return Some(pid);
            }
        }
    }
    None
}

/// What this tool knows about the server on its port: the pid, and the record if it started it.
pub struct Owner {
    pub pid: i32,
    pub ours: bool,
}

pub fn owner(s: &Settings) -> Option<Owner> {
    if let Ok(text) = fs::read_to_string(record_path(s))
        && let Ok(rec) = serde_json::from_str::<Value>(&text)
        && let Some(pid) = rec["pid"].as_i64().map(|p| p as i32)
        && alive(pid)
        && cmdline(pid).contains("server.py")
    {
        return Some(Owner { pid, ours: true });
    }
    pid_on_port(s.port).map(|pid| Owner { pid, ours: false })
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn command(s: &Settings, cfg: &RunConfig) -> Result<Command> {
    let root = s.root.as_ref().ok_or_else(|| {
        anyhow!("cannot find the Strata checkout (serve/server.py); set --root, INTELINFERENCE_ROOT or \"root\" in {}", s.settings_file.display())
    })?;
    let mut c = Command::new(&s.python);
    c.current_dir(root)
        .arg("serve/server.py")
        .args(["--engine", "strata", "--config"])
        .arg(&cfg.path)
        .args(["--port", &s.port.to_string()]);
    if !matches!(s.host.as_str(), "127.0.0.1" | "localhost") {
        c.args(["--host", &s.host]);
    }
    c.args(&s.server_args);
    for (k, v) in &s.server_env {
        c.env(k, v);
    }
    Ok(c)
}

/// Start the server in the background (its own process group: closing this terminal or the UI does not stop it);
/// output goes to the log. Returns the pid.
pub fn start_detached(s: &Settings, cfg: &RunConfig) -> Result<i32> {
    if let Some(o) = owner(s) {
        bail!("port {} is already taken (pid {}): stop it first or pick another --port", s.port, o.pid);
    }
    fs::create_dir_all(&s.state_dir).with_context(|| format!("creating {}", s.state_dir.display()))?;
    if let Some(d) = s.log.parent() {
        fs::create_dir_all(d).ok();
    }
    let mut log = fs::OpenOptions::new().create(true).append(true).open(&s.log).with_context(|| format!("opening {}", s.log.display()))?;
    writeln!(log, "\n[intelinference] starting {} on port {} at {}", cfg.file, s.port, now()).ok();
    let err = log.try_clone()?;
    let child = command(s, cfg)?
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(err))
        .process_group(0)
        .spawn()
        .with_context(|| format!("running {}", s.python))?;
    let pid = child.id() as i32;
    let rec = json!({ "pid": pid, "config": cfg.path, "file": cfg.file, "model": cfg.model, "port": s.port, "log": s.log, "started": now() });
    fs::write(record_path(s), serde_json::to_vec_pretty(&rec)?)?;
    // reaped by a thread while this process lives (the UI); after this process exits, init takes it over
    let mut child = child;
    thread::spawn(move || child.wait());
    Ok(pid)
}

/// Run the server in this terminal until it ends; Ctrl+C reaches it as well.
pub fn run_foreground(s: &Settings, cfg: &RunConfig) -> Result<i32> {
    if let Some(o) = owner(s) {
        bail!("port {} is already taken (pid {}): stop it first or pick another --port", s.port, o.pid);
    }
    let status = command(s, cfg)?.status().with_context(|| format!("running {}", s.python))?;
    Ok(status.code().unwrap_or(1))
}

/// Ask the server to stop (SIGTERM: it ends the engine cleanly) and wait up to `wait`; `force` kills what is left.
pub fn stop(s: &Settings, wait: Duration, force: bool, mut tick: impl FnMut(u64)) -> Result<i32> {
    let o = owner(s).ok_or_else(|| anyhow!("nothing is listening on port {}", s.port))?;
    unsafe { libc::kill(o.pid, libc::SIGTERM) };
    let t0 = Instant::now();
    while alive(o.pid) && t0.elapsed() < wait {
        thread::sleep(Duration::from_millis(250));
        tick(t0.elapsed().as_secs());
    }
    if alive(o.pid) {
        if !force {
            bail!("pid {} is still stopping after {} s (it may be parking conversations); add --force to kill it", o.pid, wait.as_secs());
        }
        unsafe {
            libc::kill(-o.pid, libc::SIGKILL); // its process group when this tool started it
            libc::kill(o.pid, libc::SIGKILL);
        }
    }
    if o.ours {
        fs::remove_file(record_path(s)).ok();
    }
    Ok(o.pid)
}
