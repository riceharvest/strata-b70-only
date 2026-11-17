//! The one-shot subcommands.

use std::{
    fs,
    io::{self, BufRead, Read, Seek, SeekFrom, Write},
    path::Path,
    thread,
    time::{Duration, Instant},
};

use anyhow::{Result, anyhow, bail};
use serde_json::json;

use crate::{
    api::Api,
    fmt::{self, Snap},
    server::{self, RunConfig},
    settings::Settings,
};

// ---- terminal colour, only on a terminal and without NO_COLOR ----

fn color() -> bool {
    std::env::var_os("NO_COLOR").is_none() && unsafe { libc::isatty(1) } == 1
}

fn paint(code: &str, text: &str) -> String {
    if color() { format!("\x1b[{code}m{text}\x1b[0m") } else { text.to_string() }
}

fn bold(t: &str) -> String {
    paint("1", t)
}
fn dim(t: &str) -> String {
    paint("2", t)
}
fn blue(t: &str) -> String {
    paint("38;2;0;199;253", t)
}
fn green(t: &str) -> String {
    paint("32", t)
}
fn red(t: &str) -> String {
    paint("31", t)
}

fn last_config_file(s: &Settings) -> std::path::PathBuf {
    s.state_dir.join("last-config")
}

fn pick_config(s: &Settings, name: Option<&str>) -> Result<RunConfig> {
    if let Some(n) = name {
        return server::find_config(s, n);
    }
    if let Ok(p) = fs::read_to_string(last_config_file(s))
        && let Some(c) = server::read_config(Path::new(p.trim()))
    {
        return Ok(c);
    }
    let all = server::list_configs(&s.config_dir);
    match all.len() {
        1 => Ok(all[0].clone()),
        0 => bail!("no run configs in {} (write one with setup, or point --config-dir at yours)", s.config_dir.display()),
        _ => bail!(
            "which run config? {} (intelinference up <name>)",
            all.iter().map(|c| c.file.trim_end_matches(".json").to_string()).collect::<Vec<_>>().join(", ")
        ),
    }
}

fn tail_lines(path: &Path, n: usize) -> Vec<String> {
    let Ok(mut f) = fs::File::open(path) else { return vec![] };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let start = len.saturating_sub(256 * 1024);
    f.seek(SeekFrom::Start(start)).ok();
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).ok();
    let text = String::from_utf8_lossy(&buf);
    let lines: Vec<String> = text.lines().map(str::to_string).collect();
    let skip = lines.len().saturating_sub(n);
    lines[skip..].to_vec()
}

/// Wait until /health says loaded; `pid` dying ends the wait. Prints a one-line progress.
pub fn wait_loaded(s: &Settings, api: &Api, pid: Option<i32>, limit: Duration) -> Result<f64> {
    let t0 = Instant::now();
    let spin = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    let mut i = 0;
    loop {
        if let Ok(h) = api.health()
            && h["loaded"].as_bool() == Some(true)
        {
            eprint!("\r\x1b[2K");
            return Ok(t0.elapsed().as_secs_f64());
        }
        if let Some(p) = pid
            && !server::alive(p)
        {
            eprint!("\r\x1b[2K");
            for l in tail_lines(&s.log, 15) {
                eprintln!("  {}", dim(&l));
            }
            bail!("the server stopped while starting (log: {})", s.log.display());
        }
        if t0.elapsed() > limit {
            eprintln!();
            bail!("not loaded after {} s; it may still be loading (intelinference logs -f)", limit.as_secs());
        }
        let last = tail_lines(&s.log, 1).pop().unwrap_or_default();
        let last: String = last.chars().take(90).collect();
        eprint!("\r\x1b[2K{} loading {:>4.0} s  {}", blue(&spin[i % spin.len()].to_string()), t0.elapsed().as_secs_f64(), dim(&last));
        io::stderr().flush().ok();
        i += 1;
        thread::sleep(Duration::from_millis(250));
    }
}

pub fn up(s: &Settings, name: Option<&str>, fg: bool, wait: bool) -> Result<i32> {
    let cfg = pick_config(s, name)?;
    fs::create_dir_all(&s.state_dir).ok();
    fs::write(last_config_file(s), cfg.path.display().to_string()).ok();
    if fg {
        println!("{} {} on {}", bold("starting"), cfg.file, s.base_url());
        return server::run_foreground(s, &cfg);
    }
    let pid = server::start_detached(s, &cfg)?;
    println!("{} {} ({}) on {}  pid {}  log {}", bold("started"), cfg.file, cfg.model, s.base_url(), pid, dim(&s.log.display().to_string()));
    if wait {
        let api = Api::new(s);
        let t = wait_loaded(s, &api, Some(pid), Duration::from_secs(900))?;
        println!("{} in {}", green("ready"), fmt::secs(t));
    }
    Ok(0)
}

pub fn down(s: &Settings, force: bool, wait: u64) -> Result<i32> {
    let pid = server::stop(s, Duration::from_secs(wait), force, |t| {
        eprint!("\r\x1b[2Kstopping {t} s");
        io::stderr().flush().ok();
    })?;
    eprint!("\r\x1b[2K");
    println!("{} (pid {pid})", bold("stopped"));
    Ok(0)
}

pub fn status(s: &Settings, raw: bool) -> Result<i32> {
    let api = Api::new(s);
    let m = match api.metrics() {
        Ok(m) => m,
        Err(e) => {
            match server::owner(s) {
                Some(o) => println!("{} port {} is held by pid {} but does not answer: {e}", red("●"), s.port, o.pid),
                None => println!("{} nothing on {}  {}", dim("○"), s.base_url(), dim("(intelinference up <config>)")),
            }
            return Ok(3);
        }
    };
    if raw {
        println!("{}", serde_json::to_string_pretty(&m)?);
        return Ok(0);
    }
    let x = Snap::from(&m);
    let dot = match x.state.as_str() {
        "generating" | "reading" => blue("●"),
        "idle" => green("●"),
        _ => dim("●"),
    };
    println!(
        "{dot} {}  {}  {}",
        bold(&x.model),
        x.state,
        dim(&format!(
            "{}  ctx {}  kv {}  engine {}",
            s.base_url(),
            x.context.map(fmt::tokens).unwrap_or("?".into()),
            x.kv.as_deref().unwrap_or("?"),
            x.version
        ))
    );
    if x.busy() {
        let mut line = String::new();
        if let (Some(r), Some(t)) = (x.prompt_read, x.prompt_total) {
            line += &format!("reading {} / {}  ", fmt::tokens(r), fmt::tokens(t));
        }
        if let Some(g) = x.generated {
            line += &format!("generated {}  ", fmt::tokens(g));
        }
        if let Some(t) = x.tok_s {
            line += &format!("{t:.1} tok/s  ");
        }
        if let Some(p) = &x.phase {
            line += p;
        }
        println!("  {line}");
    }
    let since = x.t_since.map(|t| fmt::secs(x.time - t)).unwrap_or_default();
    println!(
        "  this run ({since}): {} requests, {} prompt ({} reused), {} out",
        x.t_requests,
        fmt::tokens(x.t_prompt),
        x.reuse_pct().map(|p| format!("{p:.1}%")).unwrap_or("-".into()),
        fmt::tokens(x.t_output)
    );
    println!(
        "  speeds: decode {} tok/s, read {} tok/s, drafts kept {}",
        x.decode_mean().map(|v| format!("{v:.1}")).unwrap_or("-".into()),
        x.read_mean().map(|v| format!("{v:.0}")).unwrap_or("-".into()),
        x.draft_pct().map(|p| format!("{p:.0}%")).unwrap_or("-".into())
    );
    let mut sys = vec![];
    if let (Some(u), Some(t)) = (x.ram_used, x.ram_total) {
        sys.push(format!("RAM {} / {}", fmt::gib(u), fmt::gib(t)));
    }
    if let Some(v) = x.vram_free_mib {
        sys.push(format!("VRAM free {}", fmt::mib(v)));
    }
    if let Some(c) = x.cpu {
        sys.push(format!("CPU {c:.0}%"));
    }
    if !sys.is_empty() {
        println!("  {}", sys.join("  "));
    }
    if !x.threads.is_empty() {
        println!("  agents: {}", x.threads.iter().map(|t| format!("{} {} tok", t.state, fmt::tokens(t.tokens))).collect::<Vec<_>>().join(", "));
    }
    Ok(0)
}

pub fn models(s: &Settings) -> Result<i32> {
    let all = server::list_configs(&s.config_dir);
    if all.is_empty() {
        println!("no run configs in {}", s.config_dir.display());
        return Ok(1);
    }
    let api = Api::new(s);
    let current = api.config().ok().and_then(|c| c["file"].as_str().map(str::to_string));
    let loaded = api.health().ok().and_then(|h| h["loaded"].as_bool());
    println!("{}", dim(&format!("{}", s.config_dir.display())));
    for c in &all {
        let on = current.as_deref() == Some(c.file.as_str());
        let mark = if on { if loaded == Some(true) { green("●") } else { blue("◐") } } else { " ".into() };
        let name = c.file.trim_end_matches(".json");
        let detail = [
            Some(c.model.clone()),
            c.weights.clone(),
            c.context.map(|x| format!("ctx {}", fmt::tokens(x as f64))),
            c.kv.as_ref().map(|k| format!("kv {k}")),
            c.batch.map(|b| format!("batch {b}")),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join("  ");
        println!("{mark} {:<22} {}", if on { bold(name) } else { name.to_string() }, dim(&detail));
    }
    Ok(0)
}

pub fn use_config(s: &Settings, name: &str, force: bool) -> Result<i32> {
    let target = server::find_config(s, name)?;
    let api = Api::new(s);
    let Ok(cur) = api.config() else {
        // nothing running: start it
        return up(s, Some(&target.path.display().to_string()), false, true);
    };
    if cur["file"].as_str() == Some(target.file.as_str()) {
        println!("{} already runs {}", s.base_url(), target.file);
        return Ok(0);
    }
    let switchable = cur["models"].as_array().is_some_and(|a| a.iter().any(|m| m["file"].as_str() == Some(target.file.as_str())));
    if !switchable {
        bail!(
            "the running server switches only between the run configs in its own folder; {} is not there. Stop it (intelinference down) and start this one (intelinference up {})",
            target.file,
            target.path.display()
        );
    }
    api.restart(Some(&target.file), force)?;
    println!("{} to {} ({})", bold("switching"), target.file, target.model);
    fs::write(last_config_file(s), target.path.display().to_string()).ok();
    thread::sleep(Duration::from_millis(800));
    let t = wait_loaded(s, &api, None, Duration::from_secs(900))?;
    println!("{} in {}", green("ready"), fmt::secs(t));
    Ok(0)
}

pub fn restart(s: &Settings, force: bool) -> Result<i32> {
    let api = Api::new(s);
    api.restart(None, force)?;
    println!("{}", bold("restarting"));
    thread::sleep(Duration::from_millis(800));
    let t = wait_loaded(s, &api, None, Duration::from_secs(900))?;
    println!("{} in {}", green("ready"), fmt::secs(t));
    Ok(0)
}

pub fn load(s: &Settings) -> Result<i32> {
    let t0 = Instant::now();
    Api::new(s).load()?;
    println!("{} in {}", green("loaded"), fmt::secs(t0.elapsed().as_secs_f64()));
    Ok(0)
}

pub fn unload(s: &Settings) -> Result<i32> {
    let r = Api::new(s).unload()?;
    println!("{}", bold(&r));
    Ok(0)
}

pub fn logs(s: &Settings, follow: bool, n: usize) -> Result<i32> {
    if !s.log.is_file() {
        bail!("no log at {} (set --log, INTELINFERENCE_LOG or \"log\" in settings.json)", s.log.display());
    }
    for l in tail_lines(&s.log, n) {
        println!("{l}");
    }
    if !follow {
        return Ok(0);
    }
    let mut f = fs::File::open(&s.log)?;
    let mut pos = f.seek(SeekFrom::End(0))?;
    loop {
        thread::sleep(Duration::from_millis(300));
        let len = fs::metadata(&s.log).map(|m| m.len()).unwrap_or(0);
        if len < pos {
            f = fs::File::open(&s.log)?; // rotated or truncated
            pos = 0;
        }
        if len > pos {
            f.seek(SeekFrom::Start(pos))?;
            let mut r = io::BufReader::new(&f);
            let mut line = String::new();
            while r.read_line(&mut line)? > 0 {
                print!("{line}");
                line.clear();
            }
            pos = len;
            io::stdout().flush().ok();
        }
    }
}

pub fn ask(s: &Settings, prompt: &str, max_tokens: Option<u32>, think: bool) -> Result<i32> {
    if prompt.trim().is_empty() {
        bail!("ask what? intelinference ask \"...\"");
    }
    let api = Api::new(s);
    let t0 = Instant::now();
    let mut first: Option<Instant> = None;
    let mut in_reasoning = false;
    let mut out = io::stdout();
    let usage = api
        .chat_stream(&json!([{ "role": "user", "content": prompt }]), max_tokens, |kind, text| {
            first.get_or_insert_with(Instant::now);
            if kind == "reasoning" {
                if think {
                    in_reasoning = true;
                    print!("{}", dim(text));
                }
            } else {
                if in_reasoning {
                    println!("\n");
                    in_reasoning = false;
                }
                print!("{text}");
            }
            out.flush().ok();
        })
        .map_err(|e| anyhow!("{e}"))?;
    println!();
    let out_tok = usage["completion_tokens"].as_f64();
    let wall = first.map(|f| f.elapsed().as_secs_f64());
    let ttft = first.map(|f| (f - t0).as_secs_f64());
    let mut tail = vec![];
    if let Some(t) = ttft {
        tail.push(format!("first token {}", fmt::secs(t)));
    }
    if let (Some(n), Some(w)) = (out_tok, wall)
        && w > 0.0
    {
        tail.push(format!("{} tokens, {:.1} tok/s", n, n / w));
    }
    if !tail.is_empty() {
        eprintln!("{}", dim(&tail.join("  ")));
    }
    Ok(0)
}

pub fn where_(s: &Settings) -> Result<i32> {
    let row = |k: &str, v: String| println!("{:<12} {v}", dim(k));
    row("server", s.base_url());
    row("settings", format!("{}{}", s.settings_file.display(), if s.settings_file.is_file() { "" } else { "  (none)" }));
    row("configs", s.config_dir.display().to_string());
    row("checkout", s.root.as_ref().map(|r| r.display().to_string()).unwrap_or("(not found)".into()));
    row("python", s.python.clone());
    row("log", s.log.display().to_string());
    row("state", s.state_dir.display().to_string());
    if !s.server_args.is_empty() {
        row("server args", s.server_args.join(" "));
    }
    match server::owner(s) {
        Some(o) => row("running", format!("pid {}{}", o.pid, if o.ours { "  (started here)" } else { "" })),
        None => row("running", "nothing on the port".into()),
    }
    Ok(0)
}
