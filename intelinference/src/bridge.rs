//! Every command of the Python command line, run through intelinference.
//!
//! The logic that is already tested stays where it is (signed engine updates, model downloads and setup, benchmarks,
//! tuning, presets, saved profiles, recipes): intelinference runs `strata.py <command> --output json` (or `jsonl`
//! where the command streams), shows its progress as it goes and draws the result. Commands that open an editor or
//! ask questions run with the terminal handed over.

use std::{
    io::{self, BufRead, BufReader, IsTerminal, Write},
    process::{Command, Stdio},
};

use anyhow::{Context, Result, anyhow};
use clap::{Arg, ArgAction, Command as ClapCmd};
use serde_json::Value;

use crate::{render, settings::Settings};

/// How a command's output is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Out {
    /// one JSON object on stdout at the end
    Json,
    /// JSON events while it runs, then the result
    Jsonl,
    /// the terminal is handed over (editors, questions, a foreground server)
    Tty,
}

/// (path, what it does, output). The Python `strata` tree, generated from its argparse parser.
/// `logs` is named `activity` here (intelinference's `logs` is the server log); `status` and `monitor` are native.
pub const TREE: &[(&str, &str, Out)] = &[
    ("serve", "start the server in this terminal from a run config or saved profile (Ctrl+C stops it)", Out::Tty),
    ("profile create", "save a named profile from an installed run config", Out::Json),
    ("profile list", "list saved profiles", Out::Json),
    ("profile show", "show a profile and optional effective settings", Out::Json),
    ("profile validate", "check the profile and referenced run config", Out::Json),
    ("profile set", "set typed serving overrides, e.g. parallel=8", Out::Json),
    ("profile use-config", "point a saved profile at another run config file", Out::Json),
    ("profile edit", "edit a temporary JSON copy in VISUAL or EDITOR", Out::Tty),
    ("profile history", "list saved profile revisions", Out::Json),
    ("profile rollback", "restore a saved profile revision", Out::Json),
    ("repair-access", "repair the permissions of Strata's own folders and files", Out::Json),
    ("doctor", "read-only check of installed engine components", Out::Json),
    ("components verify", "check required files for an installed profile", Out::Json),
    ("components plan", "read a component-package manifest and say whether it may be used here", Out::Json),
    ("components backends", "the OS and backend combinations that are committed to, and what is not", Out::Json),
    ("components engine", "which engine package in the local component folder this PC would start (never downloads)", Out::Json),
    ("components update", "the same as `update` (engine and Control)", Out::Jsonl),
    ("components rollback", "return to the previous engine version the last update kept", Out::Jsonl),
    ("components install-file", "install an engine package you downloaded yourself (no network; the signature is checked)", Out::Jsonl),
    ("update", "download and install the engine for this PC from the signed feed; switches only after a test start", Out::Jsonl),
    ("models list", "list local installed-model config candidates", Out::Json),
    ("models inspect", "inspect a local model run config and its components", Out::Json),
    ("models import", "preview an existing GGUF file set without adopting it", Out::Json),
    ("models move", "move the model files (models, packs, mtp) to another folder: copied, checked, then the old copies removed", Out::Jsonl),
    ("drafting plan", "say what fetching the DFlash2 drafter would download; nothing is changed", Out::Jsonl),
    ("drafting fetch", "download the DFlash2 drafter (pinned, checked) and pack it for the engine", Out::Jsonl),
    ("engine calibrate", "measure, review and apply engine tuning", Out::Json),
    ("install", "the explicit installer: engine, models, Python packages (asks questions)", Out::Tty),
    ("runtime start", "start a saved profile as a background service (it keeps serving after this terminal closes)", Out::Json),
    ("runtime status", "inspect an owned local instance", Out::Json),
    ("runtime verify", "recheck the owned API and one-token generation", Out::Json),
    ("runtime stop", "drain and stop an owned local instance", Out::Json),
    ("runtime cleanup", "end servers and engines Strata started whose owner is gone", Out::Json),
    ("runtime restart", "drain and rerun an owned saved profile", Out::Json),
    ("ops init", "create the recipe store for a profile", Out::Json),
    ("ops list", "list the recipes of a profile", Out::Json),
    ("ops create", "create a recipe", Out::Json),
    ("ops show", "show a recipe", Out::Json),
    ("ops edit", "edit a recipe in VISUAL or EDITOR", Out::Tty),
    ("ops set", "set fields of a recipe", Out::Json),
    ("ops history", "list a recipe's revisions", Out::Json),
    ("ops rollback", "restore a recipe revision", Out::Json),
    ("ops plan", "what running a recipe would do", Out::Json),
    ("ops run", "run a recipe", Out::Json),
    ("ops runs", "list recent recipe runs", Out::Json),
    ("ops result", "the result of one recipe run", Out::Json),
    ("bench run", "the standard benchmark (docs/BENCHMARK_PROTOCOL.md), or a serving suite", Out::Json),
    ("bench cancel", "stop the exact active saved-profile benchmark", Out::Json),
    ("bench report", "summarize an existing benchmark report", Out::Json),
    ("bench compare", "compare two saved runs with workload checks", Out::Json),
    ("timeskip run", "checkpoint, attempt a revision, check it and decide", Out::Json),
    ("timeskip status", "read a run's durable state and evidence", Out::Json),
    ("timeskip list", "list recorded runs", Out::Json),
    ("timeskip apply", "apply the retained result, refusing if the destination changed", Out::Json),
    ("tune plan", "show exactly what a sweep would search, and why not", Out::Json),
    ("tune run", "run a bounded sweep: measure the baseline and candidates, then report", Out::Json),
    ("tune apply", "apply a reviewed recommendation, after verifying the candidate runs", Out::Json),
    ("tune rollback", "restore the configuration before an apply", Out::Json),
    ("tune status", "read a job's durable state and report", Out::Json),
    ("tune list", "list recorded jobs", Out::Json),
    ("tune auto", "find the fastest settings for this PC and save them as a preset (quick: about 20 minutes)", Out::Json),
    ("presets list", "every saved preset (or those for one model/configuration)", Out::Json),
    ("presets status", "which preset a configuration would start with, and whether it is still valid", Out::Json),
    ("presets show", "one preset in full", Out::Json),
    ("presets apply", "make a preset the active one for its choices (the previous one is kept for undo)", Out::Json),
    ("presets rename", "give a preset another name", Out::Json),
    ("presets delete", "delete a preset", Out::Json),
    ("presets undo", "go back to the preset that was active before (or the standard settings)", Out::Json),
    ("presets export", "write a preset to a file (or print it)", Out::Json),
    ("presets import", "read a preset from a file", Out::Json),
    ("presets effective", "what a start with these choices will actually run", Out::Json),
    ("presets off", "switch the saved tuning off for these choices (it is kept and can be switched on again)", Out::Json),
    ("presets recheck", "a short check (about 2 minutes) that a preset is still as fast", Out::Json),
    ("launch", "set up and start a model with flags (download, config, tune, start); `setup` is the guided screen", Out::Json),
    ("connect", "print (or write) the setup for an app that uses the running server", Out::Json),
    ("activity", "the server's recent request activity, redacted (--follow keeps watching)", Out::Json),
    ("connection create", "save a named server endpoint", Out::Json),
    ("connection show", "show a saved connection without its credential", Out::Json),
    ("connection list", "list saved connections", Out::Json),
    ("connection set", "edit one URL, prefix or credential reference", Out::Json),
    ("config show", "show connection and credential origins", Out::Json),
];

const GROUPS: &[(&str, &str)] = &[
    ("profile", "saved serving profiles: overrides kept outside the run config"),
    ("components", "installed engine packages: check, update, roll back"),
    ("drafting", "the DFlash2 drafter: what it would download, and fetch it"),
    ("engine", "engine calibration"),
    ("runtime", "owned background server instances"),
    ("ops", "typed operator recipes"),
    ("bench", "serving benchmarks with local JSON evidence"),
    ("timeskip", "keep a good result, try a bounded improvement, check it, decide"),
    ("tune", "bounded engine-configuration search"),
    ("presets", "saved tuning presets"),
    ("connection", "saved server endpoints (no credential values stored)"),
    ("config", "effective client connection settings"),
];

fn passthrough_leaf(name: &'static str, about: &'static str) -> ClapCmd {
    ClapCmd::new(name)
        .about(about)
        .disable_help_flag(true)
        .arg(Arg::new("args").num_args(0..).trailing_var_arg(true).allow_hyphen_values(true).action(ArgAction::Append))
}

/// Add the bridged commands to the command line (groups with their leaves, and single commands).
pub fn add_to(mut cmd: ClapCmd) -> ClapCmd {
    for (group, about) in GROUPS {
        let mut g = ClapCmd::new(*group).about(*about).subcommand_required(true).arg_required_else_help(true);
        for (path, leaf_about, _) in TREE {
            if let Some(rest) = path.strip_prefix(group).and_then(|r| r.strip_prefix(' ')) {
                g = g.subcommand(passthrough_leaf(rest, leaf_about));
            }
        }
        cmd = cmd.subcommand(g);
    }
    for (path, about, _) in TREE {
        if !path.contains(' ') {
            cmd = cmd.subcommand(passthrough_leaf(path, about));
        }
    }
    cmd
}

pub fn is_bridged(top: &str) -> bool {
    GROUPS.iter().any(|(g, _)| *g == top) || TREE.iter().any(|(p, _, _)| *p == top)
}

/// The command path and its arguments from clap's matches (`bench run --suite x` -> ["bench","run"], ["--suite","x"]).
pub fn from_matches(top: &str, m: &clap::ArgMatches) -> (Vec<String>, Vec<String>) {
    let mut path = vec![top.to_string()];
    let mut m = m;
    while let Some((name, sub)) = m.subcommand() {
        path.push(name.to_string());
        m = sub;
    }
    let args = m.get_many::<String>("args").map(|v| v.cloned().collect()).unwrap_or_default();
    (path, args)
}

fn out_of(path: &[String]) -> Out {
    let key = path.join(" ");
    TREE.iter().find(|(p, _, _)| *p == key).map(|(_, _, o)| *o).unwrap_or(Out::Json)
}

/// The Python command line's own name for a path.
fn python_path(path: &[String]) -> Vec<String> {
    let mut p = path.to_vec();
    if p.first().map(String::as_str) == Some("activity") {
        p[0] = "logs".into();
    }
    p
}

pub fn base(s: &Settings) -> Result<Command> {
    let root = s.root.as_ref().ok_or_else(|| {
        anyhow!("cannot find the Strata checkout (strata.py); set --root, INTELINFERENCE_ROOT or \"root\" in {}", s.settings_file.display())
    })?;
    let mut c = Command::new(&s.python);
    c.current_dir(root).arg(root.join("strata.py"));
    Ok(c)
}

/// Run a bridged command for the command line. Returns its exit code.
pub fn run(s: &Settings, path: &[String], args: &[String]) -> Result<i32> {
    let mut args = args.to_vec();
    let out = out_of(path);
    let pypath = python_path(path);
    // `launch` without --no-tui would open the old screens: intelinference has its own (`setup`)
    if pypath == ["launch"] && !args.iter().any(|a| a == "--no-tui") && !args.iter().any(|a| a == "-h" || a == "--help") {
        args.push("--no-tui".into());
    }
    // --json: the raw result, for every command (some only take --output json)
    if args.iter().any(|a| a == "--json") {
        args.retain(|a| a != "--json");
        if !args.iter().any(|a| a == "--output" || a.starts_with("--output=")) && out != Out::Tty {
            args.extend(["--output".into(), "json".into()]);
        }
    }
    let wants_help = args.iter().any(|a| a == "-h" || a == "--help");
    let raw = args.windows(2).any(|w| w[0] == "--output" && w[1] != "human") || args.iter().any(|a| a.starts_with("--output="));
    let human = args.windows(2).any(|w| w[0] == "--output" && w[1] == "human") || args.iter().any(|a| a == "--output=human");
    if wants_help || out == Out::Tty || raw || human {
        // the Python command's own output, unchanged (its help names itself `strata`)
        let mut c = base(s)?;
        c.args(&pypath).args(&args);
        if wants_help {
            let o = c.output().context("running strata.py")?;
            let text = String::from_utf8_lossy(&o.stdout).replace("usage: strata ", "usage: intelinference ").replace("strata ", "intelinference ");
            print!("{}", if path.first().map(String::as_str) == Some("activity") { text.replace("intelinference logs", "intelinference activity") } else { text });
            io::stderr().write_all(&o.stderr).ok();
            return Ok(o.status.code().unwrap_or(1));
        }
        let st = c.status().context("running strata.py")?;
        return Ok(st.code().unwrap_or(1));
    }
    // `activity --follow` streams; without it there is one result
    let out = if pypath[0] == "logs" && args.iter().any(|a| a == "--follow" || a == "-f") { Out::Jsonl } else { out };
    let (code, result, _) = run_captured(s, &pypath, &args, out, &mut |ev| render::event_line(ev))?;
    render::finish_progress();
    match result {
        Some(v) => Ok(render::result(&path.join(" "), &v).max(code)),
        None => Ok(code),
    }
}

/// Run a command and collect its JSON result; `on_event` gets each streamed event. Plain text lines on stdout are
/// passed to `on_event` as `{"event": "text", "message": line}`. Returns (exit code, result, every stdout line).
pub fn run_captured(s: &Settings, pypath: &[String], args: &[String], out: Out, on_event: &mut dyn FnMut(&Value)) -> Result<(i32, Option<Value>, Vec<String>)> {
    let mut c = base(s)?;
    c.args(pypath).args(args);
    match out {
        Out::Jsonl => {
            c.args(["--output", "jsonl"]);
        }
        Out::Json => {
            c.args(["--output", "json"]);
        }
        Out::Tty => {}
    }
    c.stdin(if io::stdin().is_terminal() { Stdio::inherit() } else { Stdio::null() }).stdout(Stdio::piped()).stderr(Stdio::inherit());
    let mut child = c.spawn().context("running strata.py")?;
    let stdout = child.stdout.take().expect("piped");
    let mut result: Option<Value> = None;
    let mut lines = vec![];
    for line in BufReader::new(stdout).lines() {
        let line = line?;
        lines.push(line.clone());
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(t) {
            Ok(v) if v.get("event").is_some() && v.get("ok").is_none() => on_event(&v),
            Ok(v) if v.is_object() => result = Some(v),
            _ => on_event(&serde_json::json!({ "event": "text", "message": line })),
        }
    }
    let code = child.wait()?.code().unwrap_or(1);
    if result.is_none() {
        // a pretty-printed object over several lines
        result = serde_json::from_str::<Value>(&lines.join("\n")).ok().filter(Value::is_object);
    }
    Ok((code, result, lines))
}

/// The models setup offers, with what fits this PC (the launch catalog), the PC in one line and the context sizes.
pub fn catalog(s: &Settings) -> Result<Value> {
    const SCRIPT: &str = "import json, dataclasses\nfrom strata_cli.launch.catalog import Catalog\nc = Catalog()\n\
rows = [dict(dataclasses.asdict(r), title=r.title) for r in c.rows()]\n\
print(json.dumps({'computer': c.computer().line(), 'contexts': c.contexts(), 'rows': rows}, default=str))\n";
    let root = s.root.as_ref().ok_or_else(|| anyhow!("cannot find the Strata checkout"))?;
    let o = Command::new(&s.python).current_dir(root).args(["-c", SCRIPT]).output().context("running python")?;
    if !o.status.success() {
        return Err(anyhow!("{}", String::from_utf8_lossy(&o.stderr).lines().last().unwrap_or("the catalog did not load")));
    }
    Ok(serde_json::from_slice(&o.stdout)?)
}
