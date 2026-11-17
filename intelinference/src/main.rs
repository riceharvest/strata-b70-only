//! intelinference: the terminal front end for the Strata B70 engine.
//!
//! `intelinference` alone opens the live screen; the subcommands do one thing each and exit.

mod api;
mod cmd;
mod fmt;
mod server;
mod settings;
mod ui;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "intelinference", version, about = "Run and watch the Strata engine on an Intel Arc B70", long_about = None)]
struct Cli {
    /// server address (also INTELINFERENCE_HOST)
    #[arg(long, global = true)]
    host: Option<String>,
    /// server port (also INTELINFERENCE_PORT; serve/server.py's default is 8095)
    #[arg(long, short, global = true)]
    port: Option<u16>,
    /// API key, if the server asks for one (also INTELINFERENCE_API_KEY)
    #[arg(long, global = true, hide_env_values = true)]
    api_key: Option<String>,
    /// folder of run configs (also INTELINFERENCE_CONFIG_DIR; default ~/.config/strata)
    #[arg(long, global = true)]
    config_dir: Option<PathBuf>,
    /// the Strata checkout (also INTELINFERENCE_ROOT; default: the checkout this binary was built in)
    #[arg(long, global = true)]
    root: Option<PathBuf>,
    /// Python that runs serve/server.py (also INTELINFERENCE_PYTHON; default python3)
    #[arg(long, global = true)]
    python: Option<String>,
    /// server log file (also INTELINFERENCE_LOG)
    #[arg(long, global = true)]
    log: Option<PathBuf>,
    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// the live screen (the default)
    Ui,
    /// start the server with a run config, in the background
    Up {
        /// run config: file name, name without .json, model name or a path (default: the last one started)
        config: Option<String>,
        /// stay in this terminal and print the server's output (Ctrl+C stops it)
        #[arg(long)]
        fg: bool,
        /// return at once instead of waiting until the model is loaded
        #[arg(long)]
        no_wait: bool,
    },
    /// stop the server (it parks conversations and ends the engine cleanly)
    Down {
        /// kill it if it has not stopped in time
        #[arg(long)]
        force: bool,
        /// seconds to wait for a clean stop
        #[arg(long, default_value_t = 90)]
        wait: u64,
    },
    /// what the server is doing, in one screen of text
    Status {
        /// the raw /metrics JSON
        #[arg(long)]
        json: bool,
    },
    /// the run configs you can start or switch to
    #[command(alias = "ls")]
    Models,
    /// switch the running server to another run config (or start it with that one)
    Use {
        config: String,
        /// also when a request is running (it is cut off)
        #[arg(long)]
        force: bool,
    },
    /// restart the server with its run config read again
    Restart {
        #[arg(long)]
        force: bool,
    },
    /// load the model now
    Load,
    /// give the GPU back (the server keeps running; the next request loads again)
    Unload,
    /// the server log
    Logs {
        /// keep printing new lines
        #[arg(short, long)]
        follow: bool,
        /// lines from the end
        #[arg(short = 'n', long, default_value_t = 40)]
        lines: usize,
    },
    /// ask the model something and stream the answer
    Ask {
        prompt: Vec<String>,
        #[arg(long)]
        max_tokens: Option<u32>,
        /// print the thinking too
        #[arg(long)]
        think: bool,
    },
    /// where this tool looks for things (settings, configs, checkout, logs)
    Where,
}

fn main() {
    let cli = Cli::parse();
    let o = settings::Overrides {
        host: cli.host,
        port: cli.port,
        api_key: cli.api_key,
        config_dir: cli.config_dir,
        root: cli.root,
        python: cli.python,
        log: cli.log,
    };
    let s = match settings::Settings::resolve(&o) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("intelinference: {e:#}");
            std::process::exit(2);
        }
    };
    let r = match cli.command.unwrap_or(Cmd::Ui) {
        Cmd::Ui => ui::run(s),
        Cmd::Up { config, fg, no_wait } => cmd::up(&s, config.as_deref(), fg, !no_wait),
        Cmd::Down { force, wait } => cmd::down(&s, force, wait),
        Cmd::Status { json } => cmd::status(&s, json),
        Cmd::Models => cmd::models(&s),
        Cmd::Use { config, force } => cmd::use_config(&s, &config, force),
        Cmd::Restart { force } => cmd::restart(&s, force),
        Cmd::Load => cmd::load(&s),
        Cmd::Unload => cmd::unload(&s),
        Cmd::Logs { follow, lines } => cmd::logs(&s, follow, lines),
        Cmd::Ask { prompt, max_tokens, think } => cmd::ask(&s, &prompt.join(" "), max_tokens, think),
        Cmd::Where => cmd::where_(&s),
    };
    match r {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("intelinference: {e:#}");
            std::process::exit(1);
        }
    }
}
