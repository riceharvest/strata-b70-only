# Security

## Reporting a vulnerability

Please report anything that should not be public until it is fixed **privately**. Reach the maintainer first on X as
[@imdariotoo](https://x.com/imdariotoo) (a direct message), or by mail at [dev@darioa.nl](mailto:dev@darioa.nl). GitHub's
private vulnerability reporting for `riceharvest/strata-b70-only` also works where it is enabled: the repository's
**Security** tab, **Report a vulnerability**
([direct link](https://github.com/riceharvest/strata-b70-only/security/advisories/new)). Say what you found, how to
reproduce it (the request, the config keys involved, the commit or version), and what an attacker gains. I answer
there, fix it on branch `b70`, and credit you unless you ask me not to.

This fork is maintained by one person on one B70, not by a security team, and upstream's advisory process only
covers `Niko1221/Strata`. Send anything that only happens on a B70 to this repository; upstream cannot reproduce it.

Ordinary hardening ideas and findings that are safe to discuss in public are welcome as an
[issue](https://github.com/riceharvest/strata-b70-only/issues) or as a pull request, which I review like my own
code. (Upstream keeps its own audit-findings thread, #544, in its tracker.)

Supported: the head of branch `b70`. The fork's one release tag so far is `b70-2026.11.11`; a fix goes into the branch, not into older
commits or tags, so build from the head of `b70` to have it.

## What the server exposes

Strata runs a model on your PC and serves it over HTTP (`serve/server.py`). Out of the box it is reachable from
this PC only. The details and every setting are in [docs/DETAILS.md](docs/DETAILS.md) ("From other devices",
"Host names", "Web pages without an API key", "Tools from MCP servers").

- **Where it listens.** `127.0.0.1` by default. `--host 0.0.0.0` (or `"host"` in `strata-<model>.json`) opens it to
  your network, and the server then warns when no API key is set.
- **API key.** `"api_key"` in the run config (or `STRATA_API_KEY`) is required on `/v1/*` and on every endpoint
  that shows the model's state, requests or answers (`/status`, `/metrics`, `/metrics/prometheus`, `/settings`,
  `/mcp`, `/props`, `/slots`, `/api/requests`, `/config`) and on every `POST`. It is compared in constant time.
  Set one before you open the
  server to your network or put a tunnel in front of it.
- **Host check (DNS rebinding, 0.1.38).** Without an API key the server answers only requests whose `Host` is a name
  it knows (`localhost`, an IP address, the address it listens on and, when it listens beyond this PC, this PC's
  name). Others get 403. `"allowed_hosts"` adds names; with an API key the key decides.
- **Origin check (0.1.38).** Without an API key, a browser `POST` to `/v1/*` from another web site's page (it carries
  an `Origin` header) gets 403 unless that origin is Strata's own page, `localhost`, an allowed host, or listed in
  `"trusted_origins"` / `"cors_origins"`. Changing settings (`/settings`, and `/config`: the few run config keys the
  page's Model settings may change - never the network, key, MCP or program keys), `/load` and `/unload` and the
  MCP tools are accepted only as JSON from Strata's own page (or `"trusted_origins"`), so a page elsewhere cannot
  change settings or run tools.
- **CORS** is off unless `"cors_origins"` lists origins, and then only for `/v1/*`.
- **MCP tools** are opt-in: only the servers you put in the run config, only for requests from Strata's own page
  that ask for them. They run with your user's rights, and the model decides when to call them.
- **The request monitor** (`/api-monitor`, which keeps the last prompts and answers in memory) is off unless
  `"api_monitor": true` is set. Apart from it, the server keeps in its process memory the text and tokens of its
  last few prompts, whatever `"api_monitor"` says, to encode the next turn faster; no endpoint shows them, and they
  are gone when the server stops.

## The Terminal

This fork ships the **Terminal** (`strata`) and the **engine** that runs the model. Upstream's desktop app and its Windows and macOS
packages are not part of this fork (`apps/` and `packaging/` were removed), so what upstream's policy says about the app does not
apply here. The section above is about the engine's HTTP server. This is what the Terminal adds.

- **The Terminal and the engine talk through an owner channel.** One owner process per installation starts and ends the engine, and
  a second client attaches to it. On Linux that is a private Unix socket in a folder only you can open
  (`strata_control/owner_transport.py`).
- **Starting a model and serving never download anything.** Downloads and updates are one explicit step: `strata update`. Models and
  saved tuning are never replaced by an update.
- **Updates come from a signed feed.** `strata update` reads upstream's release feed (`DEFAULT_FEED` in `strata_cli/update_cmd.py`,
  `strata-feed.json` and `strata-feed.json.sig`, Ed25519), so for the engine packages upstream publishes, trust rests on
  `Niko1221/Strata`, not on this fork. An installed Strata checks the signature against the public key it carries
  (`strata_control/update_keys.json`), refuses a feed older than the last one it accepted (a replayed old feed) or expired, and
  checks every engine package against its SHA-256, its signed manifest and the card in your PC before it test-starts it and
  switches to it. Any failure keeps the engine you had. The private keys are kept offline by the upstream owner. This fork's own
  engine is built from source (README, "Install") and does not come from that feed.

### How far to trust `curl | sh` and `irm | iex`

These are upstream's installers, from `Niko1221/Strata` releases; this fork's own install is a source build and does not use them.
Upstream's Terminal installs with one line:

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

Piping a script into a shell runs whatever the server sends, so it deserves the same care as running any downloaded program.
What the script does is short and written at its top: it fetches the Terminal archive named in the release's `install-manifest.txt`,
checks its SHA-256, unpacks it into your home folder (no administrator rights, nothing outside your home), checks the release's signed feed
with the Terminal it just unpacked and removes it again if that fails, and puts `strata` on your PATH.

Those checks have a limit. The checksum comes from the same release as the archive, and the public key that checks the feed arrives inside
that archive, so for the first install all trust rests on HTTPS and on the `Niko1221/Strata` GitHub release itself. The checks catch a
damaged or half-finished download and a feed that was not signed by the release key; they do not protect you if the release has been replaced
by someone who controls it. Later updates are checked against the key already on your PC, which is a stronger position.

If that is not enough: save the script (`curl -fsSLO https://github.com/Niko1221/Strata/releases/latest/download/install.sh`), read it, and
run it from the file; or build from source, which is what this fork's README describes, and check the code you build.

Strata has not had an outside security audit yet.
What Strata downloads, when it uses the internet, what stays on your PC and how downloads are checked is written up in [docs/SAFETY.md](docs/SAFETY.md).
