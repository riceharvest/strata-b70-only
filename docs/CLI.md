# Command-line reference

Regenerate with `COLUMNS=100 python strata.py --help`, then `COLUMNS=100 python strata.py <command> [<subcommand>] --help` for each command named in it; paste the output unchanged. The engine section is the text of `usage()` in `sycl/src/program/generate.cpp` and of the `strata-q36` usage line in `sycl/src/qwen35/q36_engine.dp.cpp`.

Generated on 2026-11-16 from commit `f7450910a`. The blocks below are the real output; nothing is paraphrased. I did not edit them.

Contents: [`strata`](#strata) · [`serve/server.py`](#serveserverpy) · [`setup.py` / `./setup.sh`](#setuppy--setupsh) · [Engine flags](#engine-flags)

## `strata`

`python strata.py --help` (the installed command is `strata`):

```text
usage: strata [-h] [--version]
              {serve,profile,repair-access,doctor,components,update,models,drafting,engine,install,runtime,ops,bench,timeskip,tune,presets,launch,connect,status,monitor,logs,connection,config,code} ...

Strata serving and terminal clients

positional arguments:
  {serve,profile,repair-access,doctor,components,update,models,drafting,engine,install,runtime,ops,bench,timeskip,tune,presets,launch,connect,status,monitor,logs,connection,config,code}
    serve               start the existing Strata inference server in the foreground
    profile             edit saved serving overrides outside setup's run config
    doctor              read-only check of installed engine components
    components          read-only installed component inventory
    update              download and install the engine for this PC (and Control) from the signed
                        feed; switches only after a test start
    models              inspect setup-written local model run configs
    drafting            the DFlash2 drafter file: what it would download, and fetch it
    engine              engine tuning and component actions
    install             the explicit installer: engine, models, Python packages (starting and
                        serving never do this)
    runtime             manage an owned local Strata server instance
    ops                 edit and run typed local operator recipes
    bench               explicit serving benchmark with local JSON evidence
    timeskip            retain a good result, try a bounded improvement, check it and decide
    tune                bounded engine-configuration search with retained evidence
    presets             saved tuning presets: list, apply, rename, delete, undo, re-check
    launch              guided setup, serve and monitor (the screens), or the same with flags
    connect             print (or write) the setup for an app that uses the running server
    status              one read-only telemetry snapshot
    monitor             read-only terminal monitor: seven views (Overview ... Logs, Connect), the
                        same as the one strata launch opens
    logs                bounded redacted request activity (read only)
    connection          save monitor endpoints without storing credential values
    config              inspect effective client connection settings
    code                coding client for the shared Desktop harness

options:
  -h, --help            show this help message and exit
  --version             print the version and the interpreter this terminal runs on
```

### `strata serve`

```text
usage: strata serve [-h] --profile CONFIG.json [--engine {strata,mock}] [--host HOST]
                    [--port PORT] [--parallel PARALLEL] [--batch-groups BATCH_GROUPS]
                    [--max-queue MAX_QUEUE] [--queue-timeout QUEUE_TIMEOUT]
                    [--batch-telemetry | --no-batch-telemetry] [--offline]
                    [--ready-file READY_FILE] [--shutdown-timeout SHUTDOWN_TIMEOUT] [--plan]
                    [--output {human,json,jsonl}] [--json] [--quiet]

options:
  -h, --help            show this help message and exit
  --profile, --config CONFIG.json
  --engine {strata,mock}
                        strata: the engine in the run config (default); mock: a scripted engine
                        for client tests, no GPU and no model
  --host HOST
  --port PORT
  --parallel PARALLEL   requested concurrent engine slots
  --batch-groups BATCH_GROUPS
                        requested pipeline groups (auto or positive integer)
  --max-queue MAX_QUEUE
                        maximum pending inference calls beyond effective slots
  --queue-timeout QUEUE_TIMEOUT
                        seconds to wait for the first engine turn (0.1..600)
  --batch-telemetry, --no-batch-telemetry
                        sample actual engine decode-window rows for the monitor (opt-in)
  --offline             disable remote image fetches and MCP/tool subprocesses
  --ready-file READY_FILE
                        absolute path for a new instance record (use a private directory)
  --shutdown-timeout SHUTDOWN_TIMEOUT
                        seconds to drain owned inference calls on stop (0..300)
  --plan                validate and print the effective launch without starting
  --output {human,json,jsonl}
                        human output, JSON plan, or JSONL foreground lifecycle
  --json                alias for --plan --output json
  --quiet               suppress human informational output
```

(Printed with the system Python: the Strata venv lacks the packages this command imports.)

### `strata profile`

```text
usage: strata profile [-h] {create,list,show,validate,set,use-config,edit,history,rollback} ...

positional arguments:
  {create,list,show,validate,set,use-config,edit,history,rollback}
    create              save a named profile from an installed run config
    list                list saved profiles
    show                show a profile and optional effective settings
    validate            check the profile and referenced run config
    set                 set typed serving overrides, e.g. parallel=8
    use-config          point a saved profile at another run config file
    edit                edit a temporary JSON copy in VISUAL or EDITOR
    history             list saved profile revisions
    rollback            restore a saved profile revision

options:
  -h, --help            show this help message and exit
```

#### `strata profile create`

```text
usage: strata profile create [-h] --from-config CONFIG.json [--output {human,json}] [--json]
                             [--quiet]
                             name

positional arguments:
  name

options:
  -h, --help            show this help message and exit
  --from-config CONFIG.json
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata profile list`

```text
usage: strata profile list [-h] [--output {human,json}] [--json] [--quiet]

options:
  -h, --help            show this help message and exit
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata profile show`

```text
usage: strata profile show [-h] [--effective] [--output {human,json}] [--json] [--quiet] name

positional arguments:
  name

options:
  -h, --help            show this help message and exit
  --effective
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata profile validate`

```text
usage: strata profile validate [-h] [--output {human,json}] [--json] [--quiet] name

positional arguments:
  name

options:
  -h, --help            show this help message and exit
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata profile set`

```text
usage: strata profile set [-h] [--plan] [--output {human,json}] [--json] [--quiet]
                          name assignments [assignments ...]

positional arguments:
  name
  assignments

options:
  -h, --help            show this help message and exit
  --plan                show changes without saving
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata profile use-config`

```text
usage: strata profile use-config [-h] --from-config CONFIG.json [--plan] [--if-revision REVISION]
                                 [--output {human,json}] [--json] [--quiet]
                                 name

positional arguments:
  name

options:
  -h, --help            show this help message and exit
  --from-config CONFIG.json
  --plan                show the source change without saving
  --if-revision REVISION
                        fail if the saved profile changed since preview
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata profile edit`

```text
usage: strata profile edit [-h] [--output {human,json}] [--json] [--quiet] name

positional arguments:
  name

options:
  -h, --help            show this help message and exit
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata profile history`

```text
usage: strata profile history [-h] [--output {human,json}] [--json] [--quiet] name

positional arguments:
  name

options:
  -h, --help            show this help message and exit
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata profile rollback`

```text
usage: strata profile rollback [-h] --to REVISION [--output {human,json}] [--json] [--quiet] name

positional arguments:
  name

options:
  -h, --help            show this help message and exit
  --to REVISION
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

### `strata repair-access`

```text
usage: strata repair-access [-h] [--output {human,json}]

options:
  -h, --help            show this help message and exit
  --output {human,json}
```

### `strata doctor`

```text
usage: strata doctor [-h] [--profile PROFILE] [--components] [--live] [--output {human,json}]
                     [--json] [--quiet]

options:
  -h, --help            show this help message and exit
  --profile PROFILE     saved profile name or run config JSON path
  --components          include component inventory (default)
  --live                query host GPUs and compare selected device indices
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

### `strata components`

```text
usage: strata components [-h] {verify,plan,backends,engine,update,rollback,install-file} ...

positional arguments:
  {verify,plan,backends,engine,update,rollback,install-file}
    verify              check required files for an installed profile
    plan                read a component-package manifest and say whether it may be used here
    backends            the OS and backend combinations that are committed to, and what is not
    engine              which engine package in the local component folder this PC would start
                        (never downloads)
    update              the same as `strata update` (engine and Control)
    rollback            return to the previous engine version the last update kept
    install-file        install an engine package you downloaded yourself (no network; the
                        package's signature is checked)

options:
  -h, --help            show this help message and exit
```

#### `strata components verify`

```text
usage: strata components verify [-h] --profile PROFILE [--live] [--output {human,json}] [--json]
                                [--quiet]

options:
  -h, --help            show this help message and exit
  --profile PROFILE
  --live                query host GPUs and compare selected device indices
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata components plan`

```text
usage: strata components plan [-h] [--backend BACKEND] [--target TARGET] [--output {human,json}]
                              manifest

positional arguments:
  manifest              path to the component manifest (JSON)

options:
  -h, --help            show this help message and exit
  --backend BACKEND     the backend family to check the component against
  --target TARGET       override the platform key, for checking another machine's manifest on this
                        one
  --output {human,json}
```

#### `strata components backends`

```text
usage: strata components backends [-h] [--output {human,json}]

options:
  -h, --help            show this help message and exit
  --output {human,json}
```

#### `strata components engine`

```text
usage: strata components engine [-h] [--backend BACKEND] [--components-dir COMPONENTS_DIR]
                                [--full] [--output {human,json}]

options:
  -h, --help            show this help message and exit
  --backend BACKEND     ask for one backend family instead of detecting it
  --components-dir COMPONENTS_DIR
                        the component folder (default: STRATA_COMPONENTS_DIR or the per-user
                        folder)
  --full                hash every file, not only the entrypoint
  --output {human,json}
```

#### `strata components update`

```text
usage: strata components update [-h] [--feed FEED] [--components-dir COMPONENTS_DIR]
                                [--backend BACKEND] [--wait SECONDS] [--force]
                                [--model-config RUN_CONFIG.json] [--output {human,json,jsonl}]
                                [--check] [--app] [--app-version APP_VERSION] [--no-engine]
                                [--no-control] [--json] [--quiet]

options:
  -h, --help            show this help message and exit
  --feed FEED           where the signed update feed is (URL or folder). Default:
                        $STRATA_UPDATE_FEED, the feed used last, or the project's
  --components-dir COMPONENTS_DIR
                        the component folder (default: the per-user one, or STRATA_COMPONENTS_DIR)
  --backend BACKEND     choose an engine family instead of detecting this PC's GPU
  --wait SECONDS        if an engine is serving, wait up to this long for it to stop instead of
                        refusing
  --force               try a version again that failed its test start or was rolled back from
  --model-config RUN_CONFIG.json
                        the run config of the model to use for the test start (default: an
                        installed model)
  --output {human,json,jsonl}
  --check
  --app
  --app-version APP_VERSION
  --no-engine
  --no-control
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata components rollback`

```text
usage: strata components rollback [-h] [--feed FEED] [--components-dir COMPONENTS_DIR]
                                  [--backend BACKEND] [--wait SECONDS] [--force]
                                  [--model-config RUN_CONFIG.json] [--output {human,json,jsonl}]
                                  [--json] [--quiet]

options:
  -h, --help            show this help message and exit
  --feed FEED           where the signed update feed is (URL or folder). Default:
                        $STRATA_UPDATE_FEED, the feed used last, or the project's
  --components-dir COMPONENTS_DIR
                        the component folder (default: the per-user one, or STRATA_COMPONENTS_DIR)
  --backend BACKEND     choose an engine family instead of detecting this PC's GPU
  --wait SECONDS        if an engine is serving, wait up to this long for it to stop instead of
                        refusing
  --force               try a version again that failed its test start or was rolled back from
  --model-config RUN_CONFIG.json
                        the run config of the model to use for the test start (default: an
                        installed model)
  --output {human,json,jsonl}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata components install-file`

```text
usage: strata components install-file [-h] [--feed FEED] [--components-dir COMPONENTS_DIR]
                                      [--backend BACKEND] [--wait SECONDS] [--force]
                                      [--model-config RUN_CONFIG.json]
                                      [--output {human,json,jsonl}] [--json] [--quiet]
                                      file

positional arguments:
  file                  the engine package (.tar.gz or .zip) from the release page

options:
  -h, --help            show this help message and exit
  --feed FEED           where the signed update feed is (URL or folder). Default:
                        $STRATA_UPDATE_FEED, the feed used last, or the project's
  --components-dir COMPONENTS_DIR
                        the component folder (default: the per-user one, or STRATA_COMPONENTS_DIR)
  --backend BACKEND     choose an engine family instead of detecting this PC's GPU
  --wait SECONDS        if an engine is serving, wait up to this long for it to stop instead of
                        refusing
  --force               try a version again that failed its test start or was rolled back from
  --model-config RUN_CONFIG.json
                        the run config of the model to use for the test start (default: an
                        installed model)
  --output {human,json,jsonl}
  --json                alias for --output json
  --quiet               suppress human informational output
```

### `strata update`

```text
usage: strata update [-h] [--feed FEED] [--components-dir COMPONENTS_DIR] [--backend BACKEND]
                     [--wait SECONDS] [--force] [--model-config RUN_CONFIG.json]
                     [--output {human,json,jsonl}] [--check] [--app] [--app-version APP_VERSION]
                     [--no-engine] [--no-control] [--json] [--quiet]

options:
  -h, --help            show this help message and exit
  --feed FEED           where the signed update feed is (URL or folder). Default:
                        $STRATA_UPDATE_FEED, the feed used last, or the project's
  --components-dir COMPONENTS_DIR
                        the component folder (default: the per-user one, or STRATA_COMPONENTS_DIR)
  --backend BACKEND     choose an engine family instead of detecting this PC's GPU
  --wait SECONDS        if an engine is serving, wait up to this long for it to stop instead of
                        refusing
  --force               try a version again that failed its test start or was rolled back from
  --model-config RUN_CONFIG.json
                        the run config of the model to use for the test start (default: an
                        installed model)
  --output {human,json,jsonl}
  --check               only read the feed and say what would change
  --app                 also download and verify the desktop app's installer
  --app-version APP_VERSION
                        the installed app's version (default: $STRATA_APP_VERSION)
  --no-engine
  --no-control
  --json                alias for --output json
  --quiet               suppress human informational output
```

### `strata models`

```text
usage: strata models [-h] {list,inspect,import,move} ...

positional arguments:
  {list,inspect,import,move}
    list                list local installed-model config candidates
    inspect             inspect a local model run config and its components
    import              preview an existing GGUF file set without adopting it
    move                move the model files (models, packs, mtp) to another folder: copied,
                        checked, then the old copies removed

options:
  -h, --help            show this help message and exit
```

#### `strata models list`

```text
usage: strata models list [-h] [--installed | --served] [--install-dir INSTALL_DIR] [--url URL]
                          [--connection CONNECTION] [--credential-env CREDENTIAL_ENV]
                          [--timeout TIMEOUT] [--output {human,json}] [--json] [--quiet]

options:
  -h, --help            show this help message and exit
  --installed           limit discovery to local setup run configs
  --served              inspect the currently served model
  --install-dir INSTALL_DIR
                        absolute path to a Strata installation
  --url URL             Strata server root; remote connections require HTTPS
  --connection CONNECTION
                        saved read-only connection
  --credential-env CREDENTIAL_ENV
                        environment variable containing the server credential
  --timeout TIMEOUT     bounded request timeout in seconds
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata models inspect`

```text
usage: strata models inspect [-h] [--install-dir INSTALL_DIR] [--output {human,json}] [--json]
                             [--quiet]
                             model_id

positional arguments:
  model_id

options:
  -h, --help            show this help message and exit
  --install-dir INSTALL_DIR
                        absolute path to a Strata installation
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata models import`

```text
usage: strata models import [-h] --plan [--output {human,json}] [--json] [--quiet] PATH

positional arguments:
  PATH

options:
  -h, --help            show this help message and exit
  --plan                inspect only; adoption requires a qualified model workflow
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata models move`

```text
usage: strata models move [-h] [--resume] [--leave] [--status] [--from OLD_FOLDER]
                          [--install-dir INSTALL_DIR] [--plan] [--cancel-file PATH]
                          [--output {human,json,jsonl}] [--json] [--quiet]
                          [NEW_FOLDER]

positional arguments:
  NEW_FOLDER            not needed with --resume, --leave or --status

options:
  -h, --help            show this help message and exit
  --resume              finish a move that was interrupted
  --leave               give up an interrupted move: keep the old folder and remove the partial
                        copies
  --status              say whether a move was interrupted (changes nothing)
  --from OLD_FOLDER     the folder the models are in now (default: the data folder in use)
  --install-dir INSTALL_DIR
                        absolute path to a Strata installation whose run configs should follow the
                        files
  --plan                say what would happen and change nothing
  --cancel-file PATH    stop (safely) when this file appears
  --output {human,json,jsonl}
  --json                alias for --output json
  --quiet               suppress human informational output
```

### `strata drafting`

```text
usage: strata drafting [-h] {plan,fetch} ...

positional arguments:
  {plan,fetch}
    plan        say what fetching would download; nothing is changed
    fetch       download the drafter (pinned, checked) and pack it for the engine

options:
  -h, --help    show this help message and exit
```

#### `strata drafting plan`

```text
usage: strata drafting plan [-h] [--data-dir DIR] [--config FILE] [--family NAME]
                            [--mode {auto,mtp,dflash2,off}] [--output {human,json,jsonl}]

options:
  -h, --help            show this help message and exit
  --data-dir DIR        the model data folder (default: the one Strata uses)
  --config FILE         a model's run config: its data folder, model and drafting choice decide
                        what is needed
  --family NAME         the model family when there is no run config yet (qwen, gyro, coder, ...)
  --mode {auto,mtp,dflash2,off}
                        the choice to decide for (default: the run config's own)
  --output {human,json,jsonl}
```

#### `strata drafting fetch`

```text
usage: strata drafting fetch [-h] [--data-dir DIR] [--config FILE] [--family NAME]
                             [--mode {auto,mtp,dflash2,off}] [--if-needed]
                             [--output {human,json,jsonl}]

options:
  -h, --help            show this help message and exit
  --data-dir DIR        the model data folder (default: the one Strata uses)
  --config FILE         a model's run config: its data folder, model and drafting choice decide
                        what is needed
  --family NAME         the model family when there is no run config yet (qwen, gyro, coder, ...)
  --mode {auto,mtp,dflash2,off}
                        the choice to decide for (default: the run config's own)
  --if-needed           download only when the choice would use the drafter and it is not here
  --output {human,json,jsonl}
```

### `strata engine`

```text
usage: strata engine [-h] {calibrate} ...

positional arguments:
  {calibrate}
    calibrate  measure, review and apply engine tuning

options:
  -h, --help   show this help message and exit
```

#### `strata engine calibrate`

```text
usage: strata engine calibrate [-h] --profile PROFILE (--plan | --measure |
                               --review-result REPORT.json | --apply-result REPORT.json |
                               --review-rollback REPORT.json | --rollback-result REPORT.json)
                               [--out REPORT.json] [--exclusive] [--output {human,json}] [--json]
                               [--quiet]

options:
  -h, --help            show this help message and exit
  --profile PROFILE
  --plan                read-only calibration preflight
  --measure             run decode sweep and save a report
  --review-result REPORT.json
                        show a candidate and its freshness
  --apply-result REPORT.json
                        smoke-test and apply a fresh report
  --review-rollback REPORT.json
                        check whether an applied result can be restored
  --rollback-result REPORT.json
                        smoke-test and restore the configuration before an application
  --out REPORT.json     new report path for --measure
  --exclusive           assert all model instances using these devices are stopped
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

### `strata install`

```text
usage: strata install [-h] [--update] ...

positional arguments:
  setup_args  options passed to setup.py, e.g. --family qwen --model Q2_0 --yes

options:
  -h, --help  show this help message and exit
  --update    update an existing install (what UPDATE.bat / update.sh run) instead of setting one
              up
```

### `strata runtime`

```text
usage: strata runtime [-h] {start,status,verify,stop,cleanup,restart} ...

`runtime start` starts a background service: the engine keeps serving after this terminal closes
and until `runtime stop` (or the process named by --die-with-pid ends). `strata serve` is the
foreground form: it ends with this terminal.

positional arguments:
  {start,status,verify,stop,cleanup,restart}
    start               start a saved profile as a background service under the local owner host
                        (it keeps serving after this terminal closes; stop it with `runtime stop`)
    status              inspect an owned local instance
    verify              recheck the owned API and one-token generation
    stop                drain and stop an owned local instance
    cleanup             end servers and engines Strata started whose owner is gone (each is
                        checked by PID, start time and command line first; a running instance is
                        never touched)
    restart             drain and rerun an owned saved profile

options:
  -h, --help            show this help message and exit
```

#### `strata runtime start`

```text
usage: strata runtime start [-h] --profile PROFILE [--die-with-pid PID] [--plan] [--wait WAIT]
                            [--shutdown-timeout SHUTDOWN_TIMEOUT] [--output {human,json}] [--json]
                            [--quiet]

options:
  -h, --help            show this help message and exit
  --profile PROFILE
  --die-with-pid PID    stop the instance (drained) when this process ends; for an app that owns
                        it
  --plan
  --wait WAIT
  --shutdown-timeout SHUTDOWN_TIMEOUT
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata runtime status`

```text
usage: strata runtime status [-h] --profile PROFILE [--output {human,json}] [--json] [--quiet]

options:
  -h, --help            show this help message and exit
  --profile PROFILE
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata runtime verify`

```text
usage: strata runtime verify [-h] --profile PROFILE [--wait WAIT] [--load] [--plan]
                             [--if-owner-id OWNER_ID] [--if-instance-id INSTANCE_ID]
                             [--output {human,json}] [--json] [--quiet]

options:
  -h, --help            show this help message and exit
  --profile PROFILE
  --wait WAIT
  --load                load an unloaded model for the generation check
  --plan                preview the owned instance and verification effect
  --if-owner-id OWNER_ID
                        require the owner shown by verify --plan
  --if-instance-id INSTANCE_ID
                        require the instance shown by verify --plan
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata runtime stop`

```text
usage: strata runtime stop [-h] --profile PROFILE [--plan] [--wait WAIT]
                           [--if-owner-id IF_OWNER_ID] [--if-instance-id IF_INSTANCE_ID]
                           [--output {human,json}] [--json] [--quiet]

options:
  -h, --help            show this help message and exit
  --profile PROFILE
  --plan                review the current owner and instance
  --wait WAIT
  --if-owner-id IF_OWNER_ID
                        require the owner ID from --plan
  --if-instance-id IF_INSTANCE_ID
                        require the instance ID from --plan
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata runtime cleanup`

```text
usage: strata runtime cleanup [-h] [--profile PROFILE] [--output {human,json}] [--json] [--quiet]

options:
  -h, --help            show this help message and exit
  --profile PROFILE     only this profile's leftovers
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata runtime restart`

```text
usage: strata runtime restart [-h] --profile PROFILE [--plan] [--allow-source-switch]
                              [--allow-source-migration] [--wait WAIT]
                              [--startup-wait STARTUP_WAIT] [--if-profile-revision REVISION]
                              [--if-owner-id OWNER_ID] [--if-instance-id INSTANCE_ID]
                              [--if-source-sha256 SHA256] [--output {human,json}] [--json]
                              [--quiet]

options:
  -h, --help            show this help message and exit
  --profile PROFILE
  --plan                compare current and saved launch without draining
  --allow-source-switch
                        permit a new source config file when the old file remains available for
                        rollback
  --allow-source-migration
                        permit edited run config bytes using a private rollback snapshot
  --wait WAIT
  --startup-wait STARTUP_WAIT
  --if-profile-revision REVISION
  --if-owner-id OWNER_ID
                        require the owner ID shown in --plan
  --if-instance-id INSTANCE_ID
                        require the instance ID shown in --plan
  --if-source-sha256 SHA256
                        require the candidate source bytes shown in --plan
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

### `strata ops`

```text
usage: strata ops [-h] {init,list,create,show,edit,set,history,rollback,plan,run,runs,result} ...

positional arguments:
  {init,list,create,show,edit,set,history,rollback,plan,run,runs,result}

options:
  -h, --help            show this help message and exit
```

#### `strata ops init`

```text
usage: strata ops init [-h] --profile PROFILE [--output {human,json}] [--json] [--quiet]

options:
  -h, --help            show this help message and exit
  --profile PROFILE
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata ops list`

```text
usage: strata ops list [-h] --profile PROFILE [--output {human,json}] [--json] [--quiet]

options:
  -h, --help            show this help message and exit
  --profile PROFILE
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata ops create`

```text
usage: strata ops create [-h] --profile PROFILE [--output {human,json}]
                         --kind {runtime.restart,engine.calibrate.measure,components.verify}
                         [--json] [--quiet]
                         name

positional arguments:
  name

options:
  -h, --help            show this help message and exit
  --profile PROFILE
  --output {human,json}
  --kind {runtime.restart,engine.calibrate.measure,components.verify}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata ops show`

```text
usage: strata ops show [-h] --profile PROFILE [--output {human,json}] [--json] [--quiet] name

positional arguments:
  name

options:
  -h, --help            show this help message and exit
  --profile PROFILE
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata ops edit`

```text
usage: strata ops edit [-h] --profile PROFILE [--output {human,json}] [--json] [--quiet] name

positional arguments:
  name

options:
  -h, --help            show this help message and exit
  --profile PROFILE
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata ops set`

```text
usage: strata ops set [-h] --profile PROFILE [--output {human,json}] [--plan] [--json] [--quiet]
                      name KEY=VALUE

positional arguments:
  name
  KEY=VALUE

options:
  -h, --help            show this help message and exit
  --profile PROFILE
  --output {human,json}
  --plan
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata ops history`

```text
usage: strata ops history [-h] --profile PROFILE [--output {human,json}] [--json] [--quiet] name

positional arguments:
  name

options:
  -h, --help            show this help message and exit
  --profile PROFILE
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata ops rollback`

```text
usage: strata ops rollback [-h] --profile PROFILE [--output {human,json}] --to REVISION [--json]
                           [--quiet]
                           name

positional arguments:
  name

options:
  -h, --help            show this help message and exit
  --profile PROFILE
  --output {human,json}
  --to REVISION
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata ops plan`

```text
usage: strata ops plan [-h] --profile PROFILE [--output {human,json}] [--json] [--quiet] name

positional arguments:
  name

options:
  -h, --help            show this help message and exit
  --profile PROFILE
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata ops run`

```text
usage: strata ops run [-h] --profile PROFILE [--output {human,json}] [--exclusive]
                      [--if-revision REVISION] [--if-profile-revision REVISION]
                      [--if-owner-id OWNER_ID] [--if-instance-id INSTANCE_ID]
                      [--if-source-sha256 SHA256] [--json] [--quiet]
                      name

positional arguments:
  name

options:
  -h, --help            show this help message and exit
  --profile PROFILE
  --output {human,json}
  --exclusive           assert exclusive device access for a calibration measurement
  --if-revision REVISION
                        fail if the saved recipe changed since preview
  --if-profile-revision REVISION
                        fail if the saved serving profile changed since preview
  --if-owner-id OWNER_ID
                        require the runtime owner shown in a restart recipe plan
  --if-instance-id INSTANCE_ID
                        require the runtime instance shown in a restart recipe plan
  --if-source-sha256 SHA256
                        require the candidate run-config bytes shown in a restart recipe plan
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata ops runs`

```text
usage: strata ops runs [-h] --profile PROFILE [--output {human,json}] [--limit LIMIT] [--json]
                       [--quiet]

options:
  -h, --help            show this help message and exit
  --profile PROFILE
  --output {human,json}
  --limit LIMIT
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata ops result`

```text
usage: strata ops result [-h] --profile PROFILE [--output {human,json}] [--json] [--quiet]
                         operation_id

positional arguments:
  operation_id

options:
  -h, --help            show this help message and exit
  --profile PROFILE
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

### `strata bench`

```text
usage: strata bench [-h] {run,cancel,report,compare} ...

positional arguments:
  {run,cancel,report,compare}
    run                 the standard benchmark (docs/BENCHMARK_PROTOCOL.md), or a serving suite
    cancel              stop the exact active saved-profile benchmark
    report              summarize an existing benchmark report
    compare             compare two saved runs with workload checks

options:
  -h, --help            show this help message and exit
```

(Printed with the system Python: the Strata venv lacks the packages this command imports.)

#### `strata bench run`

```text
usage: strata bench run [-h] (--profile PROFILE | --url URL)
                        [--suite {standard-v1,serving-smoke,parallel-mixed-v1,parallel-overload-v1,parallel-cancel-v1,parallel-prefill-cancel-v1}]
                        [--concurrency CONCURRENCY] [--repeat REPEAT] [--condition CONDITION]
                        [--require-managed-match] [--request-timeout REQUEST_TIMEOUT]
                        [--out NEW_DIRECTORY] [--credential-env CREDENTIAL_ENV]
                        [--telemetry-credential-env TELEMETRY_CREDENTIAL_ENV]
                        [--output {human,json}] [--optional-concurrency] [--no-start]
                        [--load-timeout SECONDS] [--track {default,custom}] [--track-notes TEXT]
                        [--model-revision REV] [--source-revision REV] [--tokenizer-revision REV]
                        [--quantization NAME] [--custom-pack TEXT] [--background TEXT]
                        [--build-options LIST] [--skip-model-hash] [--json] [--quiet]

options:
  -h, --help            show this help message and exit
  --profile PROFILE     saved serving profile or installed run config
  --url URL             server root; remote connections require HTTPS
  --suite {standard-v1,serving-smoke,parallel-mixed-v1,parallel-overload-v1,parallel-cancel-v1,parallel-prefill-cancel-v1}
                        standard-v1 (default) follows docs/BENCHMARK_PROTOCOL.md; the others are
                        serving suites for your own comparisons
  --concurrency CONCURRENCY
                        serving suites: simultaneous requests (1..16, or 1..64 for parallel-
                        overload-v1; default 4)
  --repeat REPEAT       serving suites: repeat measured phases (1..5; default 1)
  --condition CONDITION
                        serving suites: operator label for this run, e.g. baseline or monitor-
                        default
  --require-managed-match
                        serving suites: require the local owner host and current profile to match
                        the serving instance
  --request-timeout REQUEST_TIMEOUT
                        serving suites: seconds per request (1..600; default 120). The standard
                        suite fixes it at 1800
  --out NEW_DIRECTORY   new output directory (required for serving suites). Standard suite: a new
                        folder <date>-standard-<gpu>-<run id> under ./bench/results or ./bench-
                        results
  --credential-env CREDENTIAL_ENV
  --telemetry-credential-env TELEMETRY_CREDENTIAL_ENV
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output

standard suite (docs/BENCHMARK_PROTOCOL.md):
  --optional-concurrency
                        also run the optional stage: four short requests released together, three
                        rounds
  --no-start            with --profile: never start the engine; use only a server that is already
                        running
  --load-timeout SECONDS
                        with --profile: how long to wait for the engine to load the model (default
                        1800)
  --track {default,custom}
                        default: an unmodified release's setup-generated settings; custom (the
                        default here): anything else
  --track-notes TEXT    the release and setup choices (default track) or the changes made (custom
                        track)
  --model-revision REV  immutable revision of the quantization repository
  --source-revision REV
                        revision of the original Qwen/Qwen3.8-Flash-Next weights
  --tokenizer-revision REV
  --quantization NAME   when it cannot be read from the model file names
  --custom-pack TEXT    describe a prepared pack or custom expert profile
  --background TEXT     a program that ran during the benchmark (repeatable)
  --build-options LIST  comma-separated build options of a source build
  --skip-model-hash     do not hash the model files (the run is then marked nonconforming)
```

(Printed with the system Python: the Strata venv lacks the packages this command imports.)

#### `strata bench cancel`

```text
usage: strata bench cancel [-h] --profile PROFILE --path PATH [--output {human,json}]

options:
  -h, --help            show this help message and exit
  --profile PROFILE     saved serving profile
  --path PATH           absolute run directory or results.json
  --output {human,json}
```

(Printed with the system Python: the Strata venv lacks the packages this command imports.)

#### `strata bench report`

```text
usage: strata bench report [-h] [--output {human,json,markdown}] [--out NEW_FILE] [--json]
                           [--quiet]
                           path

positional arguments:
  path

options:
  -h, --help            show this help message and exit
  --output {human,json,markdown}
  --out NEW_FILE        write Markdown to a new file
  --json                alias for --output json
  --quiet               suppress human informational output
```

(Printed with the system Python: the Strata venv lacks the packages this command imports.)

#### `strata bench compare`

```text
usage: strata bench compare [-h] [--output {human,json,markdown}] [--out NEW_FILE] [--json]
                            [--quiet]
                            baseline candidate

positional arguments:
  baseline
  candidate

options:
  -h, --help            show this help message and exit
  --output {human,json,markdown}
  --out NEW_FILE        write Markdown to a new file
  --json                alias for --output json
  --quiet               suppress human informational output
```

(Printed with the system Python: the Strata venv lacks the packages this command imports.)

### `strata timeskip`

```text
usage: strata timeskip [-h] {run,status,list,apply} ...

positional arguments:
  {run,status,list,apply}
    run                 checkpoint, attempt a revision, check it and decide
    status              read a run's durable state and evidence
    list                list recorded runs
    apply               apply the retained result, refusing if the destination changed

options:
  -h, --help            show this help message and exit
```

#### `strata timeskip run`

```text
usage: strata timeskip run [-h] --workspace WORKSPACE --prompt PROMPT [--attempts ATTEMPTS]
                           [--revise-command REVISE_COMMAND] [--state-dir STATE_DIR]
                           [--output {human,json}]

options:
  -h, --help            show this help message and exit
  --workspace WORKSPACE
                        the directory holding the work (and, optionally, .strata/checks.json)
  --prompt PROMPT       what the revision should achieve
  --attempts ATTEMPTS   bounded revision attempts (0 = checkpoint and check the baseline only);
                        the specification's default is two
  --revise-command REVISE_COMMAND
                        the command that performs ONE revision attempt. It runs with the candidate
                        directory as its working directory, receives the prompt in
                        STRATA_TIMESKIP_PROMPT and the attempt number in STRATA_TIMESKIP_ATTEMPT,
                        and must not touch the workspace. Without it no revision is attempted,
                        which yields an inconclusive run rather than a fabricated improvement.
  --state-dir STATE_DIR
                        where runs are recorded (default: $STRATA_TIMESKIP_DIR or
                        $XDG_STATE_HOME/strata/time-skip, %LOCALAPPDATA%\Strata\time-skip on
                        Windows)
  --output {human,json}
```

#### `strata timeskip status`

```text
usage: strata timeskip status [-h] [--state-dir STATE_DIR] [--output {human,json}] run_id

positional arguments:
  run_id

options:
  -h, --help            show this help message and exit
  --state-dir STATE_DIR
  --output {human,json}
```

#### `strata timeskip list`

```text
usage: strata timeskip list [-h] [--state-dir STATE_DIR] [--output {human,json}]

options:
  -h, --help            show this help message and exit
  --state-dir STATE_DIR
  --output {human,json}
```

#### `strata timeskip apply`

```text
usage: strata timeskip apply [-h] [--workspace WORKSPACE] [--attempt ATTEMPT] [--dry-run]
                             [--restore] [--state-dir STATE_DIR] [--output {human,json}]
                             run_id

positional arguments:
  run_id

options:
  -h, --help            show this help message and exit
  --workspace WORKSPACE
                        destination (default: the workspace the run recorded)
  --attempt ATTEMPT     apply this revision attempt (chosen by the person after review) instead of
                        the retained checkpoint
  --dry-run             report what would change without writing anything
  --restore             put the original back instead of applying the result
  --state-dir STATE_DIR
  --output {human,json}
```

### `strata tune`

```text
usage: strata tune [-h] {plan,run,apply,rollback,status,list,auto} ...

positional arguments:
  {plan,run,apply,rollback,status,list,auto}
    plan                show exactly what a sweep would search, and why not
    run                 run a bounded sweep: measure the baseline and candidates, then report
    apply               apply a reviewed recommendation, after verifying the candidate runs
    rollback            restore the configuration before an apply
    status              read a job's durable state and report
    list                list recorded jobs
    auto                find the fastest settings for this PC and save them as a preset (quick:
                        about 20 minutes)

options:
  -h, --help            show this help message and exit
```

#### `strata tune plan`

```text
usage: strata tune plan [-h] --engine ENGINE --backend {sycl,cpu} --model MODEL
                        [--objective {generation,prefill,latency}] [--trials TRIALS]
                        [--time TOTAL_TIME] [--repeats REPEATS] [--hold HOLD]
                        [--output {human,json}]

options:
  -h, --help            show this help message and exit
  --engine ENGINE       the engine binary to interrogate and measure
  --backend {sycl,cpu}  the backend family the engine package claims
  --model MODEL         the model identifier the measurements belong to
  --objective {generation,prefill,latency}
  --trials TRIALS       the bounded short sweep (default 16)
  --time TOTAL_TIME     displayed total-time bound in seconds
  --repeats REPEATS     paired repeats per candidate (default 3, the specification's floor)
  --hold HOLD           hold a dimension constant instead of searching it (repeatable)
  --output {human,json}
```

#### `strata tune run`

```text
usage: strata tune run [-h] --engine ENGINE --backend {sycl,cpu} --model MODEL
                       [--objective {generation,prefill,latency}] [--trials TRIALS]
                       [--time TOTAL_TIME] [--repeats REPEATS] [--prompt-tokens PROMPT_TOKENS]
                       [--max-new MAX_NEW] [--mode {protocol,api}] [--api-prompt API_PROMPT]
                       [--api-expect API_EXPECT] [--api-port API_PORT] [--config CONFIG]
                       [--boot-timeout BOOT_TIMEOUT] [--request-timeout REQUEST_TIMEOUT]
                       [--model-path MODEL_PATH] [--context CONTEXT] [--full-digests]
                       [--hold HOLD] [--state-dir STATE_DIR] [--output {human,json}]

options:
  -h, --help            show this help message and exit
  --engine ENGINE       the engine binary to measure
  --backend {sycl,cpu}
  --model MODEL         the model the measurements belong to
  --objective {generation,prefill,latency}
  --trials TRIALS       bounded number of candidate configurations (default 16)
  --time TOTAL_TIME     displayed total-time bound in seconds
  --repeats REPEATS     paired repeats per candidate (default 3, the specification's floor)
  --prompt-tokens PROMPT_TOKENS
                        the workload's prompt length; fixed for the whole sweep
  --max-new MAX_NEW     the workload's output length; fixed for the whole sweep
  --mode {protocol,api}
                        protocol measures the engine's own serve protocol (cheap, for every
                        candidate); api drives the real HTTP API the agent uses (the confirmation,
                        and the only mode that reflects a user's wait)
  --api-prompt API_PROMPT
                        the prompt the api mode sends
  --api-expect API_EXPECT
                        a regular expression the answer must match for the run to count as
                        correct; without it correctness is not checked
  --api-port API_PORT
  --config CONFIG       the run configuration the api mode starts the worker with
  --boot-timeout BOOT_TIMEOUT
                        how long one candidate may take to become ready
  --request-timeout REQUEST_TIMEOUT
                        how long one measured request may take
  --model-path MODEL_PATH
                        the model's file, so the sweep's identity is a content digest rather than
                        a name. Without it the record is incomparable to everything, which is
                        safer than being falsely comparable.
  --context CONTEXT     the context the measurements were taken at
  --full-digests        hash the model and engine completely instead of sampling; slower but the
                        strongest identity
  --hold HOLD           hold a dimension constant instead of searching it (repeatable)
  --state-dir STATE_DIR
  --output {human,json}
```

#### `strata tune apply`

```text
usage: strata tune apply [-h] [--config CONFIG] [--model-path MODEL_PATH] [--context CONTEXT]
                         [--state-dir STATE_DIR] [--output {human,json}]
                         job_id

positional arguments:
  job_id                the completed sweep whose recommendation to apply

options:
  -h, --help            show this help message and exit
  --config CONFIG       the run configuration to change (strata-<model>.json)
  --model-path MODEL_PATH
                        the model file, to re-check the identity before writing
  --context CONTEXT
  --state-dir STATE_DIR
  --output {human,json}
```

#### `strata tune rollback`

```text
usage: strata tune rollback [-h] [--state-dir STATE_DIR] [--output {human,json}] application_id

positional arguments:
  application_id

options:
  -h, --help            show this help message and exit
  --state-dir STATE_DIR
  --output {human,json}
```

#### `strata tune status`

```text
usage: strata tune status [-h] [--state-dir STATE_DIR] [--output {human,json}] job_id

positional arguments:
  job_id

options:
  -h, --help            show this help message and exit
  --state-dir STATE_DIR
  --output {human,json}
```

#### `strata tune list`

```text
usage: strata tune list [-h] [--state-dir STATE_DIR] [--output {human,json}]

options:
  -h, --help            show this help message and exit
  --state-dir STATE_DIR
  --output {human,json}
```

#### `strata tune auto`

```text
usage: strata tune auto [-h] [--mode {quick,thorough}] [--goal GOAL] [--leave-ram PERCENT]
                        [--use {chat,coding}] [--allow-small-differences] [--allow-not-repeatable]
                        [--name NAME] [--time MINUTES] [--no-save] [--gpu-lock GPU_LOCK]
                        [--no-quiet-check] [--serve-dir SERVE_DIR] [--work-dir WORK_DIR]
                        [--status-file STATUS_FILE] [--boot-timeout BOOT_TIMEOUT]
                        [--output {human,json}]
                        target

Measures your configuration, tries single-knob changes in interleaved rounds against it, keeps
only gains above the noise, and saves the result as a named preset that is applied at every start.
Nothing changes if the current settings are already the fastest.

positional arguments:
  target                a saved profile name or a run configuration (JSON) to tune

options:
  -h, --help            show this help message and exit
  --mode {quick,thorough}
                        quick: about 20 minutes. thorough: up to 2 hours, 5-pair confirmation
                        (default quick)
  --goal GOAL           speed (full performance), room (leave memory free), or room:MiB (leave
                        that much graphics memory free)
  --leave-ram PERCENT   leave this share of system memory free (0-50)
  --use {chat,coding}   coding judges long-prompt speed as well
  --allow-small-differences
                        also try settings that can change answers slightly (expert cache split,
                        speculative window)
  --allow-not-repeatable
                        also try settings whose answers are not repeatable
  --name NAME           the preset's name (default: model, context and your wish)
  --time MINUTES        time to spend, in minutes
  --no-save             measure and report, but do not save a preset
  --gpu-lock GPU_LOCK   a lock file held around every engine run (shared boxes)
  --no-quiet-check      do not wait for other engines or a busy CPU
  --serve-dir SERVE_DIR
                        the folder that holds serve/ (default: this installation)
  --work-dir WORK_DIR   where the run's configs and logs go (default: a temp folder)
  --status-file STATUS_FILE
                        progress as JSON, rewritten at every step (for the desktop app)
  --boot-timeout BOOT_TIMEOUT
  --output {human,json}
```

### `strata presets`

```text
usage: strata presets [-h]
                      {list,status,show,apply,rename,delete,undo,export,import,effective,off,recheck} ...

positional arguments:
  {list,status,show,apply,rename,delete,undo,export,import,effective,off,recheck}
    list                every saved preset (or those for one model/configuration)
    status              which preset a configuration would start with, and whether it is still
                        valid
    show                one preset in full
    apply               make a preset the active one for its choices (the previous one is kept for
                        undo)
    rename              give a preset another name
    delete              delete a preset
    undo                go back to the preset that was active before (or the standard settings)
    export              write a preset to a file (or print it)
    import              read a preset from a file
    effective           what a start with these choices will actually run: the saved tuning's
                        values on top of the configuration
    off                 switch the saved tuning off for these choices (back to the standard
                        settings; it is kept and can be switched on again)
    recheck             a short check (about 2 minutes) that a preset is still as fast

options:
  -h, --help            show this help message and exit
```

#### `strata presets list`

```text
usage: strata presets list [-h] [--output {human,json}] [--model MODEL]

options:
  -h, --help            show this help message and exit
  --output {human,json}
  --model MODEL         only presets whose model or quant contains this text
```

#### `strata presets status`

```text
usage: strata presets status [-h] [--output {human,json}] target

positional arguments:
  target                a saved profile name or a run configuration (JSON)

options:
  -h, --help            show this help message and exit
  --output {human,json}
```

#### `strata presets show`

```text
usage: strata presets show [-h] [--output {human,json}] preset

positional arguments:
  preset

options:
  -h, --help            show this help message and exit
  --output {human,json}
```

#### `strata presets apply`

```text
usage: strata presets apply [-h] [--output {human,json}] preset

positional arguments:
  preset

options:
  -h, --help            show this help message and exit
  --output {human,json}
```

#### `strata presets rename`

```text
usage: strata presets rename [-h] [--output {human,json}] preset name

positional arguments:
  preset
  name

options:
  -h, --help            show this help message and exit
  --output {human,json}
```

#### `strata presets delete`

```text
usage: strata presets delete [-h] [--output {human,json}] preset

positional arguments:
  preset

options:
  -h, --help            show this help message and exit
  --output {human,json}
```

#### `strata presets undo`

```text
usage: strata presets undo [-h] [--output {human,json}] [--config CONFIG] preset

positional arguments:
  preset                a preset, or a configuration target with --config

options:
  -h, --help            show this help message and exit
  --output {human,json}
  --config CONFIG       a profile name or run configuration whose slot to undo
```

#### `strata presets export`

```text
usage: strata presets export [-h] [--output {human,json}] [--file FILE] preset

positional arguments:
  preset

options:
  -h, --help            show this help message and exit
  --output {human,json}
  --file FILE
```

#### `strata presets import`

```text
usage: strata presets import [-h] [--output {human,json}] file

positional arguments:
  file

options:
  -h, --help            show this help message and exit
  --output {human,json}
```

#### `strata presets effective`

```text
usage: strata presets effective [-h] [--output {human,json}] target

positional arguments:
  target                a saved profile name or a run configuration (JSON)

options:
  -h, --help            show this help message and exit
  --output {human,json}
```

#### `strata presets off`

```text
usage: strata presets off [-h] [--output {human,json}] target

positional arguments:
  target                a saved profile name or a run configuration (JSON)

options:
  -h, --help            show this help message and exit
  --output {human,json}
```

#### `strata presets recheck`

```text
usage: strata presets recheck [-h] [--output {human,json}] [--gpu-lock GPU_LOCK]
                              [--serve-dir SERVE_DIR] [--work-dir WORK_DIR]
                              [--status-file STATUS_FILE] [--boot-timeout BOOT_TIMEOUT]
                              [--no-quiet-check]
                              preset target

positional arguments:
  preset
  target                the profile name or run configuration the preset belongs to

options:
  -h, --help            show this help message and exit
  --output {human,json}
  --gpu-lock GPU_LOCK
  --serve-dir SERVE_DIR
  --work-dir WORK_DIR
  --status-file STATUS_FILE
  --boot-timeout BOOT_TIMEOUT
  --no-quiet-check
```

### `strata launch`

```text
usage: strata launch [-h] [--preset NAME] [--model NAME] [--family NAME] [--model-path PATH]
                     [--data-dir DIR] [--move-data] [--context SIZE] [--kv FORMAT]
                     [--vision off|on|gpu|cpu] [--parallel N] [--mtp on|off]
                     [--drafting auto|mtp|dflash2|off] [--gpu LIST] [--gpu-roles LIST]
                     [--tune quick|thorough|skip] [--tune-allow-small] [--set KEY=VALUE]
                     [--pool-cores LIST] [--access local|lan|cloudflare|tailscale] [--port N]
                     [--api-key KEY] [--api-key-env NAME] [--new-key] [--no-api-key]
                     [--save-preset NAME] [--retune] [--restart] [--no-tui] [--open] [--yes]
                     [--plan] [--feed URL|DIR] [--output {human,json}] [--list-presets]
                     [--delete-preset NAME] [--status] [--stop] [--show-key]

Set up a model, tune it, start serving and watch it, in one place. With no flags it opens the guided screens: start from a saved preset or set up from the beginning, one decision per screen. With flags it does the same without asking, and opens the monitor (or, with --no-tui, prints the address and returns). Presets are shared with the Strata app's profiles and tuning presets.

options:
  -h, --help            show this help message and exit

what to run:
  --preset NAME         start a saved preset (other flags change it for this start)
  --model NAME          the model size, e.g. iq3_xxs, q2_0, ud-q4_k_xl (any case)
  --family NAME         the model family when a size is offered by several (qwen, swift, coder,
                        ...)
  --model-path PATH     use model files you already have: a folder of shards or one .gguf file
  --data-dir DIR        where downloaded model files are stored (default: Strata's data folder)
  --move-data           with --data-dir: move the models you already have into the new folder.
                        Without it nothing is moved: your models stay where they are and stay
                        usable, and the new folder gets the new downloads
  --context SIZE        how much text it remembers at once: 32k, 64k, 128k, 256k or a number of
                        tokens
  --kv FORMAT           conversation memory format: int8 (default), q4_0 (half the memory), k8v4,
                        fp16
  --vision off|on|gpu|cpu
                        let the model read pictures. Default: on when the model can read pictures
                        and the engine has the picture reader. 'on' picks the reader by your
                        graphics memory: the CPU reader under 24 GB, the GPU reader from 24 GB
                        (any make); gpu / cpu choose one; off turns pictures off
  --parallel N          requests answered at the same time, 1 to 8 (default: fit to your graphics
                        card)
  --mtp on|off          the draft layer that makes answers faster. off starts without it: about 1
                        GB less graphics memory and less RAM, slower answers; use it when the
                        engine stops at start on a small card or a PC with little RAM. Default: on
  --drafting auto|mtp|dflash2|off
                        which drafter guesses words ahead to make answers faster: auto
                        (recommended: Strata picks from what it measured on cards like yours; that
                        is the MTP today), mtp, dflash2 (Qwen3.8-Flash-Next only; downloads a 1.2
                        GB drafter once) or off. If DFlash2 cannot run (not enough graphics
                        memory, or a model it is not for) the MTP is used and the start says why
  --gpu LIST            graphics card numbers to use, e.g. 0 or 0,1 (default: the engine's choice)
  --gpu-roles LIST      a role for each graphics card: main, helper or off, e.g. 1=main,0=helper
                        (cards you do not name are off). Main does the last steps of every answer;
                        a helper holds part of the model
  --tune quick|thorough|skip
                        find the fastest settings first (default: skip here; the guided screens
                        suggest quick the first time)
  --tune-allow-small    let the tuner also try settings that can change answers very slightly
  --set KEY=VALUE       an advanced engine setting or STRATA_* variable, e.g. prefill=65536
                        (repeat); unknown names get a warning, not a refusal
  --pool-cores LIST     experimental: pin the CPU workers to exactly these logical processors,
                        e.g. 2,4,6,8 or 2-14:2 (the numbers Task Manager or lscpu show). The
                        default is one worker per physical core, which the engine's start line
                        names; two entries on one core are kept with a warning

who can use it:
  --access local|lan|cloudflare|tailscale
                        local: this computer only; lan: your network; cloudflare: a quick tunnel;
                        tailscale: your tailnet
  --port N              port (default 8080)
  --api-key KEY         the key clients must send (made for you when you share beyond this
                        computer)
  --api-key-env NAME    read the key from this environment variable
  --new-key             make a new API key
  --no-api-key          share beyond this computer without an API key (a warning is shown, nothing
                        is refused)

running:
  --save-preset NAME    save these choices under this name
  --retune              with --tune (quick is used when --tune is not given): measure again even
                        when this model was already tuned on this computer (by default a stored
                        tuning result is reused and no measuring is done)
  --restart             when a model is already serving other choices (or holds the port), stop it
                        and start these. Without it the screens ask once and --no-tui refuses;
                        --yes alone never restarts. The same choices are never restarted: the
                        running address is printed
  --no-tui              no screens: serve, print the address and return
  --open                open the server's web page in your browser once it is serving (nothing
                        opens a browser without this)
  --yes                 do not ask questions (for one: whether to download the model on a computer
                        with no engine yet)
  --plan                show what would happen and exit; nothing is changed
  --feed URL|DIR        where the signed update feed (engine downloads) is; default:
                        STRATA_UPDATE_FEED or the project's
  --output {human,json}

manage:
  --list-presets        list saved presets
  --delete-preset NAME  delete a saved preset
  --status              show what this command is serving now
  --stop                stop what this command started (with --preset NAME: that one)
  --show-key            print the API key of --preset NAME (or of the running launch); with none
                        of those, the key of a model setup.py set up (--model NAME picks one when
                        several have a key)

examples:
  strata launch                                    open the guided screens (saved presets, then set up from the beginning)
  strata launch --preset "IQ3_XXS 128K"            start a saved preset (opens the monitor)
  strata launch --model iq3_xxs --context 128k --parallel 4 --vision off --tune quick \
                --access lan --port 8080 --save-preset "Office" --no-tui
                                                   set everything on the command line, serve, print the address
  strata launch --model-path /data/models/IQ3_XXS --no-tui
                                                   use model files you already have (a folder or one .gguf)
  strata launch --preset Office --set prefill=65536 --set STRATA_PIN_BY_NEED=1 --no-tui
                                                   change one advanced setting for this start
  strata launch --plan --model q2_0                show what would happen, change nothing
  strata launch --list-presets | --delete-preset NAME | --status | --stop [--preset NAME]
  strata connect                                   list the apps Strata can be connected to
  strata connect opencode --write                  write OpenCode's config for the running server (keeps a backup)
```

### `strata connect`

```text
usage: strata connect [-h] [--url URL] [--key KEY] [--key-env NAME] [--model NAME]
                      [--context SIZE] [--preset NAME] [--check] [--write] [--yes] [--show-key]
                      [--output {human,json}]
                      [harness]

Shows the settings file and environment an app needs to talk to the Strata server that `strata launch` started. Without a name it lists the apps. With --write it writes the file after asking, keeping a backup of any file it replaces.

positional arguments:
  harness               opencode, codex, claude-code, omp, pi, hermes, openclaw, deepseek-harness

options:
  -h, --help            show this help message and exit
  --url URL             the server address (default: the running launch, else
                        http://127.0.0.1:8080)
  --key KEY             the API key (default: the running launch's)
  --key-env NAME        read the key from this environment variable
  --model NAME          the model name the server reports (default: the running launch's)
  --context SIZE        the context the server was started with (default: the running launch's,
                        else 32k)
  --preset NAME         take the address, key and model from this saved preset
  --check               send one prompt and one tool call to the running server the way this app
                        would, and say if each worked
  --write               write the file(s), after asking; a replaced file is backed up first
  --yes                 do not ask before --write
  --show-key            print the real key instead of a masked one
  --output {human,json}

examples:
  strata launch                                    open the guided screens (saved presets, then set up from the beginning)
  strata launch --preset "IQ3_XXS 128K"            start a saved preset (opens the monitor)
  strata launch --model iq3_xxs --context 128k --parallel 4 --vision off --tune quick \
                --access lan --port 8080 --save-preset "Office" --no-tui
                                                   set everything on the command line, serve, print the address
  strata launch --model-path /data/models/IQ3_XXS --no-tui
                                                   use model files you already have (a folder or one .gguf)
  strata launch --preset Office --set prefill=65536 --set STRATA_PIN_BY_NEED=1 --no-tui
                                                   change one advanced setting for this start
  strata launch --plan --model q2_0                show what would happen, change nothing
  strata launch --list-presets | --delete-preset NAME | --status | --stop [--preset NAME]
  strata connect                                   list the apps Strata can be connected to
  strata connect opencode --write                  write OpenCode's config for the running server (keeps a backup)
```

### `strata status`

```text
usage: strata status [-h] [--url URL] [--connection NAME] [--credential-env NAME]
                     [--timeout TIMEOUT] [--output {human,json}] [--json] [--quiet]

options:
  -h, --help            show this help message and exit
  --url URL             Strata server root; remote connections require HTTPS
  --connection NAME     saved read-only connection
  --credential-env NAME
                        environment variable containing the server credential
  --timeout TIMEOUT     bounded request timeout in seconds (default 3)
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

### `strata monitor`

```text
usage: strata monitor [-h] [--url URL] [--connection NAME] [--credential-env NAME]
                      [--timeout TIMEOUT] [--once] [--output {human,json}] [--json] [--quiet]
                      [--theme {auto,dark,light,terminal}] [--color {auto,always,never}]
                      [--glyphs {auto,unicode,braille,ascii}] [--interval INTERVAL]
                      [--requests REQUESTS]

options:
  -h, --help            show this help message and exit
  --url URL             Strata server root; remote connections require HTTPS
  --connection NAME     saved read-only connection
  --credential-env NAME
                        environment variable containing the server credential
  --timeout TIMEOUT     bounded request timeout in seconds (default 3)
  --once                print one snapshot without starting the TUI
  --output {human,json}
  --json                alias for --output json with --once
  --quiet               suppress one-shot human output
  --theme {auto,dark,light,terminal}
  --color {auto,always,never}
  --glyphs {auto,unicode,braille,ascii}
  --interval INTERVAL
  --requests REQUESTS
```

### `strata logs`

```text
usage: strata logs [-h] [--url URL] [--connection NAME] [--credential-env NAME]
                   [--timeout TIMEOUT] [--source {runtime,harness}] [--follow]
                   [--interval INTERVAL] [--output {human,json,jsonl}] [--json] [--quiet]

options:
  -h, --help            show this help message and exit
  --url URL             Strata server root; remote connections require HTTPS
  --connection NAME     saved read-only connection
  --credential-env NAME
                        environment variable containing the server credential
  --timeout TIMEOUT     bounded request timeout in seconds (default 3)
  --source {runtime,harness}
  --follow              poll for new transitions until Ctrl+C
  --interval INTERVAL   follow poll interval in seconds (0.5–30; default 2)
  --output {human,json,jsonl}
  --json                alias for --output json without --follow
  --quiet               suppress human progress text
```

### `strata connection`

```text
usage: strata connection [-h] {create,show,list,set} ...

positional arguments:
  {create,show,list,set}
    create              save a named server endpoint
    show                show a saved connection without its credential
    list                list saved connections
    set                 edit one URL, prefix or credential reference

options:
  -h, --help            show this help message and exit
```

#### `strata connection create`

```text
usage: strata connection create [-h] --url URL [--prefix PREFIX] [--credential-env CREDENTIAL_ENV]
                                [--output {human,json}] [--json] [--quiet]
                                name

positional arguments:
  name

options:
  -h, --help            show this help message and exit
  --url URL
  --prefix PREFIX       reverse-proxy path prefix, e.g. /strata
  --credential-env CREDENTIAL_ENV
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata connection show`

```text
usage: strata connection show [-h] [--output {human,json}] [--json] [--quiet] name

positional arguments:
  name

options:
  -h, --help            show this help message and exit
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata connection list`

```text
usage: strata connection list [-h] [--output {human,json}] [--json] [--quiet]

options:
  -h, --help            show this help message and exit
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

#### `strata connection set`

```text
usage: strata connection set [-h] [--plan] [--if-revision REVISION] [--output {human,json}]
                             [--json] [--quiet]
                             name KEY=VALUE

positional arguments:
  name
  KEY=VALUE

options:
  -h, --help            show this help message and exit
  --plan
  --if-revision REVISION
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

### `strata config`

```text
usage: strata config [-h] {show} ...

positional arguments:
  {show}
    show      show connection and credential origins

options:
  -h, --help  show this help message and exit
```

#### `strata config show`

```text
usage: strata config show [-h] [--effective] [--url URL] [--connection NAME]
                          [--credential-env NAME] [--timeout TIMEOUT] [--output {human,json}]
                          [--json] [--quiet]

options:
  -h, --help            show this help message and exit
  --effective           show resolved settings (default)
  --url URL             Strata server root; remote connections require HTTPS
  --connection NAME     saved read-only connection
  --credential-env NAME
                        environment variable containing the server credential
  --timeout TIMEOUT     bounded request timeout in seconds (default 3)
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

### `strata code`

```text
usage: strata code [-h] [--workspace WORKSPACE] [--resume RESUME] [--quiet] {run,sessions} ...

positional arguments:
  {run,sessions}
    run                 scripted harness turn (requires the shared harness component)
    sessions            shared harness sessions

options:
  -h, --help            show this help message and exit
  --workspace WORKSPACE
  --resume RESUME
  --quiet               suppress human informational output
```

#### `strata code run`

```text
usage: strata code run [-h] --prompt-file PROMPT_FILE --policy POLICY [--output {jsonl}] [--quiet]

options:
  -h, --help            show this help message and exit
  --prompt-file PROMPT_FILE
  --policy POLICY
  --output {jsonl}
  --quiet               suppress human informational output
```

#### `strata code sessions`

```text
usage: strata code sessions [-h] [--output {human,json}] [--json] [--quiet] {list}

positional arguments:
  {list}

options:
  -h, --help            show this help message and exit
  --output {human,json}
  --json                alias for --output json
  --quiet               suppress human informational output
```

## `serve/server.py`

The server the setup starts. `python serve/server.py --help` needs `jinja2`, so this block was printed with the system Python, not the Strata venv:

```text
usage: server.py [-h] [--engine {mock,strata}] [--config CONFIG] [--host HOST] [--script-cycle]
                 [--script-delay SCRIPT_DELAY] [--lifetime-file LIFETIME_FILE] [--demo-history N]
                 [--script SCRIPT] [--port PORT] [--ready-file READY_FILE]
                 [--shutdown-timeout SHUTDOWN_TIMEOUT] [--parallel PARALLEL]
                 [--batch-groups BATCH_GROUPS] [--max-queue MAX_QUEUE]
                 [--queue-timeout QUEUE_TIMEOUT] [--batch-telemetry | --no-batch-telemetry]
                 [--offline] [--gpu GPU] [--tokenizer TOKENIZER] [--open] [--fit-max-tokens]
                 [--no-fit-max-tokens] [--api-key API_KEY] [--no-api-key]
                 [--mcp-config MCP_CONFIG] [--lazy] [--api-monitor] [--idle-unload SECONDS]
                 [--min-free-vram-mib MIN_FREE_VRAM_MIB] [--before-load BEFORE_LOAD]
                 [--slot-save-path DIR] [--replicas N] [--split-serve PREFILL,DECODE]

serve/server.py - plan v0.3 P8: OpenAI and Anthropic endpoints over any engine that maps token ids to tokens.

    python -m serve.server --engine mock --port 8095            (a scripted engine, for clients and tests)
    python -m serve.server --engine strata --config strata.json --port 8080   (the real engine, resident)

Endpoints: POST /v1/chat/completions (OpenAI, stream and non-stream), POST /v1/completions (OpenAI text completion: the raw prompt, no
chat template, stream and non-stream), POST /v1/messages (Anthropic, stream and
non-stream), GET /v1/models, GET /models, GET /props, GET /slots, GET /health, GET /mcp. One sequence at a time behind a FIFO (plan: one resident sequence).
Image generation (optional, when the config has an "images" section): POST /v1/images/generations (OpenAI's shape,
plus seed and steps) runs stable-diffusion.cpp's sd-cli for Qwen-Image-2.1-Turbo, one job at a time on the GPU the
text engine uses (serve/images.py).
Tools from MCP servers (serve/mcp.py, `"mcp_servers"` in the config or --mcp-config) are offered only to requests that
ask for them with `"strata_mcp": true` - the web app does; other clients see exactly the API they always saw.
Images (optional, when the config has a "vision" entry): OpenAI image_url parts and Anthropic image blocks (base64
data, http(s) URLs or local file paths) go through `strata-vision` (the model's mmproj file) and reach the engine as
embeddings (`GENI`).  JPEG/PNG/BMP/GIF go straight in; WebP, TIFF, AVIF, ... (agents like omp send WebP) are
converted to PNG first with Pillow.
A request whose prompt alone does not fit the engine's context is REJECTED with 400, never truncated.  When the prompt
fits but prompt plus max tokens does not (Claude Code asks for 32000, the Harness for 16384), max tokens is shortened to
the room left (less CTX_SLACK) and the thinking budget follows it; `--no-fit-max-tokens` / `"fit_max_tokens": false`
brings back the strict 400 of llama.cpp.
An unset (or 0, or -1) max tokens means "unlimited": whatever the prompt leaves of the context.

The engine boundary is `Engine.generate(prompt_ids, max_new, sampling, cancel) -> iterator of token ids`.
`StrataEngine` keeps one `strata --serve` process resident (weights, expert arena and VRAM tier load once) and
talks to it over stdin/stdout; `MockEngine` is a scripted stand-in that makes every API path testable without a GPU.

options:
  -h, --help            show this help message and exit
  --engine {mock,strata}
  --config CONFIG       strata engine config (JSON: exe, args, cwd, tokenizer, model_name),
                        written by setup.py
  --host HOST           the address to listen on: 127.0.0.1 = this PC only (the default), 0.0.0.0
                        = also other devices on your network, :: = all IPv6 interfaces (set an API
                        key); also "host" in the config
  --script-cycle        with --script given more than once: repeat the scripts by position instead
                        of advancing to the last one and staying there, so extra requests a
                        harness makes (a session title, a re-ask after a tool result) cannot shift
                        the sequence
  --script-delay SCRIPT_DELAY
                        seconds to wait between the mock engine's tokens, so a turn stays live
                        long enough for a client to cancel it; the mock honours cancellation
                        between tokens
  --lifetime-file LIFETIME_FILE
                        where the dashboard's lifetime token counters are kept between starts
                        (default: next to --config as <config>.lifetime.json; none without a
                        config)
  --demo-history N      mock engine only: fill the request history and lifetime counters with N
                        made-up requests (for looking at the dashboard without a GPU)
  --script SCRIPT       the mock engine's answer (default: a short greeting); given more than
                        once, requests get them in turn and the last one repeats
  --port PORT
  --ready-file READY_FILE
                        write a JSON instance record once the API is bound and model state is
                        known
  --shutdown-timeout SHUTDOWN_TIMEOUT
                        opt-in graceful drain deadline in seconds (0..300) before owned requests
                        are cancelled
  --parallel PARALLEL   request this many simultaneous engine slots without changing the saved
                        config
  --batch-groups BATCH_GROUPS
                        pipeline groups: auto or a positive integer, without changing the saved
                        config
  --max-queue MAX_QUEUE
                        maximum inference calls waiting beyond effective engine slots; 0 refuses
                        overload
  --queue-timeout QUEUE_TIMEOUT
                        maximum seconds waiting for the first engine turn (0.1..600; off by
                        default)
  --batch-telemetry, --no-batch-telemetry
                        opt in to low-rate engine decode-window samples for /v1/telemetry
  --offline             do not fetch remote image URLs or start MCP/tool subprocesses
  --gpu GPU             the GPU to run on, as setup numbers them, or several for a layer split
                        ("0,2"; also "gpu" in the config)
  --tokenizer TOKENIZER
                        pack tokenizer directory (falls back to a byte tokenizer if absent)
  --open                open the local page in the browser once the model is ready
  --fit-max-tokens      clamp max_tokens to the remaining context instead of rejecting the request
                        (the default; kept so older scripts keep working)
  --no-fit-max-tokens   answer 400 when prompt + max_tokens does not fit the context, like
                        llama.cpp (also "fit_max_tokens": false in the config). A prompt that
                        alone does not fit is always 400
  --api-key API_KEY     require this key on /v1/* (Authorization: Bearer ... or x-api-key); also
                        $STRATA_API_KEY. Several keys: separate them with commas (key1,key2), as
                        llama.cpp does
  --no-api-key          serve on the network without an API key (anyone who can reach the port can
                        use the model; also "api_key": "none" in the config). Warns once, never
                        refuses
  --mcp-config MCP_CONFIG
                        a JSON file with MCP servers in Claude Desktop's format ({"mcpServers":
                        {...}}); the web app's chat can use their tools (also "mcp_servers" in the
                        config)
  --lazy                start the API unloaded (the engine and the vision encoder start with the
                        first request)
  --api-monitor         the API request monitor at /api-monitor: keeps the last 100 requests'
                        prompts and answers in memory (also "api_monitor": true in the config;
                        "api_monitor_requests" sets the number; off by default)
  --idle-unload SECONDS
                        unload the model after this many seconds without requests, so other
                        programs (games, other model servers) can use the VRAM; the next request
                        loads it again (also "idle_unload_s" in the config; default: never)
  --min-free-vram-mib MIN_FREE_VRAM_MIB
                        load an unloaded model only when this much VRAM is free, else answer 503
                        (also "min_free_vram_mib" in the config; default: always load)
  --before-load BEFORE_LOAD
                        a command run before the model is loaded again (e.g. to unload another
                        server's model; also "before_load" in the config, a string or a list)
  --slot-save-path DIR  enable POST /slots/0?action=save|restore {"filename": NAME} (llama-
                        server's API): the conversation the engine holds, to or from DIR/NAME
                        (also "slot_save_path" in the config)
  --replicas N          data-parallel replicas (opt-in): cut the "gpu" list into N equal groups
                        and run one engine on each, requests spread over them (also "replicas" in
                        the config: a number, or [{"gpus": [0, 1]}, {"gpus": [2, 3]}])
  --split-serve PREFILL,DECODE
                        split serving (opt-in): one engine on card PREFILL reads the prompt, a
                        second on card DECODE makes the answer, the session handed over in RAM
                        (also "split_serve" in the config: {"prefill_gpu": 0, "decode_gpu": 1};
                        docs/SPLIT_SERVE.md)
```

## `setup.py` / `./setup.sh`

`./setup.sh` hands every run to `setup.py`; `--backend sycl` is the only backend:

```text
usage: setup.py [-h] [--family {qwen,swift,coder,unsloth,gyro}]
                [--model {Q2_0,IQ2_XS,IQ3_XXS,IQ3_S,IQ1_M,UD-IQ4_XS,UD-Q4_K_XL,UD-Q5_K_XL,UD-Q6_K_XL,Q8_0,UD-IQ1_S,Gyro-S,Gyro-M}]
                [--context CONTEXT] [--rope-scaling {none,linear,yarn}] [--rope-scale ROPE_SCALE]
                [--kv {int8,q4_0,k8v4,fp16}] [--vision {yes,no,none,gpu,cpu}] [--vision-tokens N]
                [--experimental-speed-projection on|off|GGUF] [--port PORT] [--gpu GPU]
                [--gpus GPUS] [--gpu-roles LIST] [--layer-split LAYER_SPLIT]
                [--no-remote-expert-opt] [--host HOST] [--api-key API_KEY] [--no-browser]
                [--browser] [--data-dir DATA_DIR] [--move-data] [--set-default-data-dir]
                [--models-dir MODELS_DIR] [--gguf-dir GGUF_DIR] [--yes] [--setup] [--no-start]
                [--update] [--source {auto,modelscope,huggingface}]
                [--inspect SOURCE [VARIANT ...]] [--check] [--calibrate]
                [--draft-vocab {cjk,en,cyrillic,fr,code}] [--low-ram {auto,on,off,resident,mmap}]
                [--resident-budget-gib N] [--vram-reserve-mib N] [--mtp {on,off}] [--parallel N]
                [--replicas N] [--kv-streaming {auto,on,off}] [--backend {sycl,hip}]

Strata one-click setup and start for the Intel Arc Pro B70 (Linux, the SYCL engine built from sycl/, docs/B70.md).

    ./setup.sh [--backend sycl] [options]      - installs Python if needed and runs this file

This setup has one engine: the SYCL build.  Every run goes through sycl/setup_intel.py (sycl_setup below), which finds
the Intel Arc card in sysfs, puts it into this file's card path (the steps named "AMD"/"hip" here, kept as the hooks
setup_intel.py replaces) and runs main() again with --backend hip.  The SYCL engine is built outside setup
(sycl/tools/build.sh in the strata-sycl-dev image); setup builds only the picture readers.

The first time it asks which model, which size, how much context and whether the model should also read pictures, then
installs everything and writes run-<model>.sh, which starts sycl/serve/server_intel.py on http://127.0.0.1:8080
(OpenAI- and Anthropic-compatible API).  Every later start skips straight to running the model: nothing that is already
downloaded, installed or prepared is done again.

What the first run does (each step is skipped when it is already done):

  1. checks your PC: the Intel Arc card, RAM, CPU, free disk space
  2. asks the questions
  3. installs the Python packages it needs into .venv (numpy, jinja2, ...)
  4. the engine folder: the picture readers (CPU, and with --vision gpu the SYCL one) beside the SYCL engine
  5. downloads the model from Hugging Face (resumable), and the vision encoder if you want images
  6. prepares the model for Strata and fetches the MTP draft layer (~5 GB, from the original Qwen checkpoint)
  7. writes the run config (strata-<model>.json) and run-<model>.sh and starts the model

Options: --family qwen|swift|coder|unsloth|gyro, --model Q2_0|IQ2_XS|IQ3_XXS|IQ3_S|..., --context 32768,
--rope-scaling none|linear|yarn (--rope-scale F; past the trained 262144 the setup adds yarn and the factor is the
final context over 262144, at least 1 - an explicit --rope-scaling none is refused for such a context), --vision
yes|no|gpu|cpu, --port 8080, --yes (recommended answers, no questions), --setup (install another model / change
settings instead of starting), --no-start, --host 0.0.0.0 --api-key KEY (reach it from other devices on your network),
--experimental-speed-projection on|off (EXPERIMENTAL, off by default), --models-dir DIR, --gguf-dir DIR (use GGUF files
you already have), --check (only check this PC), --resident-budget-gib N (UD-Q4_K_XL's or UD-IQ4_XS's experts in RAM),
--kv-streaming on|off|auto.

Setup recommends, it never forces: the recommended answers are the defaults (--yes, or Enter), and a bigger choice
than it recommends - a longer context, more GPUs, a bigger RAM budget, a size it thinks will not fit - is kept, with
what it risks.  With --yes, an explicit flag (--model, --gpus, ...) is the consent to a risk setup would otherwise
stop at; --yes alone is not.

options:
  -h, --help            show this help message and exit
  --family {qwen,swift,coder,unsloth,gyro}
                        qwen = Qwen3.8-Flash-Next, swift = Swift 1.5, coder, unsloth, gyro (see
                        the model list)
  --model {Q2_0,IQ2_XS,IQ3_XXS,IQ3_S,IQ1_M,UD-IQ4_XS,UD-Q4_K_XL,UD-Q5_K_XL,UD-Q6_K_XL,Q8_0,UD-IQ1_S,Gyro-S,Gyro-M}
  --context CONTEXT
  --rope-scaling {none,linear,yarn}
                        the RoPE extension for a context past the model's trained 262144: linear
                        (position interpolation) or yarn - llama.cpp's types. Omitted with a
                        scaled context, the setup picks yarn; 'none' is refused for such a context
  --rope-scale ROPE_SCALE
                        the extension factor, finite and >= 1 (default: the final context over the
                        trained 262144, at least 1 - 1.5 for 384K, 2 for 512K, 1 inside the
                        trained range)
  --kv {int8,q4_0,k8v4,fp16}
                        KV cache precision above 8K context: int8 (default), q4_0 (half the
                        memory, a little less precise) k8v4 (hybrid: INT8 K + 4-bit V, 816 B/cell)
                        or fp16 (16-bit, twice the memory of int8)
  --vision {yes,no,none,gpu,cpu}
                        let the model read images (on by default where the model can; yes = the
                        best reader for this card: the CPU under 24 GB of VRAM, the GPU from 24
                        GB, any make; gpu / cpu = that reader; no = text only)
  --vision-tokens N     the most image tokens a picture becomes (default 1024 with the encoder on
                        the GPU, 300 on the CPU): more reads small text and charts better, and
                        takes longer to encode; remembered for this model
  --experimental-speed-projection on|off|GGUF
                        EXPERIMENTAL, off by default: the control vector in data/experimental-
                        speed-projection (or another GGUF) as a projection on layers 4-44; see
                        docs/DETAILS.md
  --port PORT           the server's port (default: the one the install was set up with, 8080 for
                        a new one)
  --gpu GPU             one GPU, numbered as setup lists them (default: asked when several can be
                        used; with --setup it is saved, when starting it is for that start only)
  --gpus GPUS           several GPUs sharing one model, as setup lists them: "0,2", or "all"
                        (every card that can); Strata puts the faster card last, and the last card
                        is the Main one. Saved, also when starting (see docs/MULTI_GPU.md)
  --gpu-roles LIST      a role for each GPU: main, helper or off, e.g. 1=main,0=helper (cards you
                        do not name are off). Main does the last steps of every answer. Saved
                        (docs/MULTI_GPU.md)
  --layer-split LAYER_SPLIT
                        with --gpus: where each later GPU's layers start ("18", "16,32"), one
                        rising number per GPU after the first - not layers per card; default auto,
                        placed from each GPU's free VRAM
  --no-remote-expert-opt
                        with two or more GPUs: leave out --remote-expert-opt, which setup adds
                        there (#578)
  --host HOST           where the server listens: 127.0.0.1 = this PC only (default), 0.0.0.0 =
                        all IPv4 interfaces, :: = all IPv6 interfaces (issue #26; an --api-key is
                        recommended)
  --api-key API_KEY     require this key from clients (recommended with --host 0.0.0.0 or --host
                        ::); several keys: key1,key2
  --no-browser          do not open the chat page in the browser when the model is ready (for a
                        harness or an app that uses the API; remembered for this model, also in
                        run-<model>.bat/.sh)
  --browser             open the chat page again when the model is ready (the default; undoes
                        --no-browser)
  --data-dir DATA_DIR   where the model files go (~70-120 GB): default Strata-data next to this
                        folder, remembered for every Strata folder on this PC
  --move-data           with --data-dir: move the model files already on this PC into the new
                        folder (without it nothing is moved: the old files stay usable and the new
                        folder gets the new downloads)
  --set-default-data-dir
                        with --data-dir: remember the new folder as the data folder without moving
                        anything
  --models-dir MODELS_DIR
                        where the GGUF files go (default: <data folder>/models)
  --gguf-dir GGUF_DIR   use GGUF files you already have (a folder with every shard:
                        <name>-00001-of-0000N.gguf ... -0000N-of-0000N.gguf)
  --yes                 accept the recommended answers
  --setup               install another model or change settings
  --no-start            install only, do not start the model
  --update              update the installed engine, Python packages and model settings as a start
                        would, without starting the model (UPDATE.bat / update.sh run it after a
                        git pull)
  --source {auto,modelscope,huggingface}
                        where the model files come from: auto (default: Hugging Face), huggingface
                        or modelscope (mainland China: the same files, checked against
                        ModelScope's published SHA-256; STRATA_SOURCE)
  --inspect SOURCE [VARIANT ...]
                        what a GGUF is and whether Strata runs it, from its headers only (no
                        download): a file, a folder, a URL, ms:owner/repo (ModelScope) or
                        hf:owner/repo, and optionally a variant name
  --check               only check this PC and exit
  --calibrate           tune the engine's settings for this PC (about 15-30 minutes, longer on a
                        slow card), then start the model
  --draft-vocab {cjk,en,cyrillic,fr,code}
                        the draft layer's tokens: cjk = with Chinese, Japanese and Korean
                        (default), en = English and code only (~110 MiB less VRAM, English answers
                        1-2% faster), cyrillic = English, code and the Cyrillic script (Ukrainian,
                        Russian... answers decode ~30% faster), fr = English, code and French
                        (French answers: 18% more drafts accepted)
  --low-ram {auto,on,off,resident,mmap}
                        read the model's experts from one file in its folder instead of copying
                        them all into RAM (for a PC with a big GPU and little RAM); auto: when the
                        experts would not fit the RAM. In this mode the experts the GPU does not
                        hold are copied into RAM once when they fit (resident), else read through
                        the OS file cache (mmap); resident / mmap force one of the two
  --resident-budget-gib N
                        UD-Q4_K_XL, UD-IQ4_XS: the GiB of its experts kept in RAM (default: the
                        RAM less 24 GB, 40 on 64 GB; more is kept as you choose, with a note)
  --vram-reserve-mib N  VRAM in MiB the engine leaves free for other programs (a game, another
                        model; the engine's default: 700); the expert cache takes that much less
  --mtp {on,off}        the draft layer (MTP): off starts the model without it, which frees about
                        1 GB of graphics memory for the expert cache and takes less RAM, at the
                        price of speed; it is kept in the run config ("mtp": false) and a later
                        setup run does not undo it. on: back to the default
  --parallel N          up to N requests decode together (batch slots, opt-in; default: one at a
                        time, the others wait). Each slot takes VRAM from the expert cache; setup
                        says what it recommends
  --replicas N          several GPUs: run N independent engines, each on its own equal group of
                        the chosen cards (2 replicas on 4 cards = two 2-card engines), requests
                        spread over them (opt-in; more total throughput for a model that fits
                        fewer cards; each replica holds its own copy of the model)
  --kv-streaming {auto,on,off}
                        from a 64K context: keep the KV cache in RAM and only the attention's
                        window in VRAM (more experts fit on the GPU); auto: when the RAM has room
                        for it
  --backend {sycl,hip}  sycl = the Intel Arc engine (the default, and the only one: built from
                        sycl/, docs/B70.md). hip is internal: sycl/setup_intel.py runs this setup
                        through it
```

## Engine flags

The engine is the `strata` binary built from `sycl/`. Its flags are listed in `usage()` of `sycl/src/program/generate.cpp`; this is that text, unchanged. It does not list every flag the engine reads: for example `--w4` (and its alias `--b70-pack`), which the dense engine takes, appears in [B70_LIB.md](B70_LIB.md) and [Q36.md](Q36.md) and not here. Some lines still describe the CUDA engine this fork removed (for example the `--gpu` line). The B70 settings and the switches that exist only on this fork are in [B70.md](B70.md).

```text
strata generate --pack DIR --tokens "1,2,3" [options]

  --pack DIR           the pack directory (default pack/full)
  --tokens LIST        the prompt as comma-separated token IDS (required)
  --tokens-file PATH   pretokenized prompt, commas or whitespace (alternative to --tokens)
  --gpu LIST           run on these GPUs, comma-separated nvidia-smi/PCI indices (default: every
                       visible one); sets CUDA_VISIBLE_DEVICES/CUDA_DEVICE_ORDER inside the engine,
                       so a caller never has to export environment variables
  --ple-gguf PATH      required PLE table (original second GGUF shard); with --native, the model's
                       shard that holds per_layer_token_embd.weight when not given
  --no-ple             explicit diagnostic ablation of the PLE layer
  --ple-io direct|mmap|ram  n-gram table reads (plan v0.3 P2). direct (default): unbuffered SSD
                       reads, the table never enters RAM or the file cache; mmap: A/B arm;
                       ram: mmap with the whole table locked in RAM at start (Linux/macOS)
  --ple-row-cache N    bounded cache of fetched rows, 90 B each (default 1048576; 0 = off)
  --ple-inflight N     outstanding SSD reads (default 256)
  --ple-delay-us U     fault injection: each row read completes no earlier than U us
  --ple-sync-submit    A/B arm: submit table reads on the token thread (default: an I/O thread)
  --kv fp16|int8       KV storage (plan v0.3 P7): int8 codes + fp16 scale per 64 values, half the
                       VRAM; default fp16 until gate G-C accepts int8
  --kv q4_0            4-bit K/V after a Hadamard rotation (PR #21): half of int8's memory,
                       slightly lower precision (see bench/results/2026-09-27-kv-q4)
  --kv k8v4            hybrid: INT8 K (exact attention scores) + rotated Q4_0 V, 816 B/cell
                       (vs int8's 1,056); streams with --kv-resident too
  --kv-resident N      KV streaming: keep N cells of each QSA layer in VRAM (min 20480) and the
                       whole K/V in pinned RAM; the freed VRAM goes to expert slots. 0 (default):
                       all of it in VRAM. A context of N cells or fewer is not streamed
  --kv-grow            the K/V takes VRAM only for the cells the requests reach and the expert
                       cache holds the rest, giving slots back as the context grows (one GPU, with
                       --expert-profile; STRATA_KV_GROW=1/0 also). Default: the whole --max-context
                       allocated at start
  --stream-token       enqueue token work on the session stream (experimental)
  --check-logits       copy and check all logits in the stream-token path
  --gr-fp32-activations  experimental CUDA-oracle GR activation precision
  --gr-native-mmvf      experimental pinned GR norm/projections; implies FP32 activations
  --native-bf16         experimental CUDA-oracle SSM/router/indexer BF16 projections
  --native-bf16-extra   experimental CUDA-oracle PLE/shared gate BF16 projections
  --native-ple-key      experimental native PLE key; requires --native-dense-gguf
  --native-moe-combine  experimental pinned CUDA routed/shared combination
  --native-gdn          experimental pinned CUDA GDN norms/gates/recurrence
  --native-flash-attn-short  diagnostic pinned vector attention; --max-context <=256
  --native-qsa-indexer  experimental pinned indexer key cache and pooling
  --native-qsa          experimental pinned QSA normalization and output gate
  --native-rope         experimental pinned text-only CUDA rotary arithmetic
  --native-ple-postops  experimental pinned PLE postprojection arithmetic
  --native-router       experimental pinned CUDA 512-expert top-10 routing
  --cpu-oracle-q8-0     experimental pinned CPU expert quantization and dot reduction
  --native SHARD1      every full-context native path at once (plan v0.3 P1): stream-token,
                       GR MMVF, BF16, head, dense + PLE key, MoE combine, GDN, router, QSA,
                       indexer, RoPE, PLE postops, and the CPU q8_0 contract unless the
                       expert cache is on. Individual --native-* flags stay for A/B.
  --native-head-gguf PATH  native Q5_K head from model shard 1; requires --stream-token
  --embd-gguf PATH     the token embedding from this GGUF instead of --native's (tools/embd_bf16_pack.py:
                       BF16 as the checkpoint ships it; mapped host memory, no VRAM)
  --native-dense-gguf PATH native GDN/QSA/shared projections; repeat for each source model shard
  --expert-cache-cpu-order  experimental GPU expert reduction matching CPU order
  --max-new N          tokens to generate (default 16)
  --max-context N      KV/state capacity (default 4096)
  --rope-scaling T     extend the context past the trained one: none, linear
                       (position interpolation) or yarn - llama.cpp's types and names.
                       Default: the model file's rope keys, else none. Fixed at startup:
                       K in the cache is post-RoPE, so one run one scaling
  --rope-scale F       the extension factor for linear/yarn (default: the model file's
                       factor, else 1 = off)
  --rope-freq-base N   the raw ggml knobs: the frequency base (0 = the model's 1e7) and
  --rope-freq-scale F  the angle shrink (0 = 1/--rope-scale)
  --yarn-orig-ctx N    the trained context the correction targets (0 = 262144)
  --yarn-ext-factor F --yarn-attn-factor F --yarn-beta-fast F --yarn-beta-slow F
                       YaRN's knobs; defaults: -1 (auto: 1 for yarn), 1, 32, 1
  --greedy             argmax (the default)
  --seed S             enable sampling with this Philox seed
  --top-k N --top-p F --temperature F
  --dump-logits PATH   write one line of raw logits per position
  --logits-stride N    store every Nth row plus final input (default 1); N>1 requires --max-new 1
  --dump-residual PATH write the final R (hc x n_embd, f32) for head bisection
  --dump-layers PATH   write R after EVERY layer, per position: the C1 bisection ladder
  --dump-halves PATH   write both halves' block_out and inject per layer: the half bisection
  --dump-routing PATH  write the routed expert ids and weights per layer per position (P0.S8)
  --no-capture         run the layers directly instead of replaying graphs
  --shared-late        A/B: shared expert after the CPU pool (default: overlapped with it)
  --keep-canonical     A/B: also load canonical copies of natively served tensors (more VRAM)
  --vision             --serve takes images too (GENI requests; embeddings from strata-vision)
  --prompt-cache N     --serve: keep N conversation checkpoints between requests (default 6, ~118 MB
                       of RAM each; 0 = read every prompt from the start)
  --conversation-cache-mib N  --serve: RAM budget for parked conversations (default: sized from the free RAM; 0 = off)
  --conversation-cache-auto-pct N, --conversation-cache-auto-max-mib N  --serve: that automatic size: N%% of the
                       RAM free above --conversation-cache-min-free-mib (default 40), at most N MiB (default 8192)
  --conversation-cache-slots N  --serve: at most N parked conversations (default 4)
  --conversation-cache-min-free-mib N  --serve: RAM floor (under MemAvailable, the cgroup limits and
                       --memory-limit-mib) when parking, saving or restoring a session (default 2560)
  --memory-limit-mib N  --serve: the total RAM (MiB) this engine's container or cgroup may use, when
                       the container cannot see its real limit (also STRATA_MEMORY_LIMIT_MIB; the flag
                       wins; 0 = none). Parking and session save/restore keep their floor under it
  --conversation-cache-min-tokens N  --serve: park only conversations of at least N tokens, so
                       short side requests do not take the slots a long one needs (default 0 = park any)
  --conversation-cache-park-gain N  --serve: park only after the conversation advanced N tokens past
                       the copy it would supersede, so a turn that grew by a few hundred tokens does not
                       write a whole snapshot that the next turn drops (default 0 = park every turn)
  --conversation-cache-park-tail N  --serve: do not park a rewind on the live conversation that leaves
                       at most N tokens and 1/16 of it behind (the last reply, rewritten by the client;
                       default 4096; 0 = park every rewind)
  --session-min-free-mib N  --serve: disk space a SAVE must leave free (default 4096; 0 = no check)
  --session-save-reclaim  --serve: reclaim unpinned host caches if SAVE needs RAM (opt-in)
  --conversation-cache-park-ssd  --serve, with --conversation-cache-spill-dir: every parked conversation goes to
                       a file there and none stays in RAM (RAM is used only while a conversation is
                       written or read back). A switch costs the drive's seconds (DIR/drive-speed)
                       and writes the running state and the K/V that changed since the conversation
                       was read back (its K/V file is updated in place). -park-ssd-whole: write the whole
                       conversation at every park instead. -park-ssd-keep-pct N (default 50): a file is
                       updated in place when at least N%% of its tokens still hold, else a new file is
                       written. Every checkpoint of the conversation has its own file, written once.
                       -park-ssd-deepest-checkpoint: keep only the deepest one (less disk and a faster
                       read-back; a request that rewinds further back reads its prompt again). -park-on-exit: park the
                       conversation in use at a clean stop too, so the next start reads it back from its
                       file instead of reading its prompt again (the stop takes that write longer).
                       -park-ssd-checkpoints N: keep the shallowest checkpoint (the shared start) and
                       the N deepest instead of all of them (default 0 = all): each one is about 113 MiB
                       to read at every switch; a rewind to between the kept ones reads more prompt.
                       -park-ssd-stream: a read-back copies the K/V from its file straight into the
                       session, block by block, instead of reading the whole conversation into RAM first
  --conversation-cache-spill-dir DIR  --serve: the SSD tier: an evicted parked conversation is written to a
                       file here and read back when a prompt matches it (off by default); also
                       --conversation-cache-spill-mib N (quota, default 24576), -spill-min-free-mib N (disk
                       space kept free, default 40960), -spill-min-tokens N (default 4096), -spill-mbps N
                       and -spill-prefill-us N (the restore-vs-read gate: the drive's measured read
                       speed, 300 us/token; the drive is measured once, at the first start with the
                       folder, and the result kept in DIR/drive-speed; -spill-probe-mib N sets the
                       measurement's file, default 1024, 0: do not measure; -spill-probe-s N the
                       longest each direction may take, default 3; -spill-index-mib N: the folder's
                       index.log is written again from the live entries above N MiB, default 64, 0: never),
                       -spill-verify (check the payload hash on a read)
  --vram-elastic       --serve (#533, opt-in, NVIDIA): the expert cache in segments (--vram-segment-mib,
                       default 512), so the command `VRAM <reserve_mib>` (the server's POST /v1/vram) can
                       give VRAM back to other programs between requests and take it back later
  --batch N / --slots N  --serve (opt-in): up to N requests decode together in batch slots (2..8
                       normally; --batch-mtp waves more through eight-row windows),
                       each slot with its own session (VRAM like the main one); docs/BATCHING.md
  --batch-mtp          --batch (opt-in, needs --mtp and --spec): each slot also verifies one MTP proposal per
                       window (STRATA_BATCH_MTP=1 does the same); works on one GPU and on a layer split (the
                       slot drafters sit on the last stage's GPU; not with --batch-groups); needs VRAM per slot
  --batch-groups G     --batch with a layer split: the slots in G groups pipelined through the GPUs
  --pipeline-windows N --serve with a layer split on two GPUs (default 2 on two Arc B-series cards with
                       the MTP drafter, else 0): one conversation's verify windows
                       with the GPUs overlapped - 1 = the prompt's short reads, 2 = decode as well
                       (stage 0 runs the next window while stage 1 verifies this one); docs/MULTI_GPU.md
  --trim-stage-weights an explicit --layer-split: each GPU loads only its own layers' dense weights
  --prompt-cache-every N  --serve: also checkpoint every N fresh prompt tokens (default 16384, 0 = off)
  --prompt-cache-tail  --serve, single GPU: one extra checkpoint near the prompt's end, at an existing
                       chunk boundary (default off; requires --prompt-cache and --prompt-cache-every > 0)
  --prompt-cache-spread  --serve: keep the checkpoints spread back from the conversation's end (dense
                       near it, thinner further back) instead of the most recently used: a client that
                       rewrites its history N tokens back reads about N again, not the whole conversation
  --turn-token ID      --serve: the token that opens a chat turn (default 248045, <|im_start|>)
  --tail-role-token ID --serve: a turn of this role right before the last turn is left out of
                       the conversation checkpoint (a trailing effort turn; default -1 = off)
  --short-read N       --serve: read at most N fresh text tokens through the decode windows instead
                       of the batched prompt path (default 64, 0 = off)
  --prefill-head-alias --serve: unload the native output head while each fresh prompt is read,
                       then release prompt buffers and reload it before decode
  --suffix-draft N     prompt lookup: draft from an earlier repeat of the last N+ tokens of context
                       when it pays (default 3; 0 = MTP only)
  --mtp-max-t M        cap the MTP's windows at M tokens (0 = --spec; longer ones come from suffixes)
  --spec N             the MTP drafter's verify window: how many tokens it proposes per check (setup
                       writes 4; --suffix-draft lets it grow by 2, up to 8, where a repeat is likely)
  --spec-min-p P       how sure the draft layer must be to extend a verify window by another guess
                       (setup writes 0.5; --calibrate measures it on this PC, see docs/DETAILS.md)
  --lookup-chain K     opt-in: after the MTP's drafts, add up to K prompt-lookup drafts that continue
                       them (the window grows to at most 8; default 0 = off)
  --lookup-chain-min M  the shortest context match --lookup-chain extends on (default 3)
  --mtp-hnorm pooled|stream  the draft layer's hidden-input norm: one RMS over all four streams
                       (default) or one per stream (llama.cpp's MTP graph)
  --mtp-q4 all|proj|head  opt-in: the draft layer's projections and/or draft head as 4-bit copies
  --mtp-draft-vocab FILE  opt-in: the draft head's token subset (default <mtp>/draft_vocab.bin)
  --draft mtp|dflash2  the draft source (default mtp); dflash2 needs --draft-model PACK (tools/dflash2_pack.py)
                       and keeps the MTP layer loaded as the fallback (opt-in, Flash-Next, CUDA / HIP)
  --draft-conf P       with --draft dflash2: end the window where the product of the selector's probabilities falls below P (default 0 = --spec-min-p)
  --draft-cap N        the widest verify window with --draft dflash2 (default by card: 5 on Strix Halo / iGPU, --spec as given elsewhere)
  --control-vector-scaled FILE:SCALE[,...]  a control vector GGUF on the residual stream (llama.cpp's
                       format; --control-vector FILE = scale 1).  --serve: requests switch it (cvec=0|1)
  --control-vector-layer-range A B  the layers it follows (inclusive; default 1 .. the last)
  --cvec-mode add|project  h += s v (default) or h -= s (h.v) v with v unit
  --cvec-dir per-layer|single:L  each layer's own direction (default) or layer L's everywhere (project)
  --no-token-graph     A/B: two graphs per layer (the host launches each) instead of one per token
  --no-fused-gr        A/B: the six-kernel hyper-connection read and a separate write (native)
  --prefill CHUNK      batched prompt processing in chunks of CHUNK tokens (needs --native); auto =
                       the largest chunk up to 8192 whose buffers the expert cache can lend;
                       auto:16384 / auto:32768 (or STRATA_PREFILL_AUTO_MAX) allow bigger ones
  --prefill-pipe R     opt-in, a layer split: prompt chunks sized so the stages read at once - R is
                       this PC's b/a in tokens (setup's --calibrate measures it); 1 only evens out
                       the buffers' chunk count (STRATA_PREFILL_EQUAL's rule); 0 off (default; or
                       STRATA_PREFILL_PIPE)
  --no-pool            skip the CPU expert pool (the GPU-only floor)
  --sync-every-layer   debug: synchronise after every layer
  --ple-gguf PATH      the n-gram/PLE shard.  WITHOUT IT LAYER 1's PLE IS SILENTLY SKIPPED,
                       which changes every number downstream - pass it for any real run
  --dump-mixed PATH    write the post-attention residual (n_embd, f32)
  --stage-timing       per-stage KERNEL-COUNT shares.  NOT a time profile: an uncaptured
                       event interval includes host gaps, so run with --gpu-only-full first
  --graph-only         MEASURE: replay the 48 `pre` graphs only.  OMITS the 48 `post` graphs
                       and the LM head, so it is NOT the GPU floor (R0.3, Memory/ERRORS.md A4)
  --gpu-only-full      MEASURE: replay pre+post for all 48 layers plus the LM head, no pool.
                       THE TRUE PER-TOKEN GPU FLOOR.  Quote this one, not --graph-only.
  --stats              print the per-stage breakdown
  --spec-follow PATH   benchmarks: emit this continuation (token ids) instead of the argmax and
                       accept the drafts that match it, so speculation settings compare on the
                       same text (with --spec; not --serve)
  --window-hashes PATH per verify window, a line with its index, position, size, a 64-bit hash of
                       its final residual rows and its argmaxes: two builds that do the same
                       arithmetic write the same file (use with --spec-follow, --adapt-every 0)
  --gpu-stages         R0.9: capture the layer as three graphs (mixer / ffn+router / post)
                       and time them from OUTSIDE the capture.  The per-stage table on the
                       real graph that --stage-timing cannot give.  Prints and exits.
  --expert-profile P   R4.2e: pre-load the VRAM tier from a `profile.bin` (see
                       tools/make_profile.py) instead of admitting on first use.
  --expert-profile-save P  --serve, #477: save what the adaptive tier learned (the experts in
                       VRAM, then the routing counted since the start) as a profile at P, on
                       QUIT and every --expert-profile-save-every MIN minutes (default 10;
                       0 = on QUIT only) between requests; start from it with --expert-profile P
  --no-hit-poke        R4.2d's A/B arm.  The hit path pokes the driver once right after its
                       launch so the GPU starts while the CPU pool runs; without it the work
                       waits for the next driver entry and does not overlap at all.
  --no-ple-prefetch     A/B arm: read the PLE table's sixteen rows one at a time, instead of
                       issuing them in one PrefetchVirtualMemory call.
  --expert-cache N     R4: keep N expert blobs resident in VRAM and compute their rows on the
                       GPU via `moe_hit_grouped_s2`.  DEFAULT 0.  Measured at 4096 slots
                       with --expert-cache-per-layer: 54.4%% hits, CPU pool drain 19.1 -> 10.3
                       ms/token, -2.7 ms/token end to end.
  --expert-cache-device1 N|auto  fill CUDA1 with N experts or as many as fit
  --expert-cache-device2 N|auto  fill CUDA2 with more experts
  --expert-cache-device3 N|auto  fill CUDA3 with more experts
  --remote-expert-opt   enable remote-expert decode optimizations (serve mode)
  --expert-cache-remote-placement stripe|layer  distribute expert ranks or whole
                       layers across CUDA1..3 (default: stripe)
  --peer-device N      a second GPU as an adaptive expert-cache tier (rows over P2P; it also
                       computes its experts' prompt rows).  Not with --layer-split or
                       --expert-cache-device1..3.  Default output unchanged without it.
  --peer-reserve-mib N  VRAM the peer tier leaves free on its card (default 600)
  --peer-slots N       expert slots on the peer (default: what fits)
  --peer-adapt-swaps N  peer cache swaps per adaptation step (default: --adapt-swaps)
  --peer-prefill-rows N  prompt rows per layer the peer computes (default half of chunk x top-k;
                       0 = the prompt path stays on the primary)
  --expert-cache-per-layer  R4.2g: give each layer its OWN slots instead of letting the first
                       position take all of them.  The default policy fills in arrival order
                       from one shared counter, so 256 slots went to ~26 layers of position 0
                       and measured **2.97%%**.  Per-layer, the same routing gives 21.4%% at 8
                       slots/layer and 70.4%% at 64.  On a native pack, N is still the budget of N
                       largest blobs; each layer's slots are that layer's own blob.
  --no-host-worker     R2.2: the A/B arm.  By default the HOST THREAD joins the drain, so the
                       pool is six threads on six cores instead of five plus an idle core;
                       this flag restores the five-worker form for comparison on `pool phases`.
  --pool-workers N     R2.2: CPU expert pool worker count.  Default 0 = every physical core
                       except the one the host loop spins on (with --pool-affinity auto or
                       p-cores on a hybrid CPU: P-cores minus 1).  A sweep is how the pool's
                       deviation from `cpu_s2` is attributed.
  --pool-tasks N       Batched CPU expert tasks per GU/Down phase (0..4096). Default 0 =
                       3 per participating thread; positive counts are capped by row count.
  --host-core WHERE    The host thread's core: first (the default: the first physical core, as always) or
                       last (the last physical core, the workers on the others).  Windows sends a GPU's
                       interrupts to one logical processor, usually the first, and every copy that lands
                       raises one: a spinning host there waits for them.  Moves threads only, never a
                       result.  Not on hybrid CPUs.  sibling: the first core's other hardware thread
                       (SMT), so the interrupts keep the first logical processor and the workers keep
                       every core; hybrid CPUs too, first without SMT (STRATA_HOST_CORE sets it too).
  --pool-cores LIST    Experimental, off by default: pin the CPU workers to exactly these logical
                       processors (0,2,4,6 or 2-14:2; the numbers Task Manager and lscpu show).  The
                       default is one worker per physical core, which the start-up line names.  Two
                       entries on one core, or the host's own, are kept with a warning; processors the
                       process may not use are left out.  The worker count is the list's length unless
                       --pool-workers says otherwise.  Moves threads only, never a result.
                       STRATA_POOL_CORES sets it too.
  --aux-cpus SET       Where the threads that are neither pool workers nor the host thread run (the
                       adaptive tier's job thread, the prefill helpers, the CUDA driver's threads, ...):
                       off (the default: wherever they were created), auto (the SMT siblings of the host's
                       core, then physical cores with no worker; never a CPU that shares a core with a
                       worker; nothing is done when none is spare) or a CPU list such as 24,26 (Linux; the
                       host's and the workers' CPUs are dropped from it).  Moves threads only, never a
                       result.  STRATA_AUX_CPUS sets it too; a request's strata_tune aux_cpus=0|1 beats it.
  --pool-affinity MODE Worker CPU affinity: all (default: one worker per physical core, as
                       always), auto (hybrid CPUs: P-cores first, then their SMT siblings,
                       then E-cores) or p-cores (P-cores and their siblings only).  measure (opt-in,
                       hybrid CPUs, native packs): all and p-cores are timed on the pack's own experts
                       at start (about a second) and the faster one is taken
  --coupled-draft      enable coupled draft sampling for MTP drafter under sampling (STRATA_SPEC_COUPLED)
  --no-coupled-draft   disable coupled draft sampling (propose argmax drafts)
  --mmap-experts       R2.1: opt OUT of the resident expert arena, back to MapViewOfFile.
  --stream-experts     no host arena: experts read from the GGUF on demand (needs a native pack and
                       a VRAM cache that holds every expert; for machines with less RAM than experts)
                       The A/B arm: the mmap's rate depends on the OS page cache holding
                       34 GB, and measured 71.97 vs 34.78 ms/token cold vs warm.
  --shared-expert-arena FILE  Linux: back the resident arena with one MAP_SHARED file.
                       Put this file on /dev/shm, not ordinary SSD storage.
                       A small header binds an existing backing file to the same pack.
  --resident-cpu-experts  with mmap and a static profile, keep the experts the GPU cache does not
                       hold resident in ordinary RAM (and the prompt path's lend region as far as
                       RAM allows); adaptive swaps exchange them, so none is read from the file again.
  --resident-experts   the low-RAM PC's resident mode (setup): --mmap-experts --resident-cpu-experts
                       with the copy page-locked when possible, 4 GiB headroom, plain mmap if it
                       does not fit.  Same answers as --mmap-experts for the same placement.
  --adapt-min-gain F   the adaptive tier swaps an expert in only when its decayed routing count beats
                       the one it replaces by more than F (default 1.5; higher = fewer, surer swaps)
  --adapt-async 0|1    --serve with the resident RAM mode or the arena (one GPU): the adaptive
                       tier's swaps advance between verify windows instead of a window waiting for a
                       whole round (the arena: copy-in only).  Not bit-exact run to run.  Without the flag,
                       the expert arena on one CUDA GPU (no --batch) starts blocking and tries it for a
                       while when the swaps hold the windows up, keeping it only if it is faster; 0 = the
                       blocking tier always (also STRATA_ADAPT_ASYNC=0); 1 = always on.
```

The dense Qwen3.8-27B engine ([Q36.md](Q36.md)) prints this when it gets a flag it does not know (`sycl/src/qwen35/q36_engine.dp.cpp`):

```text
usage: strata-q36 --model GGUF [--max-context 32768] [--kv int8|fp16] [--chunk 512] [--no-graph]
       (--serve | --tokens IDS [--n 64] [--force IDS] [--dump-full-logits F] | --bench-pp N --bench-tg N)
```
