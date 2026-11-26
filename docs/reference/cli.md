# Command-line reference

Generated from `servecast --help` by `python tools/gen_cli_reference.py`; do not edit by hand.

## `servecast`

```text
 Usage: servecast [OPTIONS] COMMAND [ARGS]...

 Predict how an LLM serves on a GPU running vLLM: time to first token, time per output token and
 throughput, from a calibrated cost law and a workload.

 Start here


  servecast manifest init --template predict > m.yaml
  servecast predict -c m.yaml
  servecast replicate list


╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --version          <bool>  Print the version and exit.                                           │
│ --debug            <bool>  Show the full traceback when a command fails.                         │
│ --help     -h      <bool>  Show this message and exit.                                           │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Predict and calibrate ──────────────────────────────────────────────────────────────────────────╮
│ predict     Predict serving metrics for one deployment and workload.                             │
│ calibrate   Fit a cost law to profiling data and save the calibration.                           │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Measure on a GPU ───────────────────────────────────────────────────────────────────────────────╮
│ compare     Score a prediction against a measurement of the same manifest.                       │
│ profile     Measure on an NVIDIA GPU with vLLM.                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Helpers ────────────────────────────────────────────────────────────────────────────────────────╮
│ doctor      Check Python, the GPU, the driver and vLLM.                                          │
│ workload    Inspect a trace, or sample a workload into requests.                                 │
│ manifest    Create, check and explain manifests.                                                 │
│ list        Show what is known and installed.                                                    │
│ completion  Shell completion for bash, zsh and fish.                                             │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Reproduce the paper ────────────────────────────────────────────────────────────────────────────╮
│ replicate   Reproduce the paper with its pinned versions and data.                               │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## `servecast calibrate`

```text
 Usage: servecast calibrate [OPTIONS]

 Fit a cost law to profiling data and save the calibration.

 Fits the cost law selected by the versions to the profiling data, and saves the calibration in
 your data directory under a name that --with accepts. servecast predict runs the same fit on the
 fly when its manifest gives profiling data.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Inputs ─────────────────────────────────────────────────────────────────────────────────────────╮
│ --config    -c      MANIFEST  A manifest whose calibration.data names the profiling data.        │
│ --data              PATH      The profiling data, without a manifest: a file, a directory        │
│                               holding profiling.json, or the name of a shipped calibration whose │
│                               profiling data ships (servecast list calibrations).                │
│ --versions          DATE      Optional: the profiling data names its cost law. A year-month      │
│                               (2026-11) given here must select the same law.                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Output ─────────────────────────────────────────────────────────────────────────────────────────╮
│ --name              <str>         The calibration's name, to pass to --with (default             │
│                                   local/GPU/COST_LAW, plus /LLM for a per-LLM calibration).      │
│ --output-dir        DIR           Where calibration.json and run.json go; the calibration is     │
│                                   also saved under its name in your data directory.              │
│                                   [default: servecast-calibration]                               │
│ --format            <table|json>  How the summary is printed. Files on disk do not change.       │
│                                   [default: table]                                               │
│ --overwrite         <bool>        Replace your calibration of the same name, and an --output-dir │
│                                   that holds other files.                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  servecast calibrate -c m.yaml
  servecast calibrate --data ./a100-profiling-data --versions 2026-11 --name my-a100


 See also: servecast profile calibration, servecast list calibrations.
```

## `servecast compare`

```text
 Usage: servecast compare [OPTIONS] {PREDICTION_DIR} {MEASUREMENT_DIR}

 Score a prediction against a measurement of the same manifest.

 Refuses unless both come from the same manifest, request pool, sampler and estimator. For each
 load point, prints the signed error of every metric next to the measurement's confidence interval,
 and flags a clock or KV-capacity mismatch.

╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────╮
│ *    PREDICTION_DIR       <path>  What predict wrote.                                            │
│                                   [required]                                                     │
│ *    MEASUREMENT_DIR      <path>  What profile serving wrote.                                    │
│                                   [required]                                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Output ─────────────────────────────────────────────────────────────────────────────────────────╮
│ --output        FILE          Where comparison.json goes (default inside PREDICTION_DIR).        │
│ --format        <table|json>  How the summary is printed. Files on disk do not change.           │
│                               [default: table]                                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  servecast compare servecast-output servecast-measurement
  servecast compare prediction/ measurement/ --format json


 See also: servecast predict, servecast profile serving.
```

## `servecast completion`

```text
 Usage: servecast completion [OPTIONS] COMMAND [ARGS]...

 Shell completion for bash, zsh and fish.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────╮
│ install  Install completion for your shell, and print what was changed.                          │
│ show     Print the completion script for a shell.                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## `servecast completion install`

```text
 Usage: servecast completion install [OPTIONS] [shell]

 Install completion for your shell, and print what was changed.

╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────╮
│   shell      <bash|zsh|fish>  The shell (default the one running).                               │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  servecast completion install


 See also: servecast completion show.
```

## `servecast completion show`

```text
 Usage: servecast completion show [OPTIONS] {shell}

 Print the completion script for a shell.

╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────╮
│ *    shell      <bash|zsh|fish>  The shell.                                                      │
│                                  [required]                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  mkdir -p ~/.zfunc && servecast completion show zsh > ~/.zfunc/_servecast


 See also: servecast completion install.
```

## `servecast doctor`

```text
 Usage: servecast doctor [OPTIONS]

 Check Python, the GPU, the driver and vLLM.

 Prints what was found and, for each profile command, whether it can run on this computer or why
 not. Every other command runs on any computer.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Output ─────────────────────────────────────────────────────────────────────────────────────────╮
│ --format        <table|json>  How the summary is printed. Files on disk do not change.           │
│                               [default: table]                                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  servecast doctor


 See also: servecast profile calibration.
```

## `servecast list`

```text
 Usage: servecast list [OPTIONS] COMMAND [ARGS]...

 Show what is known and installed.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────╮
│ gpus          GPUs in the registry, and what each one's calibrations cover.                      │
│ laws          Cost laws by name: what one calibration covers, and what each computes.            │
│ components    Every component ServeCast can select, by name, with what it computes.              │
│ calibrations  Installed calibrations, shipped and your own, with what each covers.               │
│ manifests     Manifests that ship with ServeCast, examples to start from.                        │
│ models        LLMs whose geometry is known without downloading their config.json.                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## `servecast list calibrations`

```text
 Usage: servecast list calibrations [OPTIONS]

 Installed calibrations, shipped and your own, with what each covers.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Output ─────────────────────────────────────────────────────────────────────────────────────────╮
│ --format        <table|json>  How the summary is printed. Files on disk do not change.           │
│                               [default: table]                                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  servecast list calibrations


 See also: servecast calibrate.
```

## `servecast list components`

```text
 Usage: servecast list components [OPTIONS]

 Every component ServeCast can select, by name, with what it computes.

 A cost law's entry includes its calibration procedure.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Output ─────────────────────────────────────────────────────────────────────────────────────────╮
│ --format        <table|json>  How the summary is printed. Files on disk do not change.           │
│                               [default: table]                                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  servecast list components


 See also: servecast manifest explain versions.
```

## `servecast list gpus`

```text
 Usage: servecast list gpus [OPTIONS]

 GPUs in the registry, and what each one's calibrations cover.

 The shipped GPUs, then those of gpus.json in your data directory.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Output ─────────────────────────────────────────────────────────────────────────────────────────╮
│ --format        <table|json>  How the summary is printed. Files on disk do not change.           │
│                               [default: table]                                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  servecast list gpus


 See also: servecast list calibrations.
```

## `servecast list laws`

```text
 Usage: servecast list laws [OPTIONS]

 Cost laws by name: what one calibration covers, and what each computes.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Output ─────────────────────────────────────────────────────────────────────────────────────────╮
│ --format        <table|json>  How the summary is printed. Files on disk do not change.           │
│                               [default: table]                                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  servecast list laws


 See also: servecast list components.
```

## `servecast list manifests`

```text
 Usage: servecast list manifests [OPTIONS]

 Manifests that ship with ServeCast, examples to start from.

 They are inside the installed package. servecast manifest show NAME prints one, to save as a file
 of yours and give to -c by its path.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Output ─────────────────────────────────────────────────────────────────────────────────────────╮
│ --format        <table|json>  How the summary is printed. Files on disk do not change.           │
│                               [default: table]                                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  servecast list manifests


 See also: servecast manifest show.
```

## `servecast list models`

```text
 Usage: servecast list models [OPTIONS]

 LLMs whose geometry is known without downloading their config.json.

 The shipped LLMs, then those of models.json in your data directory.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Output ─────────────────────────────────────────────────────────────────────────────────────────╮
│ --format        <table|json>  How the summary is printed. Files on disk do not change.           │
│                               [default: table]                                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  servecast list models


 See also: servecast predict --to.
```

## `servecast manifest`

```text
 Usage: servecast manifest [OPTIONS] COMMAND [ARGS]...

 Create, check and explain manifests.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────╮
│ init      Write a commented manifest to start from.                                              │
│ show      Print a manifest that ships with ServeCast, to save as a file of yours.                │
│ validate  Check a manifest the way predict does, without predicting.                             │
│ explain   Explain one manifest field: meaning, type, default and allowed values.                 │
│ schema    Print the JSON Schema of the manifest or of an output file.                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## `servecast manifest explain`

```text
 Usage: servecast manifest explain [OPTIONS] {PATH}

 Explain one manifest field: meaning, type, default and allowed values.

╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────╮
│ *    PATH      <str>  A dotted field path, e.g. traffic.load.concurrency.                        │
│                       [required]                                                                 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  servecast manifest explain traffic.load.think_time_ms


 See also: servecast manifest init.
```

## `servecast manifest init`

```text
 Usage: servecast manifest init [OPTIONS]

 Write a commented manifest to start from.

 It holds the fields a prediction needs, each with an example value and a one-line note, and
 versions set to the month of the newest components. servecast manifest explain describes any
 field, those left out too.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --template           <predict|profile>  What the manifest is for.                                │
│                                         [default: predict]                                       │
│ --out                FILE               Write to FILE instead of standard output (- is standard  │
│                                         output). An existing FILE is refused without             │
│                                         --overwrite.                                             │
│ --overwrite          <bool>             Replace an existing FILE.                                │
│ --help       -h      <bool>             Show this message and exit.                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  servecast manifest init > m.yaml
  servecast manifest init --template profile --out profile.yaml


 See also: servecast manifest explain.
```

## `servecast manifest schema`

```text
 Usage: servecast manifest schema [OPTIONS]

 Print the JSON Schema of the manifest or of an output file.

 Point an editor at the manifest's schema to get completion and checks while writing it.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --kind          <manifest|pool|profiling-data|probe|ge  Which file's schema.                     │
│                 mm|calibration|prediction|measurement|  [default: manifest]                      │
│                 comparison|run|gpus|models|components>                                           │
│ --help  -h      <bool>                                  Show this message and exit.              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  servecast manifest schema > servecast.schema.json
  servecast manifest schema --kind prediction


 See also: servecast manifest validate.
```

## `servecast manifest show`

```text
 Usage: servecast manifest show [OPTIONS] {NAME}

 Print a manifest that ships with ServeCast, to save as a file of yours.

 The manifests that ship with ServeCast are examples inside the installed package, which a
 reinstall replaces. Save one to a file, edit the file, and give its path to -c.

╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────╮
│ *    NAME      <str>  A manifest that ships with ServeCast, as servecast list manifests names    │
│                       it.                                                                        │
│                       [required]                                                                 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  servecast manifest show example-a100 > example-a100.yaml


 See also: servecast list manifests.
```

## `servecast manifest validate`

```text
 Usage: servecast manifest validate [OPTIONS]

 Check a manifest the way predict does, without predicting.

 Reports every problem in the file at once, as file: path: message with the allowed values. A file
 without any is then resolved as predict resolves it (the calibration, the component versions, the
 LLM, the deployment and the request pool), and the first problem predict would refuse is reported.
 The workload law is not fitted, and predict can still refuse a small pool when it fits it.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Inputs ─────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --config  -c      MANIFEST    The manifest: a YAML file, or the name of one that ships with   │
│                                  ServeCast (servecast list manifests).                           │
│                                  [required]                                                      │
│    --set             PATH=VALUE  Override one manifest field; the value is read as YAML.         │
│                                  Repeatable, applied left to right, then the whole manifest is   │
│                                  validated.                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Output ─────────────────────────────────────────────────────────────────────────────────────────╮
│ --format        <table|json>  How the summary is printed. Files on disk do not change.           │
│                               [default: table]                                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  servecast manifest validate -c m.yaml


 See also: servecast manifest explain.
```

## `servecast predict`

```text
 Usage: servecast predict [OPTIONS]

 Predict serving metrics for one deployment and workload.

 Reads the manifest, resolves the calibration and the component versions, simulates the closed
 workload to steady state, and writes prediction.json: means, P95s and 95 % confidence intervals,
 with the components that made them. When the manifest gives profiling data instead of a
 calibration, the calibration is fitted first.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Inputs ─────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --config  -c      MANIFEST     The manifest: a YAML file, or the name of one that ships with  │
│                                   ServeCast (servecast list manifests).                          │
│                                   [required]                                                     │
│    --with            CALIBRATION  A calibration name, a calibration.json, or the folder          │
│                                   calibrate writes. Same as --set calibration.use=CALIBRATION.   │
│    --to              LLM          The LLM to predict with the calibration, a transfer: a Hugging │
│                                   Face id or a local directory holding config.json. Same as      │
│                                   --set engine.model=LLM --set calibration.transfer=true.        │
│    --set             PATH=VALUE   Override one manifest field; the value is read as YAML.        │
│                                   Repeatable, applied left to right, then the whole manifest is  │
│                                   validated.                                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Workload ───────────────────────────────────────────────────────────────────────────────────────╮
│ --sweep        PATH=V1,V2,...  Repeat for each value of ONE field, traffic.load.concurrency or   │
│                                traffic.load.think_time_ms, e.g.                                  │
│                                traffic.load.concurrency=40,80,120; each value once.              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Output ─────────────────────────────────────────────────────────────────────────────────────────╮
│ --output-dir                 DIR           Where the results go. A non-empty directory is        │
│                                            refused without --overwrite.                          │
│                                            [default: servecast-output]                           │
│ --overwrite                  <bool>        Replace the known output files in a non-empty output  │
│                                            directory.                                            │
│ --format                     <table|json>  How the summary is printed. Files on disk do not      │
│                                            change.                                               │
│                                            [default: table]                                      │
│ --capture-per-request        <bool>        Also write requests.jsonl, one record per simulated   │
│                                            request.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Advanced ───────────────────────────────────────────────────────────────────────────────────────╮
│ --jobs           <int range> [x>=1]  Load points of a sweep run in parallel processes; the       │
│                                      numbers do not change.                                      │
│                                      [default: 1]                                                │
│ --timeout        SECONDS [x>=1]      Stop the workload-law fit, or a load point that has not     │
│                                      converged, after this long: a stopped fit ends the command, │
│                                      a stopped point's numbers are written, marked as not        │
│                                      converged, and either way the command exits with status     │
│                                      124.                                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  servecast predict -c m.yaml
  servecast predict -c m.yaml --with a100_sxm_40gb/c62ebe75 --to Qwen/Qwen2.5-7B
  servecast predict -c m.yaml --sweep traffic.load.concurrency=40,80,120 --format json


 See also: servecast compare, servecast manifest explain, servecast list calibrations.
```

## `servecast profile`

```text
 Usage: servecast profile [OPTIONS] COMMAND [ARGS]...

 Measure on a real GPU: Linux, an NVIDIA GPU of compute capability 8.0 or newer, and vLLM 0.25.1,
 installed with pip install 'servecast[profile]'; servecast doctor checks them. Every other command
 runs on any computer.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────╮
│ calibration  Probe the cost law on this GPU and write its profiling data.                        │
│ serving      Serve the workload with vLLM on this GPU and measure it.                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## `servecast profile calibration`

```text
 Usage: servecast profile calibration [OPTIONS]

 Probe the cost law on this GPU and write its profiling data.

 Runs the calibration probe of the cost law the manifest's versions select: batch compositions of
 controlled sizes, each timed step by step. For the per-GPU law (c62ebe75), the default, it probes
 8 synthetic models served with random weights, then a grid of matrix multiplications; for the
 paper's law (d9cbf1d3, once per LLM and GPU), the manifest's LLM. vLLM runs with the probe's own
 engine settings, those every shipped calibration was measured with (a context of 4,096 tokens, 512
 sequences, 2,048 batched tokens, 9,965 KV blocks, or the manifest's block count if smaller for the
 paper's law); the manifest's other engine settings are for serving, so the same manifest serves
 both commands. Writes every step and the audit (probe.json, or one probe-MODEL.json per synthetic
 model and gemm.json) and, when every audit is clean, profiling.json, which servecast calibrate
 fits.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Inputs ─────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --config  -c      MANIFEST    The manifest: a YAML file, or the name of one that ships with   │
│                                  ServeCast (servecast list manifests).                           │
│                                  [required]                                                      │
│    --set             PATH=VALUE  Override one manifest field; the value is read as YAML.         │
│                                  Repeatable, applied left to right, then the whole manifest is   │
│                                  validated.                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Output ─────────────────────────────────────────────────────────────────────────────────────────╮
│ --output-dir        DIR           Where the results go. A non-empty directory is refused without │
│                                   --overwrite.                                                   │
│                                   [default: servecast-profiling-data]                            │
│ --overwrite         <bool>        Replace the known output files in a non-empty output           │
│                                   directory.                                                     │
│ --format            <table|json>  How the summary is printed. Files on disk do not change.       │
│                                   [default: table]                                               │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Advanced ───────────────────────────────────────────────────────────────────────────────────────╮
│ --poweroff-when-safe        <bool>  Power this machine off (sudo shutdown -h now) once the       │
│                                     results are written and match their checksums; not after a   │
│                                     failure. Many cloud providers bill a powered-off machine     │
│                                     until it is terminated.                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  servecast profile calibration -c m.yaml


 See also: servecast calibrate, servecast doctor.
```

## `servecast profile serving`

```text
 Usage: servecast profile serving [OPTIONS]

 Serve the workload with vLLM on this GPU and measure it.

 Starts vLLM 0.25.1 with the manifest's engine settings, drives the closed workload from the same
 request pool that servecast predict uses, and stops each load point by the manifest's rule, with
 the GPU clock sampled. Writes measurement.json, with the same summary fields as prediction.json,
 for servecast compare.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Inputs ─────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --config  -c      MANIFEST    The manifest: a YAML file, or the name of one that ships with   │
│                                  ServeCast (servecast list manifests).                           │
│                                  [required]                                                      │
│    --set             PATH=VALUE  Override one manifest field; the value is read as YAML.         │
│                                  Repeatable, applied left to right, then the whole manifest is   │
│                                  validated.                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Workload ───────────────────────────────────────────────────────────────────────────────────────╮
│ --sweep        PATH=V1,V2,...  Repeat for each value of ONE field, traffic.load.concurrency or   │
│                                traffic.load.think_time_ms, e.g.                                  │
│                                traffic.load.concurrency=40,80,120; each value once.              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Output ─────────────────────────────────────────────────────────────────────────────────────────╮
│ --output-dir        DIR           Where the results go. A non-empty directory is refused without │
│                                   --overwrite.                                                   │
│                                   [default: servecast-measurement]                               │
│ --overwrite         <bool>        Replace the known output files in a non-empty output           │
│                                   directory.                                                     │
│ --format            <table|json>  How the summary is printed. Files on disk do not change.       │
│                                   [default: table]                                               │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Advanced ───────────────────────────────────────────────────────────────────────────────────────╮
│ --timeout                   SECONDS [x>=1]  Stop a load point that has not converged after this  │
│                                             long: its numbers are written, marked as not         │
│                                             converged, and the command exits with status 124.    │
│ --poweroff-when-safe        <bool>          Power this machine off (sudo shutdown -h now) once   │
│                                             the results are written and match their checksums;   │
│                                             not after a failure. Many cloud providers bill a     │
│                                             powered-off machine until it is terminated.          │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  servecast profile serving -c m.yaml
  servecast profile serving -c m.yaml --sweep traffic.load.concurrency=40,80,120


 See also: servecast compare, servecast doctor, servecast workload sample.
```

## `servecast replicate`

```text
 Usage: servecast replicate [OPTIONS] COMMAND [ARGS]...

 Reproduce the paper with its pinned component versions, calibrations, LLMs and traces. Every
 output is labelled paper replication. rq1 and rq2 are recomputed; rq3 and appendix fit the paper's
 calibrations again from their shipped profiling data; ablation computes its tables from the
 shipped predictions of the ablated models; Vidur is not run: its numbers are the ones the paper
 recorded.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────╮
│ list     List the paper's experiments and what each one reproduces.                              │
│ fetch    Download the paper's data record (not in this release).                                 │
│ predict  Predict one of the paper's configurations beside its measurement.                       │
│ run      Recompute one of the paper's experiments and print its table.                           │
│ verify   Check the paper's printed cells against what the shipped data give.                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## `servecast replicate fetch`

```text
 Usage: servecast replicate fetch [OPTIONS]

 Download the paper's data record (not in this release).

 The record is published with the first release. replicate run --from-frozen and the recomputation
 of rq1 and rq2 need no download.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Output ─────────────────────────────────────────────────────────────────────────────────────────╮
│ --format        <table|json>  How the summary is printed. Files on disk do not change.           │
│                               [default: table]                                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  servecast replicate fetch


 See also: servecast replicate run.
```

## `servecast replicate list`

```text
 Usage: servecast replicate list [OPTIONS]

 List the paper's experiments and what each one reproduces.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Output ─────────────────────────────────────────────────────────────────────────────────────────╮
│ --format        <table|json>  How the summary is printed. Files on disk do not change.           │
│                               [default: table]                                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  servecast replicate list


 See also: servecast replicate run.
```

## `servecast replicate predict`

```text
 Usage: servecast replicate predict [OPTIONS]

 Predict one of the paper's configurations beside its measurement.

 The calibration, the LLM, the trace and the client count must all be the paper's; the output
 prints the paper's measurement next to the prediction, and replication.json in the output
 directory keeps both.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Inputs ─────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --with        CALIBRATION  One of the paper's calibrations: GPU/d9cbf1d3/LLM.                 │
│                               [required]                                                         │
│ *  --to          LLM          One of the paper's LLMs, on the same GPU.                          │
│                               [required]                                                         │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Workload ───────────────────────────────────────────────────────────────────────────────────────╮
│ *  --trace          <azure-conversation|burstgpt|servege  One of the paper's traces.             │
│                     n>                                    [required]                             │
│ *  --clients        N                                     One of the paper's client counts for   │
│                                                           that LLM and GPU.                      │
│                                                           [required]                             │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Output ─────────────────────────────────────────────────────────────────────────────────────────╮
│ --output-dir        DIR           Where replication.json goes; an earlier one there is replaced. │
│                                   [default: servecast-output]                                    │
│ --format            <table|json>  How the summary is printed. Files on disk do not change.       │
│                                   [default: table]                                               │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  servecast replicate predict --with a100_sxm_40gb/d9cbf1d3/llama-3-8b --to qwen2.5-7b \
      --trace burstgpt --clients 66


 See also: servecast replicate run, servecast list calibrations.
```

## `servecast replicate run`

```text
 Usage: servecast replicate run [OPTIONS] {experiment}

 Recompute one of the paper's experiments and print its table.

 rq1 and rq2 predict every configuration with the paper's pinned pipeline and shipped calibrations,
 score it against the paper's measurements, and print the table in the paper's layout; with
 --from-frozen the table is computed from the shipped predictions instead, in seconds. rq3 and
 appendix fit the paper's 18 calibrations again from their shipped profiling data and print the
 training MAPE and the coefficients they give, beside the times the paper measured, as printed.
 ablation computes its tables from the shipped predictions of the ablated models and fits the
 workload laws again from the traces. Vidur is not run: its numbers are the ones the paper
 recorded.

╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────╮
│ *    experiment      <rq1|rq2|rq3|ablation|appendix>  Which experiment.                          │
│                                                       [required]                                 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Inputs ─────────────────────────────────────────────────────────────────────────────────────────╮
│ --gpu                <str>   Restrict rq1 or rq2 to one of the paper's GPUs.                     │
│ --from-frozen        <bool>  Print the tables from the shipped predictions, or as published, in  │
│                              seconds, instead of recomputing.                                    │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Output ─────────────────────────────────────────────────────────────────────────────────────────╮
│ --output-dir        DIR           Where EXPERIMENT.json goes; an earlier result of the same      │
│                                   experiment is replaced.                                        │
│                                   [default: servecast-replicate]                                 │
│ --format            <table|json>  How the summary is printed. Files on disk do not change.       │
│                                   [default: table]                                               │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Advanced ───────────────────────────────────────────────────────────────────────────────────────╮
│ --jobs        <int range> [x>=1]  Load points predicted, or calibrations and workload laws       │
│                                   fitted, in parallel processes.                                 │
│                                   [default: 1]                                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  servecast replicate run rq1 --from-frozen
  servecast replicate run rq2 --gpu h100_sxm


 See also: servecast replicate verify, servecast replicate fetch.
```

## `servecast replicate verify`

```text
 Usage: servecast replicate verify [OPTIONS] [experiments]...

 Check the paper's printed cells against what the shipped data give.

 Rebuilds the accuracy table, its load bands and the transfer tables from the shipped predictions,
 the ablation tables from the shipped predictions of the ablated models and the workload-law table
 from the shipped laws, fits the paper's 18 calibrations again from their shipped profiling data
 for the coefficients and the training MAPE, and compares each cell with the printed one at its
 printed precision. It takes seconds with one process per core. A damaged data file or a changed
 fit shows here. replicate run rq1 and rq2 recompute the predictions themselves.

╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────╮
│   experiments...      <list[rq1|rq2|rq3|ablation|appendix  Which experiments (default: all of    │
│                       ]>                                   them).                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Output ─────────────────────────────────────────────────────────────────────────────────────────╮
│ --format        <table|json>  How the summary is printed. Files on disk do not change.           │
│                               [default: table]                                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Advanced ───────────────────────────────────────────────────────────────────────────────────────╮
│ --jobs        <int range> [x>=1]  Calibrations fitted in parallel processes (default: one per    │
│                                   core).                                                         │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  servecast replicate verify
  servecast replicate verify rq1


 See also: servecast replicate run.
```

## `servecast workload`

```text
 Usage: servecast workload [OPTIONS] COMMAND [ARGS]...

 Inspect a trace, or turn a mean-and-variance workload into concrete requests.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────╮
│ describe  Summarize the request lengths of a trace.                                              │
│ sample    Draw the request pool of a manifest's workload and write it.                           │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## `servecast workload describe`

```text
 Usage: servecast workload describe [OPTIONS] {TRACE}

 Summarize the request lengths of a trace.

 Prints the number of requests, the mean, variance and quantiles of the input and output lengths,
 and their correlation.

╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────╮
│ *    TRACE      <path>  A Mooncake JSONL file, a lengths file, or a trace the package ships      │
│                         (azure-conversation, burstgpt, servegen).                                │
│                         [required]                                                               │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Output ─────────────────────────────────────────────────────────────────────────────────────────╮
│ --format        <table|json>  How the summary is printed. Files on disk do not change.           │
│                               [default: table]                                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  servecast workload describe azure_conv.jsonl
  servecast workload describe burstgpt


 See also: servecast workload sample.
```

## `servecast workload sample`

```text
 Usage: servecast workload sample [OPTIONS]

 Draw the request pool of a manifest's workload and write it.

 The same seeded pool that predict and profile serving draw from; prints the requested and the
 realized moments side by side.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --help  -h      <bool>  Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Inputs ─────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --config  -c      MANIFEST    The manifest: a YAML file, or the name of one that ships with   │
│                                  ServeCast (servecast list manifests).                           │
│                                  [required]                                                      │
│    --set             PATH=VALUE  Override one manifest field; the value is read as YAML.         │
│                                  Repeatable, applied left to right, then the whole manifest is   │
│                                  validated.                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Output ─────────────────────────────────────────────────────────────────────────────────────────╮
│ --out              FILE          Where the pool goes. An existing FILE is refused without        │
│                                  --overwrite.                                                    │
│                                  [default: pool.jsonl]                                           │
│ --overwrite        <bool>        Replace an existing FILE.                                       │
│ --format           <table|json>  How the summary is printed. Files on disk do not change.        │
│                                  [default: table]                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯

 Examples


  servecast workload sample -c m.yaml --out pool.jsonl


 See also: servecast predict, servecast profile serving.
```
