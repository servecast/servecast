# Files and folders

Every command reads its inputs from a few known places and writes its outputs to a folder or a file of its own.
`predict` and `calibrate` also write a `run.json` beside their outputs, which records how they were made. This page
lists both. The format of every file is in
[File formats](formats.md).

## What ships with ServeCast

Installed with the package, under `servecast/data/`:

| folder | what it holds |
|---|---|
| `registry/gpus.json` | the GPUs ServeCast ships calibrations for, with the datasheet numbers the cost laws divide by |
| `registry/components.json` | every component ServeCast can select, by name: what it computes, its parameters, its calibration and where the paper describes it |
| `registry/models.json` | the LLMs known without downloading their `config.json`: the fields of it the cost laws read |
| `manifests/<name>.yaml` | the example manifests, which `servecast list manifests` lists and `servecast manifest show <name>` prints, to save as a file of yours |
| `calibrations/<gpu>/<law>[/<llm>].json` | the shipped calibrations: a cost law's fitted coefficients for one GPU (and, for the paper's law, one LLM) |
| `profiling/<gpu>/<law>[/<llm>].json` | the profiling data each shipped calibration was fitted on, under the same name |
| `traces/` | the paper's three traces (`azure-conversation`, `burstgpt`, `servegen`) |
| `workload_laws/` | the workload law fitted to each shipped trace |
| `paper/` | the paper's predictions and measurements (`campaigns.json`) and its printed tables (`tables.json`) |

A shipped calibration and its profiling data share a name, so `servecast calibrate --data a100_sxm_40gb/c62ebe75
--versions 2026-11` refits the shipped A100 calibration from the installed package. `servecast list calibrations`
shows which calibrations ship their profiling data.

A `gpus.json` or `models.json` in your data directory has the format of the shipped one (`servecast manifest
schema --kind gpus`, `--kind models`). It adds entries and cannot change a shipped one, whose numbers the shipped
calibrations were fitted with; an LLM you add is read as its `config.json` would be, with the same refusals.

## Your files

| place | what it holds | where |
|---|---|---|
| your manifests | YAML files, each given to `-c` by its path, such as `example-a100.yaml` from `servecast manifest show example-a100` | wherever you keep them, such as the directory you run ServeCast in |
| data directory | the calibrations you fitted, under their names (`calibrations/<name>.json`), `gpus.json` and `models.json` for GPUs and LLMs you add | `~/Library/Application Support/servecast` on macOS, `~/.local/share/servecast` on Linux, or `SERVECAST_DATA_DIR` |
| cache | workload laws fitted to new request pools (each with a checksum, fitted again if it does not match), and downloaded `config.json` files | `~/Library/Caches/servecast` on macOS, `~/.cache/servecast` on Linux, or `SERVECAST_CACHE_DIR` |

A relative path in a manifest, such as a trace, profiling data or a calibration file, is read from the directory you
run ServeCast in, not from the manifest's own directory. The `manifest.yaml` a prediction writes keeps the paths as the
manifest gave them, so run it again from the same directory, or give absolute paths. `servecast doctor` names the data
directory and the cache.

## What each command writes

| command | folder (`--output-dir`) | files |
|---|---|---|
| `predict` | `servecast-output/` | `prediction.json`, `pool.jsonl`, `requests.jsonl` (with `--capture-per-request`), `manifest.yaml`, `run.json` |
| `calibrate` | `servecast-calibration/` | `calibration.json`, `run.json`; the calibration is also saved in your data directory |
| `compare` | the prediction's folder (`--output` changes it) | `comparison.json` |
| `replicate run` | `servecast-replicate/` | `<experiment>.json` |
| `replicate predict` | `servecast-output/` | `replication.json` |
| `workload sample` | the file `--out` names (`pool.jsonl`) | the request pool |
| `profile calibration` | `servecast-profiling-data/` | `profiling.json`; `probe.json` for the paper's law, or one `probe-<model>.json` per synthetic model and `gemm.json` for the per-GPU law; `environment.json`, `engine.log`, `SHA256SUMS` |
| `profile serving` | `servecast-measurement/` | `measurement.json`, `requests.jsonl`, `pool.jsonl`, `environment.json`, `engine.log`, `SHA256SUMS` |

`manifest.yaml` is the manifest as it ran, with every default filled in: `servecast predict -c manifest.yaml`
predicts the same numbers again. `run.json` records the command line, the version of ServeCast, when it ran, each
input with its sha256 (the manifest, the calibration, the request pool, the profiling data), the component versions,
and the other files of the folder.

Look at two files before you share a folder. `run.json` keeps the command line and the input paths as you typed them,
so an absolute path shows your user name. `engine.log` is vLLM's own output, kept unedited, so it holds whatever vLLM
prints about this machine. `environment.json` never records a GPU's serial number, UUID or bus id, or the host name.
ServeCast never writes a token. It reads `HF_TOKEN` from the environment and sends it only to the Hugging Face hub.
