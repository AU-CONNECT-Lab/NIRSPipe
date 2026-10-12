---
title: nirspipe.utils
---

# `nirspipe.utils`

Cross-cutting utilities: logging, run records, JSONL events, SQLite database, and helpers used by `cli/`.

## `logging`

| Function | Description |
|----------|-------------|
| `get_logger(name) → logging.Logger` | Return a named logger configured for NIRSPipe. Use `--verbose` at the CLI to enable `DEBUG` output. |
| `setup_logging(...)` | Initialise the root logger and the file handler that writes `output_dir/sub-<label>/logs/sub-<label>.log`. |

See [Logging](../internals/logging.md) for logger names and conventions.

---

## `run_record`

| Function | Description |
|----------|-------------|
| `write_run_record(args, subject, timestamp, output_dir, sub_dir=None, prep_config=None, post_config=None)` | Write a per-subject TOML run record under `sub_dir/logs/sub-<subject>.toml`. Sections: `[environment]`, `[execution]`, `[prep]`, `[post]`. |
| `write_group_run_record(args, group_id, task, timestamp, output_dir, group_dir, members=None) → Path` | The dyad mirror, `group_dir/logs/group-<id>_task-<task>.toml`. Sections: `[environment]`, `[execution]`, `[hyper]`. |

See [Run Records](../logging-and-database/run-records.md#per-subject-toml-record) for the schema.

---

## `job_db`: JSONL events + SQLite

JSONL event writers (used by `pipeline/` and `cli/` to emit events during a run):

| Function | Event |
|----------|-------|
| `log_execution(db_path, command_line, nirspipe_version, …) → execution_id` | `execution_start` (also returns the `int(time.time() * 1000)` execution id) |
| `update_execution(db_path, execution_id, status, error_msg)` | `execution_update` |
| `log_run_start(db_path, execution_id, subject, …)` | `run_start` |
| `log_run_end(db_path, execution_id, subject, status, …)` | `run_end` |
| `log_sqm(db_path, execution_id, subject, checkpoint, sqm, …)` | `sqm`, one call per section of a run's quality record; `checkpoint` is the section name (scalar values only, nested dicts are dropped) |
| `log_output(db_path, execution_id, subject, …)` | `output` (stdout / stderr truncated to last 50 lines) |

Merger (called by `nirspipe-log merge` and `nirspipe-log rebuild`):

| Function | Description |
|----------|-------------|
| `merge_jsonl(db_path) → (files, rows)` | Insert every finished execution's JSONL under `db_path.parent/json/` into the SQLite database at `db_path`, then move those files to `archived/`. Backs the database up to `backup/` beside it first. |
| `rebuild_db(db_path) → Path` | Build a new, timestamped database beside `db_path` from every finished execution's logs, archived ones included; `db_path` and the logs are left as they were. |

See [SQM Database](../logging-and-database/sqm-database.md) for the schema, [Run Records](../logging-and-database/run-records.md#jsonl-event-stream) for the event field reference.

---

## `lineage`: the stage stamp and the file manifest

Each transformation stamps its output with the pipeline stage that produced it, so a downstream step can check what it was handed rather than trust a parameter name. The stamp lives in `raw.info["temp"]`: it survives copy, filter, resample and crop, but not a SNIRF round trip, where readers re-stamp from the `desc-` entity in the filename.

| Function / class | Description |
|------------------|-------------|
| `Lineage` | The stamp itself: the stage, and the path it was written to |
| `stamp(raw, stage, ...)` | Mark a `Raw` as the output of that stage |
| `lineage_of(raw)` / `stage_of(raw)` / `path_from(raw)` | Read the stamp back |
| `paths_from(raws) → list[str]` | The files several objects were read from, in order, skipping any made in memory: the `Sources` of a dyad output |
| `carried_params(raw)` | The stamp's params, for a re-stamp that has to keep them. Refuses a param sharing a name with one of `stamp()`'s own arguments, which would bind twice |
| `require_stage(raw, *allowed)` | Raise unless the object carries one of those stages. What keeps an optical-density file out of a step that expects haemoglobin |
| `Recorder` | Turns the stamps into file-level provenance: it remembers which path each stage was written to, so an output's `Sources` resolves from the stamp rather than being threaded through by hand |

---

## `snirf_prep`

Helpers used by `nirspipe-prep align` and other commands that write derivative SNIRFs. Their `dataset_description.json` goes through `io.derivatives.write_dataset_description`, which refreshes the version stamp and links the input dataset under `DatasetLinks` and `SourceDatasets`.

| Function | Description |
|----------|-------------|
| `find_snirf(bids_dir, sub, ses, task, run, validate=False, desc=None) → Path` | Locate one SNIRF in BIDS using PyBIDS. |
| `bids_stem(snirf_path) → str` | Extract the BIDS filename stem (no `_nirs.snirf` suffix). |
| `deriv_nirs_dir(derivatives_dir, deriv_name, sub, ses) → Path` | Construct the per-subject derivative directory. |
| `copy_dataset_root(bids_dir, deriv_root)` | Carry `participants.tsv` and `README` across, so the subtree stands on its own as BIDS. |
| `copy_sidecars(snirf_path, stem, dest_dir, dest_stem=None)` | Copy the run's `_nirs.json` and `_channels.tsv` next to the new SNIRF, renamed onto `dest_stem` when the output takes a different name; the montage pair (`_optodes.tsv`, `_coordsystem.json`) keeps its subject-level name. |
| `annotations_to_df(raw)` | Convert `raw.annotations` to a BIDS-style events DataFrame. |

---

## `net`: choosing a port

| Function | Description |
|----------|-------------|
| `resolve_port(preferred, *, explicit, host="127.0.0.1") → int` | The port an interactive server binds to, at or just above `preferred`. |

With `explicit` set, an unbindable port raises a `SystemExit` naming it instead of moving to the next free one. Used by `nirspipe-gui --port` and the `nirspipe-rate` viewers.

---

## Package-level helpers

| Function | Description |
|----------|-------------|
| `is_optical_density(raw)` | True when the object is already optical density, so a step that converts can skip rather than convert twice. `qc.metrics._helpers.as_optical_density(raw)` returns the object itself when it is, else a converted copy. |
| `pair_of(ch) → str` | A channel's source-detector pair: `"S1_D1 760"` and `"S1_D1 hbo"` both give `"S1_D1"`. The one place a channel name is split. |
| `fisher_r_to_z(r) → ndarray` | Fisher r-to-z, elementwise, with r clipped just below ±1 and a non-finite r giving NaN. Used by resting-state and hyperscanning alike. |
| `bare_roi_map(roi_map) → dict` / `roi_overlaps(roi_map) → dict` | An ROI map with its entries reduced to pairs, and the pairs more than one region lists. See [ROI Mapping File](../configuration/roi-mapping.md). |
| `load_toml(path) → dict` | Read a TOML file, through `tomli` on Python < 3.11. |
| `unwrap_enum(val, default=None)` | The value behind an enum member, passing plain values through. |
