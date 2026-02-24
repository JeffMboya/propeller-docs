# Reference

This page provides a quick reference for the Propeller CLI and the core data models used throughout the system.

## CLI Reference

The `propeller-cli` binary is installed to `GOBIN` by `make install`. All commands communicate with the Manager over HTTP.

### Global Flags

| Flag                     | Short | Default                   | Description                                |
| ------------------------ | ----- | ------------------------- | ------------------------------------------ |
| `--manager-url`          | `-m`  | `http://localhost:7070`   | URL of the Manager HTTP API.               |
| `--tls-verification`     | `-v`  | —                         | Enable TLS certificate verification.       |
| `--users-url`            | `-u`  | `http://localhost:9002`   | SuperMQ users service URL.                 |
| `--domains-url`          | `-d`  | `http://localhost:9003`   | SuperMQ domains service URL.               |
| `--clients-url`          | `-c`  | `http://localhost:9006`   | SuperMQ clients service URL.               |
| `--channels-url`         | `-z`  | `http://localhost:9005`   | SuperMQ channels service URL.              |
| `--content-type`         | `-t`  | —                         | HTTP Content-Type header value.            |

### Tasks

```bash
propeller-cli tasks create <name> [--cli-args <args>]
propeller-cli tasks view   <id>
propeller-cli tasks update <id>
propeller-cli tasks delete <id>
propeller-cli tasks start  <id>
propeller-cli tasks stop   <id>
propeller-cli tasks        [--offset <n>] [--limit <n>]   # list all tasks
```

| Flag       | Short | Default | Description                      |
| ---------- | ----- | ------- | -------------------------------- |
| `--offset` | `-o`  | `0`     | Pagination offset for list.      |
| `--limit`  | `-l`  | `100`   | Maximum results to return.       |

### Provision

The `provision` command runs an interactive TUI-based wizard that:

1. Creates a SuperMQ domain.
2. Creates a Manager client and N proplet clients.
3. Creates a channel and connects all clients to it.
4. Writes the resulting credentials to a `config.toml` file.

```bash
propeller-cli provision [--file-name <path>]
```

| Flag          | Short | Default       | Description                              |
| ------------- | ----- | ------------- | ---------------------------------------- |
| `--file-name` | `-f`  | `config.toml` | Output path for the generated config.    |

---

## Data Models

### Task

| Field                | Type            | Description                                                                 |
| -------------------- | --------------- | --------------------------------------------------------------------------- |
| `id`                 | string (UUID)   | Unique task identifier. Assigned by the Manager on creation.                |
| `name`               | string          | Human-readable task name.                                                   |
| `kind`               | string          | Task kind: `standard` or `federated`.                                       |
| `state`              | integer         | Current execution state (see Task States below).                            |
| `image_url`          | string          | OCI image reference for the Wasm binary (e.g. `docker.io/myorg/app:v1`).   |
| `file`               | string          | Path or identifier of a directly uploaded `.wasm` file.                     |
| `cli_args`           | array of string | Arguments passed to the Wasm runtime (e.g. `["--invoke", "add"]`).         |
| `inputs`             | array           | Numeric inputs forwarded to the Wasm function.                              |
| `env`                | object          | Environment variables injected into the Wasm execution context.             |
| `daemon`             | boolean         | If `true`, the task runs continuously until explicitly stopped.             |
| `encrypted`          | boolean         | If `true`, the Wasm binary is encrypted and requires a TEE for decryption.  |
| `kbs_resource_path`  | string          | KBS resource path used to fetch the decryption key for an encrypted task.   |
| `proplet_id`         | string (UUID)   | Pin execution to a specific proplet, bypassing the round-robin scheduler.   |
| `depends_on`         | array of string | IDs of tasks that must complete before this task starts (workflow DAG).     |
| `run_if`             | string          | Condition for execution: `success`, `failure`, or `always`.                 |
| `workflow_id`        | string (UUID)   | ID of the workflow this task belongs to.                                    |
| `job_id`             | string (UUID)   | ID of the job this task belongs to.                                         |
| `results`            | any             | Output stored after the task completes successfully.                        |
| `error`              | string          | Error message if the task failed.                                           |
| `monitoring_profile` | object          | Per-task monitoring configuration (see monitoring.md).                      |
| `schedule`           | string          | Cron expression for recurring execution (e.g. `"0 * * * *"`).              |
| `timezone`           | string          | Timezone for cron schedule evaluation (default: `UTC`).                     |
| `is_recurring`       | boolean         | `true` if a `schedule` is set.                                              |
| `next_run`           | timestamp       | Computed time of the next scheduled execution.                              |
| `priority`           | integer         | Task priority hint for the scheduler.                                       |
| `mode`               | string          | Federated learning mode: `infer` or `train`.                                |
| `start_time`         | timestamp       | Time the task started executing.                                            |
| `finish_time`        | timestamp       | Time the task completed or failed.                                          |
| `created_at`         | timestamp       | Time the task was created.                                                  |
| `updated_at`         | timestamp       | Time the task was last updated.                                             |

#### Task States

| Value | Name          | Description                                      |
| ----- | ------------- | ------------------------------------------------ |
| `0`   | `Pending`     | Task created, not yet started.                   |
| `1`   | `Scheduled`   | Task assigned to a proplet.                      |
| `2`   | `Running`     | Task is executing on a proplet.                  |
| `3`   | `Completed`   | Task finished successfully.                      |
| `4`   | `Failed`      | Task encountered an error.                       |
| `5`   | `Skipped`     | Task was skipped (e.g. dependency condition not met). |
| `6`   | `Interrupted` | Task was stopped externally.                     |

---

### Job

| Field            | Type          | Description                                                                |
| ---------------- | ------------- | -------------------------------------------------------------------------- |
| `id`             | string (UUID) | Unique job identifier.                                                     |
| `name`           | string        | Human-readable job name.                                                   |
| `execution_mode` | string        | `parallel`, `sequential`, or `configurable`.                               |
| `tasks`          | array         | List of task objects belonging to this job.                                |
| `created_at`     | timestamp     | Time the job was created.                                                  |
| `updated_at`     | timestamp     | Time the job was last updated.                                             |

#### Job Execution Modes

| Mode           | Description                                                                                   |
| -------------- | --------------------------------------------------------------------------------------------- |
| `parallel`     | All tasks in the job start simultaneously.                                                    |
| `sequential`   | Tasks start one at a time in order. A task failure stops the job (fail-fast).                |
| `configurable` | Execution mode is read from the `JOB_EXECUTION_MODE` environment variable at runtime.        |

---

### Proplet

| Field           | Type            | Description                                                     |
| --------------- | --------------- | --------------------------------------------------------------- |
| `id`            | string (UUID)   | Unique proplet identifier, set by the proplet on startup.       |
| `name`          | string          | Human-readable proplet name.                                    |
| `task_count`    | integer         | Number of tasks currently running on this proplet.              |
| `alive`         | boolean         | `true` if the most recent heartbeat was within the last 10 s.   |
| `alive_history` | array           | Last 10 heartbeat timestamps recorded by the Manager.           |
