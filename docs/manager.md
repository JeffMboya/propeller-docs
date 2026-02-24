# Manager

## Overview

The Manager service is the central component of the Propeller system. It manages tasks, proplets, jobs, and workflows; handles task scheduling and execution; coordinates with proplets via MQTT; and exposes a REST HTTP API for external clients.

## Configuration

The Manager reads configuration from environment variables. If `MANAGER_DOMAIN_ID`, `MANAGER_CHANNEL_ID`, `MANAGER_CLIENT_ID`, or `MANAGER_CLIENT_KEY` are empty, it falls back to reading them from a `config.toml` file in the current directory.

### Core Variables

| Environment Variable    | Description                                                             | Default                |
| ----------------------- | ----------------------------------------------------------------------- | ---------------------- |
| `MANAGER_LOG_LEVEL`     | Log level (`debug`, `info`, `warn`, `error`).                           | `info`                 |
| `MANAGER_INSTANCE_ID`   | Unique instance ID. Auto-generated UUID if empty.                       | Generated UUID         |
| `MANAGER_MQTT_ADDRESS`  | Address of the MQTT broker.                                             | `tcp://localhost:1883` |
| `MANAGER_MQTT_QOS`      | MQTT Quality of Service level.                                          | `2`                    |
| `MANAGER_MQTT_TIMEOUT`  | Timeout for MQTT operations.                                            | `30s`                  |
| `MANAGER_DOMAIN_ID`     | SuperMQ domain ID. Required.                                            |                        |
| `MANAGER_CHANNEL_ID`    | SuperMQ channel ID. Required.                                           |                        |
| `MANAGER_CLIENT_ID`     | MQTT client ID. Required.                                               |                        |
| `MANAGER_CLIENT_KEY`    | MQTT client key. Required.                                              |                        |

### HTTP Server Variables

The HTTP server uses the prefix `MANAGER_HTTP_`. The default port is `7070`.

| Environment Variable          | Description                      | Default   |
| ----------------------------- | -------------------------------- | --------- |
| `MANAGER_HTTP_HOST`           | HTTP listen address.             | `manager` |
| `MANAGER_HTTP_PORT`           | HTTP listen port.                | `7070`    |
| `MANAGER_HTTP_SERVER_CERT`    | Path to TLS certificate file.    | `""`      |
| `MANAGER_HTTP_SERVER_KEY`     | Path to TLS private key file.    | `""`      |

### Observability Variables

| Environment Variable  | Description                                                     | Default |
| --------------------- | --------------------------------------------------------------- | ------- |
| `MANAGER_OTEL_URL`    | OpenTelemetry collector URL (e.g., `http://jaeger:4318/v1/traces`). Tracing disabled if empty. | `""` |
| `MANAGER_TRACE_RATIO` | Fraction of requests to trace (0.0–1.0).                        | `0`     |

### Storage Variables

Select the storage backend with `MANAGER_STORAGE_TYPE`. The default is in-memory (non-persistent).

| Environment Variable      | Description                        | Default              |
| ------------------------- | ---------------------------------- | -------------------- |
| `MANAGER_STORAGE_TYPE`    | Backend: `memory`, `sqlite`, `postgres`, `badger`. | `memory` |

**PostgreSQL** (when `MANAGER_STORAGE_TYPE=postgres`):

| Variable                    | Default       |
| --------------------------- | ------------- |
| `MANAGER_POSTGRES_HOST`     | `localhost`   |
| `MANAGER_POSTGRES_PORT`     | `5432`        |
| `MANAGER_POSTGRES_USER`     | `propeller`   |
| `MANAGER_POSTGRES_PASS`     | `propeller`   |
| `MANAGER_POSTGRES_DB`       | `propeller`   |
| `MANAGER_POSTGRES_SSLMODE`  | `disable`     |

**SQLite** (when `MANAGER_STORAGE_TYPE=sqlite`):

| Variable                  | Default           |
| ------------------------- | ----------------- |
| `MANAGER_SQLITE_PATH`     | `./propeller.db`  |

**Badger** (when `MANAGER_STORAGE_TYPE=badger`):

| Variable                  | Default            |
| ------------------------- | ------------------ |
| `MANAGER_BADGER_PATH`     | `./data/badger`    |

## Architectural Components

### 1. Service Interface

The `Service` interface defines all core operations: managing proplets and tasks, creating jobs and workflows, subscribing to MQTT topics, and coordinating federated learning experiments.

### 2. API Layer

HTTP endpoints are implemented with the Go-Kit library and registered on a chi router (`manager/api/transport.go`). All routes are listed in the API Reference section below.

### 3. Middleware

Three middleware layers wrap the service:

- **Logging**: Logs method name, duration, and any error for every service call.
- **Metrics**: Collects call counts and latency, exposed via Prometheus at `/metrics`.
- **Tracing**: Adds distributed trace spans using OpenTelemetry (active when `MANAGER_OTEL_URL` is set).

### 4. Storage

The storage layer is abstracted behind repository interfaces. Four backends are supported, all storing the same entities: tasks, proplets, task-proplet mappings, jobs, and metrics. The `memory` backend is ephemeral; the others (`sqlite`, `postgres`, `badger`) are persistent.

### 5. Scheduler

The current scheduler implementation is **round-robin**: it cycles through available proplets in order. A task may also target a specific proplet by setting `proplet_id` in the task definition, bypassing the scheduler.

### 6. Cron Scheduler

Tasks that include a `schedule` field (a cron expression) are managed by an internal cron scheduler. At the scheduled time, the manager starts the task automatically. The `timezone` field controls the cron timezone (default: `UTC`). The `next_run` and `is_recurring` fields reflect the computed schedule.

### 7. PubSub

The Manager subscribes to a set of MQTT topics on startup and publishes control commands to proplets. It uses the SuperMQ MQTT broker for all inter-service communication.

### 8. Health and Metrics

- `GET /health` — Returns service name, version, and instance ID.
- `GET /metrics` — Exposes Prometheus metrics for all service operations.

## API Reference

All endpoints are served on the HTTP server (default port `7070`). Pagination query parameters `offset` (default `0`) and `limit` (default `100`) are supported on list endpoints.

### Proplets

| Method   | Path                            | Description                  |
| -------- | ------------------------------- | ---------------------------- |
| `GET`    | `/proplets`                     | List all proplets.           |
| `GET`    | `/proplets/{propletID}`         | Get a proplet by ID.         |
| `DELETE` | `/proplets/{propletID}`         | Delete a proplet by ID.      |
| `GET`    | `/proplets/{propletID}/metrics` | Get metrics for a proplet.   |

### Tasks

| Method   | Path                          | Description                                         |
| -------- | ----------------------------- | --------------------------------------------------- |
| `POST`   | `/tasks`                      | Create a task. Body: JSON task object.              |
| `GET`    | `/tasks`                      | List tasks.                                         |
| `GET`    | `/tasks/{taskID}`             | Get a task by ID.                                   |
| `PUT`    | `/tasks/{taskID}`             | Update a task. Body: JSON task object.              |
| `PUT`    | `/tasks/{taskID}/upload`      | Upload a WASM file. Body: multipart form with `file` field (`.wasm` only, max 100 MB). |
| `DELETE` | `/tasks/{taskID}`             | Delete a task.                                      |
| `POST`   | `/tasks/{taskID}/start`       | Start a task.                                       |
| `POST`   | `/tasks/{taskID}/stop`        | Stop a task.                                        |
| `GET`    | `/tasks/{taskID}/metrics`     | Get metrics for a task.                             |
| `GET`    | `/tasks/{taskID}/results`     | Get the results stored for a completed task.        |

### Workflows

A workflow is a set of tasks with dependency ordering (DAG). Tasks within a workflow can declare `depends_on` and `run_if` fields to control execution order.

| Method | Path         | Description                                        |
| ------ | ------------ | -------------------------------------------------- |
| `POST` | `/workflows` | Create a workflow. Body: JSON array of task objects. |

Example request body:

```json
[
  { "name": "step-1", "image_url": "docker.io/myorg/step1:latest" },
  { "name": "step-2", "image_url": "docker.io/myorg/step2:latest", "depends_on": ["<step-1-id>"], "run_if": "success" }
]
```

### Jobs

A job groups multiple tasks and executes them with a configured execution mode.

| Method | Path                  | Description                                |
| ------ | --------------------- | ------------------------------------------ |
| `POST` | `/jobs`               | Create a job. Body: JSON job object.       |
| `GET`  | `/jobs`               | List jobs.                                 |
| `GET`  | `/jobs/{jobID}`       | Get a job by ID.                           |
| `POST` | `/jobs/{jobID}/start` | Start a job.                               |
| `POST` | `/jobs/{jobID}/stop`  | Stop a job.                                |

**Job execution modes** (set via `execution_mode` field or `JOB_EXECUTION_MODE` env var):

| Mode           | Description                                                                                |
| -------------- | ------------------------------------------------------------------------------------------ |
| `parallel`     | All tasks in the job start simultaneously.                                                 |
| `sequential`   | Tasks start one at a time in order. A task failure stops the job (fail-fast).              |
| `configurable` | Execution mode is read from the `JOB_EXECUTION_MODE` environment variable at runtime.     |

Example job creation request:

```bash
curl -X POST "http://localhost:7070/jobs" \
-H "Content-Type: application/json" \
-d '{
  "name": "my-batch-job",
  "execution_mode": "sequential",
  "tasks": [
    { "name": "task-1", "image_url": "docker.io/myorg/task1:latest" },
    { "name": "task-2", "image_url": "docker.io/myorg/task2:latest" }
  ]
}'
```

### Federated Learning

The Manager acts as an orchestrator for federated learning experiments, proxying requests to a configured FL Coordinator service.

> **Note:** FL clients should call the FL Coordinator directly for the `GET /fl/task` and `POST /fl/update` flows. These endpoints on the Manager exist for MQTT forwarding compatibility only.

| Method | Path                              | Description                                                    |
| ------ | --------------------------------- | -------------------------------------------------------------- |
| `POST` | `/fl/experiments`                 | Configure an FL experiment (Orchestrator → Coordinator step).  |
| `GET`  | `/fl/task`                        | Get an FL task. Query params: `round_id` (required), `proplet_id`. |
| `POST` | `/fl/update`                      | Post an FL model update. Body: JSON `FLUpdate` object.         |
| `POST` | `/fl/update_cbor`                 | Post an FL model update. Body: CBOR-encoded payload (`Content-Type: application/cbor`). |
| `GET`  | `/fl/rounds/{round_id}/complete`  | Get the completion status of an FL round.                      |

### System

| Method | Path       | Description                              |
| ------ | ---------- | ---------------------------------------- |
| `GET`  | `/health`  | Service health check.                    |
| `GET`  | `/metrics` | Prometheus metrics for all operations.   |

## Data Flow

### Task Lifecycle

1. **Create**: Client sends `POST /tasks` → Manager assigns a UUID, sets state to `Pending`, stores the task, returns it.
2. **Upload WASM** (optional): Client sends `PUT /tasks/{id}/upload` with a `.wasm` file.
3. **Start**: Client sends `POST /tasks/{id}/start` → Manager selects a proplet (round-robin or `proplet_id`), publishes a start command over MQTT.
4. **Execute**: Proplet receives the command, fetches the WASM binary (via Proxy or OCI registry), executes it, publishes results back over MQTT.
5. **Complete**: Manager receives results via MQTT, updates task state to `Completed` (or `Failed`), stores results.
6. **Query results**: Client calls `GET /tasks/{id}/results`.

### Task States

| Value | Name          | Description                                    |
| ----- | ------------- | ---------------------------------------------- |
| `0`   | `Pending`     | Task created, not yet started.                 |
| `1`   | `Scheduled`   | Task assigned to a proplet.                    |
| `2`   | `Running`     | Task is executing on a proplet.                |
| `3`   | `Completed`   | Task finished successfully.                    |
| `4`   | `Failed`      | Task encountered an error.                     |
| `5`   | `Skipped`     | Task was skipped (e.g., dependency condition). |
| `6`   | `Interrupted` | Task was stopped externally.                   |

### Proplet Management

- Proplets register themselves on startup by publishing to the `create` MQTT topic.
- Proplets send periodic heartbeats on the `alive` topic. The Manager retains the last 10 heartbeat timestamps per proplet.
- A proplet is marked alive if its most recent heartbeat was within the last 10 seconds.
- If a proplet disconnects, the MQTT broker delivers its Last Will message, and the Manager marks it as offline.
