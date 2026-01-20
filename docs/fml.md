# Federated Machine Learning

The FML (Federated Machine Learning) system is implemented as a **workload-agnostic** federated learning framework built on top of Propeller's generic orchestration capabilities. The system enables distributed machine learning training across multiple edge devices (proplets) without centralizing raw data.

## Key Design Principles

1. **Manager is Workload-Agnostic**: The Manager service has no FL-specific logic. It simply orchestrates task distribution and forwards messages.
2. **External Coordinator**: FL-specific logic (aggregation, round management, model versioning) is handled by an external FML Coordinator service.
3. **MQTT-Based Communication**: All components communicate via SuperMQ MQTT topics for asynchronous, scalable message passing.
4. **WASM-Based Training**: Training workloads are executed as WebAssembly modules for portability and security.

## Architecture

The FML system consists of the following components:

```markdown
                         ┌──────────────────────┐
                         │  External Trigger    │
                         │   Test Script        │
                         └───────────┬──────────┘
                                     │
                                     │ fl/rounds/start
                                     ▼
                         ┌──────────────────────┐
                         │     Coordinator      │
                         │  (FL Logic + FedAvg) │
                         └───────────┬──────────┘
                                     │
                                     │ fl/rounds/start
                                     ▼
                                ┌──────────┐
                                │ Manager  │
                                │(Task     │
                                │ Orchestrator)│
                                └────┬─────┘
                                     │
                                     │ Task Start Commands
                                     │ m/{domain}/c/{channel}/control/manager/start
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
  ┌──────────┐                 ┌──────────┐                 ┌──────────┐
  │ Proplet 1│                 │ Proplet 2│                 │ Proplet 3│
  │ (Wasm FL)│                 │ (Wasm FL)│                 │ (Wasm FL)│
  └────┬─────┘                 └────┬─────┘                 └────┬─────┘
       │                            │                            │
       │ fl/rounds/{round_id}/updates/{proplet_id}              │
       └────────────────────────────┼────────────────────────────┘
                                    │
                                    ▼
                             ┌─────────────┐
                             │   Manager   │
                             │  (Forwards  │
                             │  Updates)   │
                             └───────┬─────┘
                                     │
                                     │ fml/updates
                                     ▼
                         ┌──────────────────────┐
                         │     Coordinator      │
                         │ Collect + Aggregate  │
                         └───────────┬──────────┘
                                     │
                                     │ fl/models/publish
                                     ▼
                              ┌──────────────┐
                              │  Model Server│
                              │  (MQTT relay)│
                              └───────┬──────┘
                                      │
                                      │ fl/models/global_model_v{N}
                                      │ (Retained Message)
                                      ▼
                          ┌─────────────────────────┐
                          │        Proplets         │
                          │    (Model Receiver)     │
                          └─────────────────────────┘
```

### Component Overview

1. **SuperMQ MQTT Adapter**: Central message bus for all MQTT communication
2. **Manager**: Task orchestration and message forwarding
3. **FML Coordinator**: FL round management and aggregation
4. **Model Server**: Model storage and distribution
5. **Proplets**: WASM execution environments (Rust or embedded)
6. **Client WASM**: Training workload module

## Component Details

### Manager Service

**Responsibilities**:

- Task creation and lifecycle management
- Proplet selection and task distribution
- MQTT message forwarding (workload-agnostic)

**Key Functions**:

#### Round Start Handling

- **Subscribes to**: `fl/rounds/start`
- **Purpose**: Listens for FL round start messages and launches tasks for each participant
- **Process**:
  1. Parses round start message containing:
     - `round_id`: Unique identifier for the round
     - `model_uri`: MQTT topic for the base model (e.g., `fl/models/global_model_v0`)
     - `task_wasm_image`: OCI image reference for WASM module (optional, can use `file` field)
     - `participants`: List of proplet IDs to participate
     - `hyperparams`: Training hyperparameters (epochs, lr, batch_size)
     - `k_of_n`: Minimum number of updates required for aggregation
     - `timeout_s`: Round timeout in seconds
  2. Validates each participant (checks if proplet exists and is alive)
  3. Creates a task for each participant with:
     - Environment variables: `ROUND_ID`, `MODEL_URI`, `HYPERPARAMS`
     - Task name: `fl-round-{round_id}-{proplet_id}`
     - Pinned to specific proplet
  4. Starts each task immediately after creation

#### Update Forwarding

- **Subscribes to**: `fl/rounds/+/updates/+` (wildcard pattern)
- **Purpose**: Forwards FL update messages verbatim to the FML coordinator
- **Process**:
  1. Extracts `round_id` and `proplet_id` from topic: `fl/rounds/{round_id}/updates/{proplet_id}`
  2. Adds metadata: `forwarded_at` timestamp
  3. Publishes to: `fml/updates` topic
  4. Does NOT inspect, validate, or modify the update payload

**Key Design**: Manager remains completely workload-agnostic. It doesn't understand FL semantics, only forwards messages.

### FML Coordinator

**Responsibilities**:

- Round state management
- Update aggregation using FedAvg (Federated Averaging)
- Model versioning
- Round completion handling

**Key Data Structures**:

```go
type RoundState struct {
    RoundID   string
    ModelURI  string
    KOfN      int           // Minimum updates required
    TimeoutS  int           // Round timeout in seconds
    StartTime time.Time
    Updates   []Update      // Collected updates
    Completed bool
    mu        sync.Mutex    // Per-round mutex
}

type Update struct {
    RoundID      string                 `json:"round_id"`
    PropletID    string                 `json:"proplet_id"`
    BaseModelURI string                 `json:"base_model_uri"`
    NumSamples   int                    `json:"num_samples"`
    Metrics      map[string]interface{} `json:"metrics"`
    Update       map[string]interface{} `json:"update"`  // Model weights
    ForwardedAt  string                 `json:"forwarded_at,omitempty"`
}

type Model struct {
    W       []float64 `json:"w"`  // Weights
    B       float64   `json:"b"`  // Bias
    Version int       `json:"version"`
}
```

**Key Functions**:

#### Round Initialization

- **Subscribes to**: `fl/rounds/start`
- **Purpose**: Initializes round state when a round starts
- **Process**:
  1. Parses round start message
  2. Creates `RoundState` with:
     - Default `k_of_n = 3` if not specified
     - Default `timeout_s = 30` if not specified
  3. Stores in `rounds` map keyed by `round_id`
  4. Logs round initialization

#### Update Handling

- **Subscribes to**: `fml/updates`
- **Purpose**: Receives and processes FL updates from proplets
- **Process**:
  1. Parses update message
  2. **Lazy Initialization**: If round doesn't exist, creates it with defaults
     - This handles cases where tasks are started via HTTP API without MQTT round start message
  3. Adds update to round's update list
  4. Checks if `len(updates) >= k_of_n`
  5. If threshold reached:
     - Marks round as completed
     - Triggers aggregation in goroutine

#### Aggregation

- **Purpose**: Performs FedAvg aggregation and creates new global model
- **Algorithm**: Weighted Federated Averaging
  - For each update, weight by `num_samples`
  - Sum weighted updates: `aggregated_w[i] += update.w[i] * num_samples`
  - Normalize by total samples: `aggregated_w[i] /= total_samples`
- **Process**:
  1. Extracts updates from round state (with mutex protection)
  2. Initializes aggregated model from first update's structure
  3. Performs weighted aggregation
  4. Normalizes by total samples
  5. Increments model version
  6. Saves model to file: `/tmp/fl-models/global_model_v{N}.json`
  7. Publishes model to `fl/models/publish` (model server picks it up)
  8. Publishes round completion to `fl/rounds/{round_id}/complete`

#### Timeout Checking

- **Purpose**: Background goroutine that checks for round timeouts
- **Process**:
  1. Runs every 5 seconds
  2. For each incomplete round:
     - Calculates elapsed time
     - If `elapsed >= timeout_s`:
       - Marks round as completed
       - Triggers aggregation if updates exist

**State Management**:

- Round state stored in memory (`rounds` map)
- Thread-safe with mutexes for map access and per-round update lists
- Model version counter with mutex protection

### Model Server

**Responsibilities**:

- Model storage and persistence
- Model distribution via MQTT (retained messages)
- Initial default model creation

**Key Functions**:

#### Model Publishing

- **Subscribes to**: `fl/models/publish`
- **Purpose**: Receives new models from coordinator and publishes them
- **Process**:
  1. Parses model JSON from coordinator
  2. Saves to file: `/tmp/fl-models/global_model_v{N}.json`
  3. Publishes to MQTT topic: `fl/models/global_model_v{N}` (retained message)
  4. Retained messages allow clients to get the model immediately when subscribing

#### Model Watching

- **Purpose**: Background goroutine that watches for new model files
- **Process**:
  1. Polls `/tmp/fl-models/` directory every 5 seconds
  2. Finds latest model version
  3. If new version detected, publishes to MQTT
  4. Uses retained messages for immediate availability

**Initialization**:

- Creates default model `global_model_v0.json` if none exists
- Publishes default model on startup

### Proplet Service

The FL implementation differs between the Rust proplet and embedded proplet, each optimized for their respective execution environments.

#### Rust Proplet FL Implementation

**Responsibilities**:

- WASM module execution using Wasmtime runtime
- Task result collection
- FL update publication (HTTP-first with MQTT fallback)

**FL Update Detection and Publishing**:

- **Detection**: Checks for `ROUND_ID` environment variable in task
- **Process**:
  1. After WASM execution completes, captures stdout
  2. Parses stdout as JSON (expected FL update format)
  3. If valid JSON and `ROUND_ID` present:
     - **Primary**: Attempts HTTP POST to coordinator: `{COORDINATOR_URL}/update`
     - **Fallback**: If HTTP fails, publishes to SuperMQ MQTT topic: `fl/rounds/{round_id}/updates/{proplet_id}`
  4. If JSON parsing fails, logs warning (unless task failed)

**HTTP-First Strategy**:

- Rust proplet uses HTTP POST for direct communication with coordinator
- Falls back to MQTT if HTTP fails (network issues, coordinator unavailable)
- Provides better performance and lower latency when coordinator is accessible
- MQTT fallback ensures reliability in distributed scenarios

**WASM Execution**:

- Uses Wasmtime runtime (external) or wazero (embedded)
- Executes exported function from WASM module
- Captures stdout as task result
- Sets environment variables from task spec including:
  - `ROUND_ID`: Round identifier
  - `MODEL_URI`: Model MQTT topic or HTTP URL
  - `COORDINATOR_URL`: HTTP coordinator endpoint
  - `HYPERPARAMS`: JSON-encoded hyperparameters

#### Embedded Proplet FL Implementation

**Responsibilities**:

- WASM module execution using WAMR (WebAssembly Micro Runtime)
- FL task detection and data fetching
- Host function registration for WASM modules
- FL update publication via MQTT

**FL Task Detection and Workflow**:

- **Detection**: Checks for `ROUND_ID` environment variable in task start command
- **Process**:
  1. **Task Detection**: Proplet detects FL task via `ROUND_ID` environment variable
  2. **PROPLET_ID Setup**: Sets `PROPLET_ID` from `config.client_id` (Manager-known identity)
  3. **Model Fetching**: Fetches model from Model Registry via HTTP GET
     - URL: `{MODEL_REGISTRY_URL}/models/{version}`
     - Stores result for WASM module access
     - Falls back to MQTT subscription if HTTP fails
  4. **Dataset Fetching**: Fetches dataset from Local Data Store via HTTP GET
     - URL: `{DATA_STORE_URL}/datasets/{proplet_id}`
     - Stores result for WASM module access
  5. **WASM Execution**: Executes WASM module with host functions registered
  6. **Host Function Calls**: WASM module calls host functions to get:
     - `PROPLET_ID` via `get_proplet_id()`
     - `MODEL_DATA` via `get_model_data()`
     - `DATASET_DATA` via `get_dataset_data()`
  7. **Training**: WASM module performs local training and outputs JSON update to stdout
  8. **Update Submission**: Proplet captures stdout, parses JSON, and publishes to SuperMQ MQTT:
     - Topic: `fl/rounds/{round_id}/updates/{proplet_id}`
     - Message: JSON update with `round_id`, `proplet_id`, `update`, `metrics`, etc.

**Host Functions**:

The embedded proplet provides three host functions for WASM modules:

1. **`get_proplet_id(ret_offset *i32, ret_len *i32) -> i32`**
   - Returns PROPLET_ID as string in WASM linear memory
   - Used by WASM module to identify itself in FL updates

2. **`get_model_data(ret_offset *i32, ret_len *i32) -> i32`**
   - Returns MODEL_DATA JSON string in WASM linear memory
   - Contains global model weights fetched from Model Registry

3. **`get_dataset_data(ret_offset *i32, ret_len *i32) -> i32`**
   - Returns DATASET_DATA JSON string in WASM linear memory
   - Contains local dataset fetched from Data Store

**Environment Variable Fallback**:

For compatibility with TinyGo/WASI, the embedded proplet also sets these as environment variables:

- `PROPLET_ID`: Set from `config.client_id`
- `MODEL_DATA`: Set from fetched model JSON
- `DATASET_DATA`: Set from fetched dataset JSON

**WASM Execution**:

- Uses WAMR runtime (compiled into Zephyr firmware)
- Supports both interpreter mode and AOT compilation
- Executes exported function from WASM module
- Captures stdout for update extraction
- Memory-constrained environment (40 KB heap pool)

#### Differences from Rust Proplet

| Feature | Rust Proplet | Embedded Proplet |
| --- | --- | --- |
| **Update Submission** | HTTP POST (primary), MQTT (fallback) | MQTT only |
| **Data Access** | Environment variables | Host functions + env vars |
| **Model Fetching** | WASM handles via MQTT/HTTP | Proplet fetches before execution |
| **Dataset Fetching** | WASM handles | Proplet fetches before execution |
| **Runtime** | Wasmtime (external) or wazero | WAMR (embedded in Zephyr) |
| **Memory Constraints** | Host system resources | 40 KB heap pool |

### Client WASM Module

**Purpose**: Sample FL training workload that runs on each proplet

**Implementation Details**:

#### Environment Variables

- `ROUND_ID`: Current round identifier
- `MODEL_URI`: MQTT topic for base model (e.g., `fl/models/global_model_v0`)
- `HYPERPARAMS`: JSON string with training hyperparameters

#### Training Process

1. **Model Initialization**:
   - Default model: `{"w": [0.0, 0.0, 0.0], "b": 0.0}`
   - In production, would subscribe to `MODEL_URI` MQTT topic to fetch model

2. **Local Training**:
   - Simulates training with random gradient updates
   - Applies learning rate: `weights[i] += lr * gradient`
   - Runs for specified number of epochs

3. **Update Generation**:
   - Creates update JSON with `round_id`, `proplet_id`, `num_samples`, `metrics`, and `update` fields
   - Outputs to stdout (captured by proplet)

## Message Flow and MQTT Topics

| Topic | Publisher | Subscriber | Purpose |
| --- | --- | --- | --- |
| `fl/rounds/start` | External trigger / Test script | Manager, Coordinator | Round start message |
| `fl/rounds/{round_id}/updates/{proplet_id}` | Proplet | Manager | FL update from proplet |
| `fml/updates` | Manager | Coordinator | Forwarded updates for aggregation |
| `fl/models/publish` | Coordinator | Model Server | New aggregated model |
| `fl/models/global_model_v{N}` | Model Server | Clients (future) | Published model (retained) |
| `fl/rounds/{round_id}/complete` | Coordinator | External (future) | Round completion notification |

### Complete Message Flow

```markdown
1. Round Start
   ┌─────────────────────────────────────┐
   │ External trigger / test script     │
   │ publishes to: fl/rounds/start      │
   └──────────────┬──────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
   ┌────▼────┐        ┌─────▼─────┐
   │ Manager │        │ Coordinator│
   │ (creates│        │ (initializes│
   │  tasks) │        │  round)    │
   └────┬────┘        └────────────┘
        │
        │ Publishes start commands
        │ to: m/{domain}/c/{channel}/control/manager/start
        │
   ┌────▼──────────────────────────────┐
   │ Proplets receive start commands   │
   │ Execute WASM modules              │
   └────┬──────────────────────────────┘
        │
        │ After training, publish updates
        │ to: fl/rounds/{round_id}/updates/{proplet_id}
        │
   ┌────▼────┐
   │ Manager │ (forwards verbatim)
   └────┬────┘
        │
        │ Publishes to: fml/updates
        │
   ┌────▼─────┐
   │Coordinator│ (aggregates when k_of_n reached)
   └────┬──────┘
        │
        │ Publishes to: fl/models/publish
        │
   ┌────▼──────────┐
   │ Model Server │ (saves and republishes)
   └────┬──────────┘
        │
        │ Publishes to: fl/models/global_model_v{N}
        │ (retained message)
```

### Message Formats

#### Round Start Message

```json
{
  "round_id": "r-1768464194",
  "model_uri": "fl/models/global_model_v0",
  "task_wasm_image": "oci://example/fl-client-wasm:latest",
  "participants": ["proplet-1", "proplet-2", "proplet-3"],
  "hyperparams": {
    "epochs": 1,
    "lr": 0.01,
    "batch_size": 16
  },
  "k_of_n": 3,
  "timeout_s": 30
}
```

#### FL Update Message (from Proplet)

```json
{
  "round_id": "r-1768464194",
  "base_model_uri": "fl/models/global_model_v0",
  "num_samples": 512,
  "metrics": {
    "loss": 0.73
  },
  "update": {
    "w": [0.12, -0.05, 1.01],
    "b": 0.33
  }
}
```

#### Aggregated Model

```json
{
  "w": [0.08, -0.02, 0.95],
  "b": 0.25,
  "version": 1
}
```

#### Round Completion Message

```json
{
  "round_id": "r-1768464194",
  "model_version": 1,
  "model_topic": "fl/models/global_model_v1",
  "num_updates": 3,
  "total_samples": 1536,
  "completed_at": "2026-01-12T10:30:45Z"
}
```

## Implementation Details

### Thread Safety

**Coordinator**:

- Mutexes protect `rounds` map access
- Per-round mutexes protect individual round's update lists
- Model version counter with mutex protection

**Manager**:

- Stateless forwarding (no shared state)
- Goroutines for async message handling

### Error Handling

**Coordinator**:

- Lazy round initialization if update received before round start
- Timeout handling for incomplete rounds
- JSON parsing errors logged and ignored

**Proplet**:

- WASM execution errors published as task results
- JSON parsing failures logged (non-fatal for non-FL tasks)

### Persistence

**Current Implementation**:

- Round state: In-memory only (lost on restart)
- Models: Persisted to `/tmp/fl-models/` (Docker volume)
- Model versions: Incremented counter (persists across restarts if coordinator restarts)

**Future Enhancements**:

- Database-backed round state
- Model version history
- Round completion logs

### Scalability Considerations

**Current Limitations**:

- Single coordinator instance (no horizontal scaling)
- In-memory round state (limited by RAM)
- No distributed locking for coordinator

**Design for Future**:

- Coordinator can be made stateless with external storage
- Multiple coordinators with consistent hashing
- Distributed locking for aggregation

## Running the Demo

For detailed instructions on running the FML demo, see the [FML Demo README](../examples/fl-demo/README.md).

The demo includes:

- **Rust Proplet Demo**: Full FL workflow with Rust proplets using SuperMQ
- **Embedded Proplet Demo**: FL workflow with embedded proplets (see [Embedded Proplet FL README](../examples/fl-embedded/README.md))

### Quick Start

1. **Prerequisites**:
   - Docker and Docker Compose
   - Go 1.21+ (for building WASM client)
   - SuperMQ infrastructure (see demo README for setup)

2. **Build WASM Client**:

   ```bash
   cd examples/fl-demo/client-wasm
   GOOS=wasip1 GOARCH=wasm go build -o fl-client.wasm fl-client.go
   ```

3. **Start Services**:

   ```bash
   cd examples/fl-demo
   docker compose up -d
   ```

4. **Trigger a Round**:
   - Use test scripts (see demo README)
   - Or publish to MQTT topic `fl/rounds/start` via SuperMQ

5. **Monitor Progress**:
   - Coordinator logs: `docker compose logs -f coordinator`
   - Manager logs: `docker compose logs -f manager`
   - Check aggregated models: `docker compose exec model-server ls -la /tmp/fl-models/`

## Monitoring

### Expected Log Patterns

**Coordinator**:

```bash
INFO Received update round_id=r-... proplet_id=proplet-1 total_updates=1 k_of_n=3
INFO Received update round_id=r-... proplet_id=proplet-2 total_updates=2 k_of_n=3
INFO Received update round_id=r-... proplet_id=proplet-3 total_updates=3 k_of_n=3
INFO Round complete: received k_of_n updates
INFO Aggregating updates round_id=r-... num_updates=3
INFO Aggregated model saved round_id=r-... version=1
INFO Published model to model server version=1
INFO Published round completion round_id=r-...
```

**Manager**:

```bash
INFO launched task for FL round participant round_id=r-... proplet_id=proplet-1
INFO launched task for FL round participant round_id=r-... proplet_id=proplet-2
INFO launched task for FL round participant round_id=r-... proplet_id=proplet-3
INFO forwarded FL update to coordinator round_id=r-... proplet_id=proplet-1
INFO forwarded FL update to coordinator round_id=r-... proplet_id=proplet-2
INFO forwarded FL update to coordinator round_id=r-... proplet_id=proplet-3
```

**Proplet**:

```bash
INFO Received start command for task <task-id>
INFO Executing WASM module for task <task-id>
INFO Detected FL task via ROUND_ID env. Publishing update to coordinator topic: fl/rounds/r-.../updates/proplet-1
INFO Successfully published FL update to coordinator: fl/rounds/r-.../updates/proplet-1
```

### Monitoring MQTT Topics

Subscribe to SuperMQ MQTT topics to monitor messages:

```bash
# Monitor all FL topics (connects to SuperMQ MQTT adapter)
mosquitto_sub -h localhost -p 1883 -t "fl/#" -v

# Monitor coordinator updates
mosquitto_sub -h localhost -p 1883 -t "fml/updates" -v

# Monitor round completions
mosquitto_sub -h localhost -p 1883 -t "fl/rounds/+/complete" -v
```
