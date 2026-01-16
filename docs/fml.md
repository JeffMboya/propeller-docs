# FML (Federated Machine Learning) Implementation Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Component Details](#component-details)
4. [Message Flow and MQTT Topics](#message-flow-and-mqtt-topics)
5. [Implementation Details](#implementation-details)
6. [Testing Guide](#testing-guide)
7. [Expected Results](#expected-results)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The FML (Federated Machine Learning) system is implemented as a **workload-agnostic** federated learning framework built on top of Propeller's generic orchestration capabilities. The system enables distributed machine learning training across multiple edge devices (proplets) without centralizing raw data.

### Key Design Principles

1. **Manager is Workload-Agnostic**: The Manager service has no FL-specific logic. It simply orchestrates task distribution and forwards messages.
2. **External Coordinator**: FL-specific logic (aggregation, round management, model versioning) is handled by an external FML Coordinator service.
3. **MQTT-Based Communication**: All components communicate via MQTT topics for asynchronous, scalable message passing.
4. **WASM-Based Training**: Training workloads are executed as WebAssembly modules for portability and security.

---

## Architecture

The FML system consists of the following components:

```
┌─────────────┐
│   Manager   │  ← Orchestrates tasks, forwards messages
└──────┬──────┘
       │
       ├─────────────────────────────────┐
       │                                 │
┌──────▼──────┐                   ┌──────▼──────┐
│  Proplet-1  │                   │  Proplet-2 │  ← Execute WASM training
│  Proplet-3  │                   │            │
└──────┬──────┘                   └──────┬──────┘
       │                                 │
       │  fl/rounds/{round_id}/updates/  │
       └──────────────┬──────────────────┘
                      │
              ┌───────▼────────┐
              │    Manager     │  ← Forwards to fml/updates
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │   Coordinator  │  ← Aggregates updates
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │  Model Server  │  ← Publishes aggregated models
              └────────────────┘
```

### Component Overview

1. **MQTT Broker** (Eclipse Mosquitto): Central message bus
2. **Manager**: Task orchestration and message forwarding
3. **FML Coordinator**: FL round management and aggregation
4. **Model Server**: Model storage and distribution
5. **Proplets** (3 instances): WASM execution environments
6. **Client WASM**: Training workload module

---

## Component Details

### 1. Manager Service

**Location**: `manager/service.go`

**Responsibilities**:
- Task creation and lifecycle management
- Proplet selection and task distribution
- MQTT message forwarding (workload-agnostic)

**Key Functions**:

#### `handleRoundStart(ctx context.Context)`
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

**Code Location**: `manager/service.go:485-583`

#### `handleUpdateForward(ctx context.Context)`
- **Subscribes to**: `fl/rounds/+/updates/+` (wildcard pattern)
- **Purpose**: Forwards FL update messages verbatim to the FML coordinator
- **Process**:
  1. Extracts `round_id` and `proplet_id` from topic: `fl/rounds/{round_id}/updates/{proplet_id}`
  2. Adds metadata: `forwarded_at` timestamp
  3. Publishes to: `fml/updates` topic
  4. Does NOT inspect, validate, or modify the update payload

**Code Location**: `manager/service.go:585-621`

**Key Design**: Manager remains completely workload-agnostic. It doesn't understand FL semantics, only forwards messages.

### 2. FML Coordinator

**Location**: `examples/fl-demo/coordinator/main.go`

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

#### `handleRoundStart(client mqtt.Client, msg mqtt.Message)`
- **Subscribes to**: `fl/rounds/start`
- **Purpose**: Initializes round state when a round starts
- **Process**:
  1. Parses round start message
  2. Creates `RoundState` with:
     - Default `k_of_n = 3` if not specified
     - Default `timeout_s = 30` if not specified
  3. Stores in `rounds` map keyed by `round_id`
  4. Logs round initialization

**Code Location**: `coordinator/main.go:98-142`

#### `handleUpdate(client mqtt.Client, msg mqtt.Message)`
- **Subscribes to**: `fml/updates`
- **Purpose**: Receives and processes FL updates from proplets
- **Process**:
  1. Parses update message
  2. **Lazy Initialization**: If round doesn't exist, creates it with defaults
     - This handles cases where tasks are started via HTTP API (test script) without MQTT round start message
  3. Adds update to round's update list
  4. Checks if `len(updates) >= k_of_n`
  5. If threshold reached:
     - Marks round as completed
     - Triggers `aggregateAndAdvance()` in goroutine

**Code Location**: `coordinator/main.go:144-206`

#### `aggregateAndAdvance(round *RoundState)`
- **Purpose**: Performs FedAvg aggregation and creates new global model
- **Algorithm**: Weighted Federated Averaging
  - For each update, weight by `num_samples`
  - Sum weighted updates: `aggregated_w[i] += update.w[i] * num_samples`
  - Normalize by total samples: `aggregated_w[i] /= total_samples`
- **Process**:
  1. Extracts updates from round state (with mutex protection)
  2. Initializes aggregated model from first update's structure
  3. Performs weighted aggregation:
     ```go
     weight := float64(update.NumSamples)
     totalSamples += update.NumSamples
     aggregatedW[i] += update.Update["w"][i] * weight
     aggregatedB += update.Update["b"] * weight
     ```
  4. Normalizes by total samples
  5. Increments model version
  6. Saves model to file: `/tmp/fl-models/global_model_v{N}.json`
  7. Publishes model to `fl/models/publish` (model server picks it up)
  8. Publishes round completion to `fl/rounds/{round_id}/complete`

**Code Location**: `coordinator/main.go:208-339`

#### `checkRoundTimeouts()`
- **Purpose**: Background goroutine that checks for round timeouts
- **Process**:
  1. Runs every 5 seconds
  2. For each incomplete round:
     - Calculates elapsed time
     - If `elapsed >= timeout_s`:
       - Marks round as completed
       - Triggers aggregation if updates exist

**Code Location**: `coordinator/main.go:341-364`

**State Management**:
- Round state stored in memory (`rounds` map)
- Thread-safe with `roundsMu` (RWMutex) for map access
- Per-round mutex (`round.mu`) for update list access
- Model version counter with mutex protection

### 3. Model Server

**Location**: `examples/fl-demo/model-server/main.go`

**Responsibilities**:
- Model storage and persistence
- Model distribution via MQTT (retained messages)
- Initial default model creation

**Key Functions**:

#### `handleModelPublish(_ mqtt.Client, msg mqtt.Message, client mqtt.Client, modelsDir string)`
- **Subscribes to**: `fl/models/publish`
- **Purpose**: Receives new models from coordinator and publishes them
- **Process**:
  1. Parses model JSON from coordinator
  2. Saves to file: `/tmp/fl-models/global_model_v{N}.json`
  3. Publishes to MQTT topic: `fl/models/global_model_v{N}` (retained message)
  4. Retained messages allow clients to get the model immediately when subscribing

**Code Location**: `model-server/main.go:99-129`

#### `watchAndPublishModels(client mqtt.Client, modelsDir string)`
- **Purpose**: Background goroutine that watches for new model files
- **Process**:
  1. Polls `/tmp/fl-models/` directory every 5 seconds
  2. Finds latest model version
  3. If new version detected, publishes to MQTT
  4. Uses retained messages for immediate availability

**Code Location**: `model-server/main.go:148-197`

**Initialization**:
- Creates default model `global_model_v0.json` if none exists:
  ```json
  {
    "w": [0.0, 0.0, 0.0],
    "b": 0.0,
    "version": 0
  }
  ```
- Publishes default model on startup

### 4. Proplet Service

**Location**: `proplet/src/service.rs`

**Responsibilities**:
- WASM module execution
- Task result collection
- FL update publication

**Key Functions**:

#### FL Update Detection and Publishing
- **Detection**: Checks for `ROUND_ID` environment variable in task
- **Process**:
  1. After WASM execution completes, captures stdout
  2. Parses stdout as JSON (expected FL update format)
  3. If valid JSON and `ROUND_ID` present:
     - Constructs topic: `fl/rounds/{round_id}/updates/{proplet_id}`
     - Publishes update JSON to MQTT
  4. If JSON parsing fails, logs warning (unless task failed)

**Code Location**: `proplet/src/service.rs:503-527`

**WASM Execution**:
- Uses Wasmtime runtime (Rust proplet) or WAMR (embedded proplet)
- Executes `run()` function exported from WASM module
- Captures stdout as task result
- Sets environment variables from task spec

### 5. Client WASM Module

**Location**: `examples/fl-demo/client-wasm/fl-client.go`

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
   - Creates update JSON:
     ```json
     {
       "round_id": "r-...",
       "base_model_uri": "fl/models/global_model_v0",
       "num_samples": 512,
       "metrics": {"loss": 0.73},
       "update": {
         "w": [0.12, -0.05, 1.01],
         "b": 0.33
       }
     }
     ```
   - Outputs to stdout (captured by proplet)

**Code Location**: `client-wasm/fl-client.go:24-113`

**Build Command**:
```bash
cd client-wasm
GOOS=wasip1 GOARCH=wasm go build -o fl-client.wasm fl-client.go
```

---

## Message Flow and MQTT Topics

### Topic Structure

| Topic | Publisher | Subscriber | Purpose |
|-------|-----------|------------|---------|
| `fl/rounds/start` | External trigger / Test script | Manager, Coordinator | Round start message |
| `fl/rounds/{round_id}/updates/{proplet_id}` | Proplet | Manager | FL update from proplet |
| `fml/updates` | Manager | Coordinator | Forwarded updates for aggregation |
| `fl/models/publish` | Coordinator | Model Server | New aggregated model |
| `fl/models/global_model_v{N}` | Model Server | Clients (future) | Published model (retained) |
| `fl/rounds/{round_id}/complete` | Coordinator | External (future) | Round completion notification |

### Complete Message Flow

```
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

---

## Implementation Details

### Thread Safety

**Coordinator**:
- `roundsMu` (RWMutex): Protects `rounds` map
- `round.mu` (Mutex): Protects individual round's update list
- `modelMu` (Mutex): Protects model version counter

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

---

## Testing Guide

### Prerequisites

1. **Docker and Docker Compose** installed
2. **Go 1.21+** (for building WASM client)
3. **Python 3** with `requests` library
4. All services on `fl-demo` Docker network

### Step 1: Build WASM Client

```bash
cd /home/jeff-mboya/Documents/propeller/examples/fl-demo/client-wasm
GOOS=wasip1 GOARCH=wasm go build -o fl-client.wasm fl-client.go
cd ..
```

**Expected Output**:
```
# No errors, fl-client.wasm file created
```

**Verification**:
```bash
ls -lh client-wasm/fl-client.wasm
# Should show file size ~4-5 MB
```

### Step 2: Configure MQTT Broker (Optional)

To prevent connection drops, update MQTT configuration:

```bash
cat > mqtt/mosquitto.conf << 'EOF'
listener 1883
allow_anonymous true

persistence true
persistence_location /mosquitto/data/

keepalive_interval 60
max_connections -1

max_inflight_messages 100
max_queued_messages 1000

connection_messages true
retry_interval 20

listener 9001
protocol websockets
allow_anonymous true
EOF
```

### Step 3: Start All Services

```bash
cd /home/jeff-mboya/Documents/propeller/examples/fl-demo
docker compose up -d
```

**Expected Output**:
```
[+] Running 8/8
 ✔ Container fl-demo-mqtt-1         Started
 ✔ Container fl-demo-manager-1      Started
 ✔ Container fl-demo-model-server-1 Started
 ✔ Container fl-demo-coordinator-1  Started
 ✔ Container fl-demo-proplet-1-1    Started
 ✔ Container fl-demo-proplet-2-1    Started
 ✔ Container fl-demo-proplet-3-1    Started
```

**Verification**:
```bash
docker compose ps
# All services should show "Up" status
```

### Step 4: Verify Services Are Running

**Check Manager Health**:
```bash
curl http://localhost:7070/health
```

**Expected Response**:
```json
{
  "status": "pass",
  "version": "v0.3.0",
  "commit": "22541f09d2d2fdda32f94b0322b0f4b96b276e92",
  "description": "manager service",
  "build_time": "2026-01-12T05:58:08Z",
  "instance_id": "7b14279e-b01f-4108-b257-2ecb86b76576"
}
```

**Check Service Logs**:
```bash
# Coordinator (should show MQTT connection)
docker compose logs coordinator | tail -10

# Manager (should show HTTP server listening)
docker compose logs manager | tail -10

# Proplet (should show MQTT connection)
docker compose logs proplet-1 | tail -10
```

**Expected Logs**:
- Coordinator: `FML Coordinator connected to MQTT broker`
- Manager: `manager service http server listening at localhost:7070`
- Proplet: MQTT connection messages (may have initial connection attempts)

### Step 5: Install Python Dependencies

```bash
pip3 install requests
```

Or with virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install requests
```

### Step 6: Run the Federated Learning Test

```bash
cd /home/jeff-mboya/Documents/propeller/examples/fl-demo
python3 test-fl-local.py
```

**Expected Output**:
```
Reading WASM file: client-wasm/fl-client.wasm
WASM file encoded: 4279132 characters

Creating tasks for round: r-1768464194
Participants: ['proplet-1', 'proplet-2', 'proplet-3']

Creating task for proplet-1...
  Task created: <task-id-1>
  Starting task...
  Task started successfully

Creating task for proplet-2...
  Task created: <task-id-2>
  Starting task...
  Task started successfully

Creating task for proplet-3...
  Task created: <task-id-3>
  Starting task...
  Task started successfully

✅ Successfully launched 3 tasks

Monitor progress:
  docker compose logs -f coordinator
  docker compose logs -f manager
  docker compose logs -f proplet-1

Check aggregated models:
  docker compose exec model-server ls -la /tmp/fl-models/
```

### Step 7: Monitor Federated Learning Progress

#### Watch Coordinator (Aggregation)

```bash
docker compose logs -f coordinator
```

**What to Look For**:

1. **Round Initialization** (if round start message received):
   ```
   INFO Initialized round state round_id=r-1768464194 k_of_n=3 timeout_s=60
   ```

2. **Update Reception**:
   ```
   INFO Received update round_id=r-1768464194 proplet_id=proplet-1 total_updates=1 k_of_n=3
   INFO Received update round_id=r-1768464194 proplet_id=proplet-2 total_updates=2 k_of_n=3
   INFO Received update round_id=r-1768464194 proplet_id=proplet-3 total_updates=3 k_of_n=3
   ```

3. **Aggregation Trigger**:
   ```
   INFO Round complete: received k_of_n updates round_id=r-1768464194 updates=3
   INFO Aggregating updates round_id=r-1768464194 num_updates=3
   ```

4. **Model Saving**:
   ```
   INFO Aggregated model saved round_id=r-1768464194 version=1 file=/tmp/fl-models/global_model_v1.json
   ```

5. **Model Publication**:
   ```
   INFO Published model to model server version=1
   INFO Published round completion round_id=r-1768464194 topic=fl/rounds/r-1768464194/complete
   ```

**Sample Complete Coordinator Log**:
```
2026-01-12T10:30:00Z INFO FML Coordinator connected to MQTT broker
2026-01-12T10:30:00Z INFO Subscribed to fml/updates
2026-01-12T10:30:00Z INFO Subscribed to fl/rounds/start
2026-01-12T10:30:15Z INFO Received update for unknown round, lazy initializing round_id=r-1768464194
2026-01-12T10:30:15Z INFO Received update round_id=r-1768464194 proplet_id=proplet-1 total_updates=1 k_of_n=3
2026-01-12T10:30:16Z INFO Received update round_id=r-1768464194 proplet_id=proplet-2 total_updates=2 k_of_n=3
2026-01-12T10:30:17Z INFO Received update round_id=r-1768464194 proplet_id=proplet-3 total_updates=3 k_of_n=3
2026-01-12T10:30:17Z INFO Round complete: received k_of_n updates round_id=r-1768464194 updates=3
2026-01-12T10:30:17Z INFO Aggregating updates round_id=r-1768464194 num_updates=3
2026-01-12T10:30:17Z INFO Aggregated model saved round_id=r-1768464194 version=1 file=/tmp/fl-models/global_model_v1.json
2026-01-12T10:30:17Z INFO Published model to model server version=1
2026-01-12T10:30:17Z INFO Published round completion round_id=r-1768464194 topic=fl/rounds/r-1768464194/complete
```

#### Watch Manager (Orchestration)

```bash
docker compose logs -f manager
```

**What to Look For**:

1. **Task Creation**:
   ```
   INFO launched task for FL round participant round_id=r-1768464194 proplet_id=proplet-1 task_id=<task-id>
   ```

2. **Update Forwarding**:
   ```
   INFO forwarded FL update to coordinator round_id=r-1768464194 proplet_id=proplet-1
   ```

**Sample Manager Log**:
```
2026-01-12T10:30:10Z INFO launched task for FL round participant round_id=r-1768464194 proplet_id=proplet-1 task_id=abc123
2026-01-12T10:30:11Z INFO launched task for FL round participant round_id=r-1768464194 proplet_id=proplet-2 task_id=def456
2026-01-12T10:30:12Z INFO launched task for FL round participant round_id=r-1768464194 proplet_id=proplet-3 task_id=ghi789
2026-01-12T10:30:15Z INFO forwarded FL update to coordinator round_id=r-1768464194 proplet_id=proplet-1
2026-01-12T10:30:16Z INFO forwarded FL update to coordinator round_id=r-1768464194 proplet_id=proplet-2
2026-01-12T10:30:17Z INFO forwarded FL update to coordinator round_id=r-1768464194 proplet_id=proplet-3
```

#### Watch Proplet (Training Execution)

```bash
docker compose logs -f proplet-1
```

**What to Look For**:

1. **Task Start**:
   ```
   INFO Received start command for task <task-id>
   ```

2. **WASM Execution**:
   ```
   INFO Executing WASM module for task <task-id>
   ```

3. **FL Update Publication**:
   ```
   INFO Detected FL task via ROUND_ID env. Publishing update to coordinator topic: fl/rounds/r-1768464194/updates/proplet-1
   INFO Successfully published FL update to coordinator: fl/rounds/r-1768464194/updates/proplet-1
   ```

**Sample Proplet Log**:
```
2026-01-12T10:30:10Z INFO MQTT client connected successfully
2026-01-12T10:30:13Z INFO Received start command for task abc123
2026-01-12T10:30:13Z INFO Executing WASM module for task abc123
2026-01-12T10:30:15Z INFO Detected FL task via ROUND_ID env. Publishing update to coordinator topic: fl/rounds/r-1768464194/updates/proplet-1
2026-01-12T10:30:15Z INFO Successfully published FL update to coordinator: fl/rounds/r-1768464194/updates/proplet-1
2026-01-12T10:30:15Z INFO Publishing result for task abc123
```

#### Watch All Services Together

```bash
docker compose logs -f
```

Press `Ctrl+C` to stop watching.

### Step 8: Verify Results

#### Check Aggregated Models

```bash
docker compose exec model-server ls -la /tmp/fl-models/
```

**Expected Output**:
```
total 16
drwxr-xr-x 2 root root 4096 Jan 12 10:30 .
drwxr-xr-x 1 root root 4096 Jan 12 10:30 ..
-rw-r--r-- 1 root root  123 Jan 12 10:30 global_model_v0.json
-rw-r--r-- 1 root root  125 Jan 12 10:30 global_model_v1.json
```

#### View Model Contents

```bash
docker compose exec model-server cat /tmp/fl-models/global_model_v0.json
```

**Expected Output** (Default Model):
```json
{
  "w": [0.0, 0.0, 0.0],
  "b": 0.0,
  "version": 0
}
```

```bash
docker compose exec model-server cat /tmp/fl-models/global_model_v1.json
```

**Expected Output** (Aggregated Model):
```json
{
  "w": [0.008234, -0.001567, 0.012345],
  "b": 0.002341,
  "version": 1
}
```

**Note**: Actual values will vary due to random training updates in the demo.

#### Check Task Status via Manager API

```bash
curl http://localhost:7070/tasks
```

**Expected Response**:
```json
{
  "offset": 0,
  "limit": 100,
  "total": 3,
  "tasks": [
    {
      "id": "abc123",
      "name": "fl-round-r-1768464194-proplet-1",
      "state": "Completed",
      "proplet_id": "proplet-1",
      "created_at": "2026-01-12T10:30:10Z",
      ...
    },
    {
      "id": "def456",
      "name": "fl-round-r-1768464194-proplet-2",
      "state": "Completed",
      "proplet_id": "proplet-2",
      ...
    },
    {
      "id": "ghi789",
      "name": "fl-round-r-1768464194-proplet-3",
      "state": "Completed",
      "proplet_id": "proplet-3",
      ...
    }
  ]
}
```

### Step 9: Testing Multiple Rounds

To test multiple FL rounds, simply run the test script again:

```bash
python3 test-fl-local.py
```

Each run creates a new round with a unique round ID (timestamp-based).

**Expected Behavior**:
- New round ID generated (e.g., `r-1768464195`)
- New tasks created for each proplet
- New aggregated model version (e.g., `global_model_v2`)
- Previous round state preserved in coordinator (until restart)

---

## Expected Results

### Successful Round Execution

**Timeline** (approximate):
- **T+0s**: Test script creates tasks
- **T+1-3s**: Manager launches tasks, proplets start WASM execution
- **T+3-5s**: Proplets complete training, publish updates
- **T+5-6s**: Coordinator receives all updates, aggregates
- **T+6-7s**: New model saved and published

**Expected Outcomes**:

1. **All Tasks Complete**:
   - 3 tasks created (one per proplet)
   - All tasks reach "Completed" state
   - No task failures

2. **Updates Received**:
   - Coordinator receives 3 updates (one per proplet)
   - All updates have valid JSON structure
   - Updates contain `round_id`, `proplet_id`, `num_samples`, `update` fields

3. **Aggregation Successful**:
   - Coordinator aggregates when 3 updates received (k_of_n=3)
   - New model version created (incremented from previous)
   - Model file saved to `/tmp/fl-models/global_model_v{N}.json`

4. **Model Published**:
   - Model server receives model from coordinator
   - Model published to MQTT topic `fl/models/global_model_v{N}`
   - Model available as retained message

5. **Round Completion**:
   - Completion message published to `fl/rounds/{round_id}/complete`
   - Round marked as completed in coordinator

### Sample Model Evolution

**Round 1** (Starting from default model):
- Input: `global_model_v0.json` with `w: [0.0, 0.0, 0.0]`, `b: 0.0`
- Updates from 3 proplets with random training
- Output: `global_model_v1.json` with aggregated weights (e.g., `w: [0.008, -0.002, 0.012]`)

**Round 2** (Starting from round 1 model):
- Input: `global_model_v1.json`
- Updates from 3 proplets
- Output: `global_model_v2.json` with further aggregated weights

**Pattern**: Each round refines the model based on distributed training.

### Expected Log Patterns

**Coordinator**:
```
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
```
INFO launched task for FL round participant round_id=r-... proplet_id=proplet-1
INFO launched task for FL round participant round_id=r-... proplet_id=proplet-2
INFO launched task for FL round participant round_id=r-... proplet_id=proplet-3
INFO forwarded FL update to coordinator round_id=r-... proplet_id=proplet-1
INFO forwarded FL update to coordinator round_id=r-... proplet_id=proplet-2
INFO forwarded FL update to coordinator round_id=r-... proplet_id=proplet-3
```

**Proplet**:
```
INFO Received start command for task <task-id>
INFO Executing WASM module for task <task-id>
INFO Detected FL task via ROUND_ID env. Publishing update to coordinator topic: fl/rounds/r-.../updates/proplet-1
INFO Successfully published FL update to coordinator: fl/rounds/r-.../updates/proplet-1
```

### Expected File Structure

```
/tmp/fl-models/
├── global_model_v0.json  (default, created on model-server startup)
├── global_model_v1.json  (after round 1)
├── global_model_v2.json  (after round 2)
└── ...
```

### Expected MQTT Topics and Messages

**Topic**: `fl/rounds/{round_id}/updates/{proplet_id}`
- **Publisher**: Proplet
- **Message**: FL update JSON
- **Frequency**: Once per proplet per round

**Topic**: `fml/updates`
- **Publisher**: Manager (forwarding)
- **Message**: FL update JSON with `forwarded_at` timestamp
- **Frequency**: Once per proplet per round

**Topic**: `fl/models/publish`
- **Publisher**: Coordinator
- **Message**: Aggregated model JSON
- **Frequency**: Once per completed round

**Topic**: `fl/models/global_model_v{N}`
- **Publisher**: Model Server
- **Message**: Aggregated model JSON (retained)
- **Frequency**: Once per completed round

**Topic**: `fl/rounds/{round_id}/complete`
- **Publisher**: Coordinator
- **Message**: Round completion JSON
- **Frequency**: Once per completed round

---

## Troubleshooting

### Issue: Manager Not Accessible on Port 7070

**Symptoms**:
- `curl http://localhost:7070/health` returns connection refused
- Test script fails with connection error

**Solution**:
1. Check if port is exposed in `compose.yaml`:
   ```bash
   grep -A 5 "manager:" compose.yaml | grep "ports"
   ```
   Should show: `- "7070:7070"`

2. Restart manager:
   ```bash
   docker compose restart manager
   ```

3. Wait a few seconds and verify:
   ```bash
   sleep 5
   curl http://localhost:7070/health
   ```

4. Check manager logs:
   ```bash
   docker compose logs manager | grep listening
   ```
   Should show: `manager service http server listening at localhost:7070`

### Issue: Proplets Showing "Unhealthy"

**Symptoms**:
- `docker compose ps` shows proplets as unhealthy
- Tasks not starting on proplets

**Solution**:
1. Healthcheck is disabled in compose file, so this shouldn't occur
2. If it does, restart proplets:
   ```bash
   docker compose restart proplet-1 proplet-2 proplet-3
   ```

3. Check logs for actual errors:
   ```bash
   docker compose logs proplet-1 | tail -20
   ```

### Issue: MQTT Connection Errors

**Symptoms**:
- Coordinator logs show MQTT connection failures
- Proplets can't connect to MQTT broker
- Updates not being received

**Solution**:
1. Verify MQTT broker is running:
   ```bash
   docker compose ps mqtt
   ```
   Should show "Up" status

2. Check MQTT logs:
   ```bash
   docker compose logs mqtt | tail -20
   ```

3. Verify network connectivity:
   ```bash
   docker compose exec proplet-1 getent hosts mqtt
   ```
   Should return: `172.x.x.x mqtt`

4. Restart MQTT broker:
   ```bash
   docker compose restart mqtt
   ```

### Issue: "Connection Refused" When Running Test Script

**Symptoms**:
- Test script fails immediately with connection error
- Manager health endpoint not responding

**Solution**:
1. Wait for manager to fully start (may take 10-15 seconds after `docker compose up`)
2. Verify manager is running:
   ```bash
   docker compose ps manager
   ```
3. Check manager logs:
   ```bash
   docker compose logs manager | grep -i "listening\|error"
   ```
4. Try health endpoint:
   ```bash
   curl http://localhost:7070/health
   ```
5. If still failing, restart manager:
   ```bash
   docker compose restart manager
   sleep 10
   curl http://localhost:7070/health
   ```

### Issue: Tasks Not Starting

**Symptoms**:
- Test script creates tasks but they don't start
- No proplet execution logs

**Solution**:
1. Check if proplets are alive:
   ```bash
   curl http://localhost:7070/proplets
   ```
   Should return list of proplets with `"alive": true`

2. Verify proplets are connected to MQTT:
   ```bash
   docker compose logs proplet-1 | grep -i "connected\|mqtt"
   ```
   Should show connection success messages

3. Check manager logs for errors:
   ```bash
   docker compose logs manager | grep -i error
   ```

4. Verify task creation:
   ```bash
   curl http://localhost:7070/tasks
   ```
   Check if tasks exist and their state

### Issue: No Updates Received by Coordinator

**Symptoms**:
- Proplets publish updates but coordinator doesn't receive them
- Coordinator logs show no update messages

**Solution**:
1. Verify coordinator is subscribed:
   ```bash
   docker compose logs coordinator | grep -i "subscribed"
   ```
   Should show: `Subscribed to fml/updates`

2. Check if proplets are publishing updates:
   ```bash
   docker compose logs proplet-1 | grep -i "update\|fl"
   ```
   Should show: `Successfully published FL update to coordinator`

3. Verify manager is forwarding:
   ```bash
   docker compose logs manager | grep -i "forwarded"
   ```
   Should show: `forwarded FL update to coordinator`

4. Check MQTT topic structure:
   - Proplet publishes to: `fl/rounds/{round_id}/updates/{proplet_id}`
   - Manager forwards to: `fml/updates`
   - Coordinator subscribes to: `fml/updates`

5. Verify MQTT broker is routing messages:
   ```bash
   docker compose logs mqtt | tail -20
   ```

### Issue: Aggregation Not Triggering

**Symptoms**:
- Updates received but aggregation doesn't happen
- No new model version created

**Solution**:
1. Check if `k_of_n` threshold is met:
   ```bash
   docker compose logs coordinator | grep -i "total_updates\|k_of_n"
   ```
   Should show: `total_updates=3 k_of_n=3` (or matching values)

2. Verify round state:
   - Coordinator logs should show round initialization
   - Check if round is marked as completed prematurely

3. Check for timeout:
   - If updates arrive slowly, timeout may trigger aggregation
   - Check timeout logs: `Round timeout exceeded`

4. Verify update format:
   - Updates must have valid JSON structure
   - Must include `round_id`, `proplet_id`, `update` fields

### Issue: Model Not Saved

**Symptoms**:
- Aggregation completes but no model file created
- Model server doesn't receive model

**Solution**:
1. Check coordinator logs for save errors:
   ```bash
   docker compose logs coordinator | grep -i "save\|error"
   ```

2. Verify models directory exists:
   ```bash
   docker compose exec model-server ls -la /tmp/fl-models/
   ```

3. Check file permissions:
   ```bash
   docker compose exec model-server ls -ld /tmp/fl-models/
   ```

4. Verify coordinator has write access:
   - Models directory is shared via Docker volume
   - Both coordinator and model-server should have access

### Issue: WASM Execution Fails

**Symptoms**:
- Proplet logs show WASM execution errors
- Tasks fail with WASM-related errors

**Solution**:
1. Verify WASM file is built correctly:
   ```bash
   file client-wasm/fl-client.wasm
   ```
   Should show: `WebAssembly (wasm) binary module`

2. Check WASM file size:
   ```bash
   ls -lh client-wasm/fl-client.wasm
   ```
   Should be ~4-5 MB

3. Rebuild WASM if needed:
   ```bash
   cd client-wasm
   GOOS=wasip1 GOARCH=wasm go build -o fl-client.wasm fl-client.go
   ```

4. Check proplet logs for specific error:
   ```bash
   docker compose logs proplet-1 | grep -i "wasm\|error"
   ```

5. Verify proplet has Wasmtime installed:
   ```bash
   docker compose exec proplet-1 wasmtime --version
   ```

---

## Advanced Testing

### Manual Round Start via MQTT

You can trigger rounds manually via MQTT (requires OCI registry for WASM):

```bash
mosquitto_pub -h localhost -t "fl/rounds/start" -m '{
  "round_id": "r-manual-001",
  "model_uri": "fl/models/global_model_v0",
  "task_wasm_image": "oci://example/fl-client-wasm:latest",
  "participants": ["proplet-1", "proplet-2", "proplet-3"],
  "hyperparams": {"epochs": 1, "lr": 0.01, "batch_size": 16},
  "k_of_n": 3,
  "timeout_s": 30
}'
```

### Monitoring MQTT Messages

Subscribe to MQTT topics to monitor messages:

```bash
# Monitor all FL topics
mosquitto_sub -h localhost -t "fl/#" -v

# Monitor coordinator updates
mosquitto_sub -h localhost -t "fml/updates" -v

# Monitor round completions
mosquitto_sub -h localhost -t "fl/rounds/+/complete" -v
```

### Testing with Different Hyperparameters

Modify `test-fl-local.py` to test different hyperparameters:

```python
hyperparams = {
    "epochs": 5,        # More epochs
    "lr": 0.001,        # Lower learning rate
    "batch_size": 32    # Larger batch size
}
```

### Testing with Different k_of_n Values

Modify test script to require fewer updates:

```python
# In round start message (if using MQTT directly)
"k_of_n": 2  # Aggregate with 2 updates instead of 3
```

**Note**: Coordinator defaults to `k_of_n=3` if not specified.
