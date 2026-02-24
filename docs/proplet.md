# Proplet

The `proplet` is a Rust-based worker that executes WebAssembly workloads and communicates with the Manager via MQTT. It connects to SuperMQ as an authenticated MQTT client, receives task commands from the Manager, fetches WebAssembly binaries (via the Proxy or directly from an OCI registry), executes them, and reports results back.

## Runtimes

The proplet supports three runtime modes, selected by environment variable:

| Mode | Description |
| ---- | ----------- |
| **Embedded Wasmtime** (default) | Runs WASM in-process using [Wasmtime](https://wasmtime.dev/) 41.0 with component-model support. No external runtime needed. |
| **Host runtime** | Delegates execution to an external WebAssembly runtime binary on the host (e.g., `wasmtime`, `wasmer`). Controlled by `PROPLET_EXTERNAL_WASM_RUNTIME`. |
| **TEE runtime** | Decrypts and executes encrypted WASM workloads inside a hardware Trusted Execution Environment (Intel TDX, AMD SEV/SNP, or Intel SGX). Auto-detected at startup. |

## Configuration

The proplet is configured using environment variables.

### Core Variables

| Environment Variable          | Description                                                                 | Default                |
| ----------------------------- | --------------------------------------------------------------------------- | ---------------------- |
| `PROPLET_LOG_LEVEL`           | Log level (`debug`, `info`, `warn`, `error`)                                | `info`                 |
| `PROPLET_INSTANCE_ID`         | A unique ID for this proplet instance. Auto-generated if empty.             | Generated UUID         |
| `PROPLET_MQTT_ADDRESS`        | Address of the MQTT broker.                                                 | `tcp://localhost:1883` |
| `PROPLET_MQTT_TIMEOUT`        | Timeout for MQTT operations (seconds).                                      | `30`                   |
| `PROPLET_MQTT_QOS`            | MQTT Quality of Service level.                                              | `2`                    |
| `PROPLET_LIVELINESS_INTERVAL` | Interval at which the proplet sends heartbeat messages to the Manager.      | `10s`                  |
| `PROPLET_DOMAIN_ID`           | SuperMQ domain ID. Required.                                                |                        |
| `PROPLET_CHANNEL_ID`          | SuperMQ channel ID. Required.                                               |                        |
| `PROPLET_CLIENT_ID`           | MQTT client ID for authentication. Required.                                |                        |
| `PROPLET_CLIENT_KEY`          | MQTT client key for authentication. Required.                               |                        |

### Runtime Variables

| Environment Variable            | Description                                                                  | Default  |
| ------------------------------- | ---------------------------------------------------------------------------- | -------- |
| `PROPLET_EXTERNAL_WASM_RUNTIME` | Path to an external Wasm runtime binary. Uses embedded Wasmtime if not set. | `""` (empty) |

### TEE Variables

| Environment Variable       | Description                                                        | Default              |
| -------------------------- | ------------------------------------------------------------------ | -------------------- |
| `PROPLET_KBS_URI`          | Key Broker Service URL. Required for encrypted workloads.          |                      |
| `PROPLET_AA_CONFIG_PATH`   | Path to the Attestation Agent configuration file.                  |                      |
| `PROPLET_LAYER_STORE_PATH` | OCI layer cache path used when pulling encrypted images.           | `/tmp/proplet/layers` |

### Monitoring Variables

| Environment Variable        | Description                               | Default |
| --------------------------- | ----------------------------------------- | ------- |
| `PROPLET_ENABLE_MONITORING` | Enable or disable OS-level task monitoring. | `true`  |

## Usage

### Using the Embedded Wasmtime Runtime

By default, the proplet uses its embedded Wasmtime runtime. Set the required credentials and start:

```bash
export PROPLET_DOMAIN_ID="your_domain_id"
export PROPLET_CHANNEL_ID="your_channel_id"
export PROPLET_CLIENT_ID="your_client_id"
export PROPLET_CLIENT_KEY="your_client_key"
./target/release/proplet
```

When using `propeller-cli provision`, these values are written to `config.toml` and the proplet reads them automatically if the env vars are not set.

### Using an External Host Runtime

Set `PROPLET_EXTERNAL_WASM_RUNTIME` to the path of the runtime binary. The proplet will invoke it as a subprocess:

```bash
export PROPLET_DOMAIN_ID="your_domain_id"
export PROPLET_CHANNEL_ID="your_channel_id"
export PROPLET_CLIENT_ID="your_client_id"
export PROPLET_CLIENT_KEY="your_client_key"
export PROPLET_EXTERNAL_WASM_RUNTIME="/usr/bin/wasmtime"
./target/release/proplet
```

CLI arguments and numeric inputs are passed through the task definition. For example, to run the `addition` example with the `wasmtime` host runtime and invoke the `add` function:

```json
{
  "name": "add",
  "cli_args": ["--invoke", "add"],
  "inputs": [10, 20]
}
```

### Running Inside a TEE

The proplet automatically detects TEE hardware at startup by checking for device files:

| TEE Type   | Device File Checked  |
| ---------- | -------------------- |
| Intel TDX  | `/dev/tdx_guest`     |
| AMD SEV/SNP | `/dev/sev`          |
| Intel SGX  | `/dev/sgx_enclave`   |

No flag is required. When a TEE is detected, the proplet logs:

```
INFO TEE detected automatically: TDX (method: device_file, details: "/dev/tdx_guest exists")
```

When no TEE is found, it runs in standard mode:

```
INFO No TEE detected, running in standard mode
```

Start the proplet in TEE mode:

```bash
export PROPLET_DOMAIN_ID="your_domain_id"
export PROPLET_CHANNEL_ID="your_channel_id"
export PROPLET_CLIENT_ID="your_client_id"
export PROPLET_CLIENT_KEY="your_client_key"
export PROPLET_MQTT_ADDRESS="your_mqtt_address"
export PROPLET_KBS_URI="http://10.0.2.2:8082"
export PROPLET_AA_CONFIG_PATH="/etc/default/proplet.toml"
./target/release/proplet
```

`PROPLET_AA_CONFIG_PATH` points to an Attestation Agent config file:

```toml
[token_configs]
[token_configs.coco_kbs]
url = "http://10.0.2.2:8082"
```

To submit an encrypted task, set `encrypted: true` and provide the `kbs_resource_path`. Do not include a `file` field:

```json
{
  "name": "add",
  "image_url": "docker.io/myorg/tee-wasm-addition:encrypted",
  "encrypted": true,
  "kbs_resource_path": "default/key/propeller-addition",
  "cli_args": ["--invoke", "add"],
  "inputs": [10, 20]
}
```

## Proplet Command Handling

### Start Command Flow

The Manager sends a start command to the proplet on the MQTT topic:

```
m/{domain_id}/c/{channel_id}/control/manager/start
```

1. The proplet parses the `StartRequest` payload containing the `AppName`.
2. A fetch request is published to the Proxy on the registry topic requesting the WASM binary.
3. The proplet waits for WASM binary chunks from the Proxy.
4. Once all chunks are received (`chunk_idx` reaches `total_chunks - 1`), the binary is assembled and validated.
5. The assembled binary is passed to the Wasmtime runtime for instantiation and execution.

### Stop Command Flow

The Manager sends a stop command on:

```
m/{domain_id}/c/{channel_id}/control/manager/stop
```

The proplet parses the `StopRequest` containing the `AppName`, stops the running Wasmtime instance, and releases all associated resources.

## Proplet Registration and Liveliness

The Manager discovers and tracks proplets through three mechanisms:

### 1. Startup Notification (`create` topic)

When a proplet starts, it publishes on:

```
m/{domain_id}/c/{channel_id}/messages/control/proplet/create
```

Payload:

```json
{
  "PropletID": "{PropletID}",
  "ChanID": "{ChannelID}"
}
```

### 2. Liveliness Updates (`alive` topic)

The proplet periodically publishes heartbeats on:

```
m/{domain_id}/c/{channel_id}/messages/control/proplet/alive
```

Payload:

```json
{
  "status": "alive",
  "PropletID": "{PropletID}",
  "ChanID": "{ChannelID}"
}
```

### 3. Last Will & Testament (LWT)

If the proplet disconnects unexpectedly, the MQTT broker publishes on the same `alive` topic:

```json
{
  "status": "offline",
  "PropletID": "{PropletID}",
  "ChanID": "{ChannelID}"
}
```

The Manager marks the proplet as unavailable upon receiving an `offline` status.

## Registry Workflow

The proplet fetches WebAssembly binaries from the Proxy in chunks over MQTT.

### 1. Fetch Request

The proplet requests a WASM binary from the Proxy:

- **Topic:** `m/{domain_id}/c/{channel_id}/registry/proplet`
- **Payload:**

```json
{
  "app_name": "{AppName}"
}
```

### 2. Image Chunks Delivery

The Proxy streams the WASM binary back as sequential chunks:

- **Topic:** `m/{domain_id}/c/{channel_id}/registry/server`
- **Payload:**

```json
{
  "app_name": "{AppName}",
  "chunk_idx": 0,
  "total_chunks": 3,
  "data": "{Base64EncodedChunkData}"
}
```

The proplet assembles all chunks in order once `chunk_idx` reaches `total_chunks - 1`.

### 3. Registry Configuration Update

The Manager can update the proplet's registry configuration dynamically:

- **Topic:** `m/{domain_id}/c/{channel_id}/control/manager/updateRegistry`
- **Payload:**

```json
{
  "registry_url": "{NewRegistryURL}",
  "registry_token": "{NewRegistryToken}"
}
```

### 4. Registry Update Acknowledgment

The proplet acknowledges the update:

- **Topic:** `m/{domain_id}/c/{channel_id}/control/manager/registry`
- **Success payload:**

```json
{ "status": "success" }
```

- **Failure payload:**

```json
{ "status": "failure", "error": "{ErrorMessage}" }
```
