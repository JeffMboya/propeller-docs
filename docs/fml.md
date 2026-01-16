# Testing Federated Learning Demo

This guide provides step-by-step instructions to test the federated learning demo
application.

## Prerequisites

- Docker and Docker Compose installed
- Go 1.21 or later installed (for building the WASM client)
- Python 3 with the `requests` library installed
- All services can communicate on the `fl-demo` Docker network

## Recent Implementation Updates

The FML demo has been updated with several critical fixes to ensure robust
operation.

### Environment Variable Passing

- **Proplet runtime**
  - Environment variables are correctly passed to WASM modules
  - External Wasmtime uses `--env KEY=VALUE` flags before the WASM file argument
  - Embedded Wasmtime injects variables via `WasiCtxBuilder`
- **Verification**
  - Proplet logs show:
    - `Setting X environment variables for task`
    - `--env` flags when using external Wasmtime

### FL Update Publishing

- **Proplet detection**
  - FL tasks are detected via the `ROUND_ID` environment variable
- **Topic structure**
  - Updates are published to:
    - `fl/rounds/{round_id}/updates/{proplet_id}`
- **Verification**
  - Proplet logs show:
    - `Detected FL task via ROUND_ID env. Publishing update to coordinator topic`

### Manager Async Handlers

- **Non-blocking**
  - MQTT handlers run asynchronously to prevent deadlocks
- **Forwarding**
  - Updates are forwarded from:
    - `fl/rounds/+/updates/+`
    - to `fml/updates`
- **Verification**
  - Manager logs do not contain:
    - `failed to publish due to timeout reached`

### Coordinator Lazy Initialization

- **On-demand rounds**
  - Round state is created when the first update arrives
- **Defaults**
  - `k_of_n = 3`
  - `timeout_s = 60`
  - `model_uri` taken from the update
- **Verification**
  - Logs show:
    - `Received update for unknown round, lazy initializing`
  - Logs do not show:
    - `unknown round, ignoring`

## Runtime Options

The demo supports two runtime modes.

### Embedded Wasmtime (Default)

- Uses embedded Wasmtime compiled into the proplet binary
- Faster startup with no external process
- Full feature support

### External Wasmtime CLI

- Uses the `wasmtime` CLI as an external process
- Useful for debugging or CLI-specific testing
- Enabled by setting:
  - `PROPLET_EXTERNAL_WASM_RUNTIME=/usr/local/bin/wasmtime`
- The Dockerfile includes Wasmtime CLI version 39.0.1

> **Note**
>
> The current `compose.yaml` uses external Wasmtime. Remove
> `PROPLET_EXTERNAL_WASM_RUNTIME` to revert to embedded mode.

## Step 1: Build the WASM Client

Build the federated learning client WebAssembly module.

```bash
cd /home/jeff-mboya/Documents/propeller/examples/fl-demo/client-wasm
GOOS=wasip1 GOARCH=wasm go build -o fl-client.wasm fl-client.go
cd ..
