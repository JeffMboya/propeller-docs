# Propeller

**Propeller** is a cutting-edge orchestrator for **WebAssembly (Wasm)** workloads across the **Cloud-Edge continuum**. It enables seamless deployment of Wasm applications from powerful cloud servers to constrained microcontrollers, combining flexibility, security, and performance.

---

## 🌟 Features

- 🌐 **Cloud-Edge Orchestration**: Deploy Wasm workloads effortlessly across diverse environments, from robust cloud servers to lightweight microcontrollers.
- ⚡ **Fast Boot Times**: Take advantage of Wasm's near-instant startup for efficient workload execution.
- 📦 **FaaS Deployment**: Enable Function-as-a-Service (FaaS) capabilities for scalable and event-driven applications.
- 🖥️ **OCI Registry Support**: Push and pull Wasm workloads from OCI-compliant registries for streamlined workflow integration.
- 🔧 **WAMR on Zephyr RTOS**: Deploy lightweight Wasm workloads on constrained devices running Zephyr RTOS via the WebAssembly Micro Runtime (WAMR).
- 🛠️ **Powerful Service Mesh**: Integrates with **[SuperMQ](https://docs.supermq.abstractmachines.fr/)** for secure, efficient IoT device communication.
- 🔒 **Security at the Core**: Propeller ensures secure workload execution and communication for IoT environments.
- 🔁 **Job Orchestration**: Group tasks into jobs with parallel, sequential, or configurable execution modes.
- 🧠 **Federated Learning**: Orchestrate federated learning experiments across distributed proplets via the Manager's FL API.
- 📊 **Process Monitoring**: Collect per-task OS-level metrics (CPU, memory, disk I/O, network) and publish them over MQTT.

---

## 🛠️ How It Works

1. **Develop in WebAssembly**: Write portable, lightweight Wasm workloads for your application.
2. **Register Workloads**: Push your workloads to an OCI-compliant registry for easy deployment.
3. **Deploy Anywhere**: Use Propeller to orchestrate and manage workload deployment across the cloud, edge, and IoT devices.
4. **Monitor & Scale**: Leverage real-time monitoring and dynamic scaling to optimize your system's performance.
