# RAD-FL
RAD-FL: Resource-aware dynamic topology learning for communication-efficient decentralized federated learning on heterogeneous on-device physical AI platforms.

# RAD-FL: Resource-Aware Dynamic Topology Learning for Decentralized Federated Learning

This repository contains the implementation of **RAD-FL**, a resource-aware dynamic topology learning framework for decentralized federated learning (DFL) on heterogeneous edge devices.

RAD-FL dynamically adapts inter-group communication links according to both statistical and system information while keeping model exchange and aggregation decentralized.

The framework combines:

- update-based statistical alignment,
- measured local computation time,
- measured peer-to-peer RTT,
- straggler-aware link filtering,
- bounded topology adaptation,
- Metropolis-Hastings graph weighting,
- sample-aware hybrid aggregation, and
- peer-to-peer model exchange.

The topology manager acts as a **control-plane component**. It receives topology-related information such as local model updates, training times, and RTT measurements, determines the communication topology and mixing weights, and returns the corresponding information to the clients.  
**Client models are exchanged and aggregated directly between neighboring devices.**

---

## Repository Structure

The main implementation files are:

```text
RAD-FL/
│
├── topology.py
├── hardware_topology_manager.py
├── hardware_training_client.py
├── hardware_training_client_lr_decay.py
│
├── cifer_models.py
├── mnist_fashion_mnist_models.py
│
├── config/
│   ├── experiment_config.yaml
│   ├── experiment_config_lr_decay.yaml
│   └── peer_config.yaml
│
├── checkpoints/
│   ├── initial_model_seed42.pt
│   ├── initial_model_seed45.pt
│   └── initial_model_seed51.pt
│
├── network/
│   ├── __init__.py
│   ├── http_client.py
│   └── http_server.py
│
└── requirement.txt
