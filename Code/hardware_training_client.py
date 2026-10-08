#!/usr/bin/env python3
"""
Production hardware client for the proposed Dynamic Hardware-Aware D-Clique DFL.

Designed for an 8-device heterogeneous cluster:
  - Raspberry Pi 5: CPU
  - NVIDIA Jetson Orin Nano: CUDA/Tegra

Core guarantees
---------------
1. Exact shared model initialization before Round 1 (byte-identical checkpoint + SHA256).
2. Synchronized Round-1 release barrier across all eight devices.
3. Strict and robust CPU/CUDA verification using device *types* (cuda == cuda:0 by type).
4. Jetson-safe cuDNN defaults: benchmark=False, deterministic=True.
5. Explicit torch.cuda.synchronize() around local-training wall-clock measurements.
6. JSON-safe network metadata (torch.device/Path never sent raw).
7. One persistent .log plus round/startup CSV files per client.
8. Model exchange only with W(t)-active peers; aggregation remains decentralized.
9. Real local training time every round; lightweight cross-clique RTT probes only on
   topology-update rounds (e.g. 5,10,15,...) for hardware-aware topology learning.
10. Common global-validation evaluation after aggregation every round, plus global-test
    evaluation on the final round only. These datasets are NEVER used for training.

The PC topology manager is control-plane only: it posts W(t) to each client's
/control/topology endpoint and reads update vectors/reports. It never aggregates models.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import io
import json
import logging
import os
import platform
import random
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import aiohttp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from flask import jsonify, request
from torch.utils.data import DataLoader, TensorDataset

from models import GNLeNet
from network.http_client import DFLClient
from network.http_server import DFLServer


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj or {}


def load_config(peer_path: str, experiment_path: str) -> Dict[str, Any]:
    peer_cfg = load_yaml(peer_path)
    exp_cfg = load_yaml(experiment_path)
    if "peers" not in peer_cfg:
        raise KeyError("peer_config.yaml must contain a top-level 'peers:' mapping")
    merged = dict(peer_cfg)
    merged.update(exp_cfg)
    merged["peers"] = {int(k): dict(v) for k, v in merged["peers"].items()}
    return merged


def json_safe(value: Any) -> Any:
    """Recursively convert metadata to JSON-serializable Python values."""
    if isinstance(value, (torch.device, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        # Tensors should not normally be embedded in metadata. This fallback is
        # intentionally explicit rather than allowing json.dumps() to crash.
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def safe_json_dumps(value: Any, **kwargs: Any) -> str:
    return json.dumps(json_safe(value), **kwargs)


def cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sync_cuda(device: torch.device) -> None:
    """Synchronize only when the selected runtime device is CUDA."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


# -----------------------------------------------------------------------------
# Logging / CSV
# -----------------------------------------------------------------------------

def configure_logging(client_id: int, log_dir: str) -> Tuple[logging.Logger, Path]:
    root_dir = Path(log_dir).expanduser().resolve()
    root_dir.mkdir(parents=True, exist_ok=True)
    log_path = root_dir / f"client_{client_id}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    formatter = logging.Formatter(
        fmt=f"[Client {client_id}] %(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    root.addHandler(fh)

    logger = logging.getLogger(f"Client-{client_id}")
    logger.info("=" * 80)
    logger.info("Hardware DFL client process starting")
    logger.info("Persistent log file: %s", log_path)
    return logger, log_path


def append_csv(path: Path, row: Dict[str, Any], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    cleaned = json_safe(row)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(cleaned)


STARTUP_FIELDS = [
    "utc", "event", "run_id", "client_id", "hardware_type", "expected_device",
    "selected_device", "torch_version", "compiled_cuda", "cuda_available", "gpu_name",
    "init_sha256", "checkpoint_path", "checkpoint_size_bytes", "model_parameter_device",
    "actual_start_utc", "status_request_rtt_seconds", "corrected_wait_seconds",
]

ROUND_FIELDS = [
    "utc", "run_id", "method", "round", "client_id", "hardware_type", "device",
    "gpu_name", "init_sha256", "local_train_samples", "local_val_samples",
    "global_val_samples", "global_test_samples", "global_val_sha256", "global_test_sha256",
    "local_epochs", "batch_size", "learning_rate", "local_train_loss",
    "local_training_seconds", "pre_local_val_loss", "pre_local_val_accuracy",
    "post_local_val_loss", "post_local_val_accuracy", "accuracy_gain",
    "global_val_loss", "global_val_accuracy", "global_val_eval_seconds",
    "global_test_loss", "global_test_accuracy", "global_test_eval_seconds",
    "waiting_time_seconds", "total_bytes_received", "active_peer_ids",
    "topology_update_probe", "peer_rtt_seconds", "round_wall_seconds",
]

PEER_FIELDS = [
    "utc", "run_id", "round", "client_id", "peer_id", "ready_wait_seconds",
    "download_seconds", "total_peer_wait_seconds", "bytes_received", "status_polls",
]


# -----------------------------------------------------------------------------
# Jetson / Raspberry Pi runtime handling
# -----------------------------------------------------------------------------

def configure_torch_runtime() -> None:
    # Tegra-safe, reproducible defaults. In particular, do not enable cuDNN
    # autotuning/benchmark search on Jetson Orin Nano.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def detect_device(
    client_id: int,
    local_cfg: Dict[str, Any],
    strict_expected_device: bool,
    logger: logging.Logger,
) -> Tuple[torch.device, Dict[str, Any]]:
    expected = str(local_cfg.get("expected_device", "auto")).strip().lower()
    if expected not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Invalid expected_device={expected!r}; use auto/cpu/cuda")

    cuda_available = bool(torch.cuda.is_available())
    if strict_expected_device and expected == "cuda" and not cuda_available:
        raise RuntimeError(
            f"Client {client_id}: expected CUDA, but torch.cuda.is_available() is False. "
            "Refusing silent CPU fallback."
        )

    if expected == "cpu":
        device = torch.device("cpu")
    elif expected == "cuda":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cuda:0" if cuda_available else "cpu")

    gpu_name: Optional[str] = None
    if cuda_available:
        try:
            gpu_name = str(torch.cuda.get_device_name(0))
        except Exception as exc:  # metadata failure should not hide CUDA availability
            gpu_name = f"unavailable ({exc})"

    info = {
        "client_id": int(client_id),
        "hardware_type": str(local_cfg.get("hardware_type", "unknown")),
        "expected_device": expected,
        "selected_device": str(device),
        "selected_device_type": torch.device(device).type,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_compiled_cuda": None if torch.version.cuda is None else str(torch.version.cuda),
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "gpu_name": gpu_name,
        "detected_utc": utc_now_iso(),
    }
    logger.info("Runtime detected: %s", safe_json_dumps(info, sort_keys=True))
    return device, info


def warmup_cuda_with_retry(
    device: torch.device,
    logger: logging.Logger,
    retries: int = 2,
) -> None:
    """
    Initialize Tegra CUDA gently.

    NvMapMemAllocInternalTagged messages produced by the Tegra kernel during the
    first unified-memory allocation are not treated as a failure if the PyTorch
    operation succeeds. If PyTorch itself raises RuntimeError on the first tiny
    allocation, retry once after clearing the cache before declaring failure.
    """
    if torch.device(device).type != "cuda":
        return

    last_exc: Optional[BaseException] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            x = torch.empty((16,), dtype=torch.float32, device=device)
            x.add_(1.0)
            torch.cuda.synchronize(device)
            del x
            logger.info("CUDA warm-up successful on attempt %d", attempt)
            return
        except RuntimeError as exc:
            last_exc = exc
            logger.warning(
                "CUDA first-allocation RuntimeError on attempt %d/%d: %s. "
                "Tegra first-allocation issues are retried once before failing.",
                attempt, retries, exc,
            )
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            time.sleep(1.0)

    raise RuntimeError(f"CUDA warm-up failed after {retries} attempts: {last_exc}")


def assert_model_device_type(model: nn.Module, device: torch.device) -> None:
    """Correct comparison: cuda and cuda:0 are the same device TYPE."""
    param_type = torch.device(next(model.parameters()).device).type
    selected_type = torch.device(device).type
    if param_type != selected_type:
        raise RuntimeError(
            f"Model device TYPE mismatch: selected={selected_type}, parameter={param_type} "
            f"(full parameter device={next(model.parameters()).device})"
        )


# -----------------------------------------------------------------------------
# Exact shared initialization
# -----------------------------------------------------------------------------

def create_initial_checkpoint_bytes(seed: int) -> Tuple[bytes, str]:
    """Coordinator only: create and serialize the initialization exactly once."""
    cpu_rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        model = GNLeNet()  # CPU creation is deliberate for cross-platform identity.
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        bio = io.BytesIO()
        torch.save(state, bio)
        payload = bio.getvalue()
        return payload, sha256_bytes(payload)
    finally:
        torch.random.set_rng_state(cpu_rng_state)


def load_torch_bytes(payload: bytes) -> Any:
    try:
        return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(io.BytesIO(payload), map_location="cpu")


def persist_and_load_initial_model(
    payload: bytes,
    expected_sha256: str,
    seed: int,
    device: torch.device,
    checkpoint_dir: str,
) -> Tuple[GNLeNet, Path, str, int]:
    actual = sha256_bytes(payload)
    if actual != expected_sha256:
        raise RuntimeError(
            f"Initial checkpoint SHA256 mismatch before save: expected={expected_sha256} actual={actual}"
        )

    ckpt_dir = Path(checkpoint_dir).expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"initial_model_seed{int(seed)}.pt"
    tmp = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, ckpt_path)

    saved = ckpt_path.read_bytes()
    saved_hash = sha256_bytes(saved)
    if saved_hash != expected_sha256:
        raise RuntimeError(
            f"Checkpoint changed on disk: expected={expected_sha256} actual={saved_hash}"
        )

    state = load_torch_bytes(saved)
    model = GNLeNet()
    model.load_state_dict(state, strict=True)
    model.to(device)
    assert_model_device_type(model, device)
    sync_cuda(device)
    return model, ckpt_path, saved_hash, len(saved)


# -----------------------------------------------------------------------------
# Local dataset loading
# -----------------------------------------------------------------------------

def safe_torch_load_dataset(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tensor_pair_from_object(obj: Any, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(obj, TensorDataset):
        tensors = obj.tensors
        if len(tensors) != 2:
            raise TypeError(f"TensorDataset in {path} must contain exactly two tensors")
        return tensors[0], tensors[1]
    if isinstance(obj, dict):
        for x_key, y_key in (
            ("images", "labels"),
            ("data", "targets"),
            ("x", "y"),
            ("features", "labels"),
        ):
            if x_key in obj and y_key in obj:
                return torch.as_tensor(obj[x_key]), torch.as_tensor(obj[y_key])
    if isinstance(obj, (list, tuple)) and len(obj) == 2:
        return torch.as_tensor(obj[0]), torch.as_tensor(obj[1])
    raise TypeError(
        f"Unsupported dataset object in {path}. Expected dict(images/labels), "
        "dict(data/targets), (X,y), or TensorDataset."
    )


def resolve_dataset_file(base: Path, explicit: Optional[str], candidates: Sequence[str]) -> Path:
    paths: List[Path] = []
    if explicit:
        paths.append(base / explicit)
    paths.extend(base / name for name in candidates)
    for path in paths:
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError(
        "No compatible dataset file found. Checked:\n  " + "\n  ".join(str(p) for p in paths)
    )


def load_local_datasets(local_cfg: Dict[str, Any]) -> Tuple[TensorDataset, TensorDataset, Path, Path]:
    base = Path(str(local_cfg["dataset_path"])).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Configured dataset_path does not exist: {base}")

    train_path = resolve_dataset_file(
        base,
        local_cfg.get("train_file"),
        ["local_train.pt", "train.pt", "train_data.pt"],
    )
    val_path = resolve_dataset_file(
        base,
        local_cfg.get("val_file"),
        ["local_validation.pt", "local_val.pt", "validation.pt", "val.pt", "test.pt"],
    )

    x_train, y_train = tensor_pair_from_object(safe_torch_load_dataset(train_path), train_path)
    x_val, y_val = tensor_pair_from_object(safe_torch_load_dataset(val_path), val_path)
    return TensorDataset(x_train, y_train), TensorDataset(x_val, y_val), train_path, val_path


def load_global_evaluation_datasets(
    local_cfg: Dict[str, Any],
    evaluation_cfg: Dict[str, Any],
) -> Tuple[TensorDataset, TensorDataset, Path, Path, str, str]:
    """Load common evaluation sets only; these are never passed to the optimizer."""
    base = Path(str(local_cfg["dataset_path"])).expanduser().resolve()
    global_val_path = resolve_dataset_file(
        base,
        evaluation_cfg.get("global_validation_file"),
        ["global_validation.pt"],
    )
    global_test_path = resolve_dataset_file(
        base,
        evaluation_cfg.get("global_test_file"),
        ["global_test.pt"],
    )

    x_gv, y_gv = tensor_pair_from_object(
        safe_torch_load_dataset(global_val_path), global_val_path
    )
    x_gt, y_gt = tensor_pair_from_object(
        safe_torch_load_dataset(global_test_path), global_test_path
    )
    return (
        TensorDataset(x_gv, y_gv),
        TensorDataset(x_gt, y_gt),
        global_val_path,
        global_test_path,
        sha256_file(global_val_path),
        sha256_file(global_test_path),
    )


# -----------------------------------------------------------------------------
# Local learning / aggregation
# -----------------------------------------------------------------------------

def get_parameter_snapshot(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters()}


def compute_update_vector(before: Dict[str, torch.Tensor], model: nn.Module) -> torch.Tensor:
    current = dict(model.named_parameters())
    parts = []
    for name, old in before.items():
        parts.append((old - current[name].detach().cpu()).reshape(-1))
    return torch.cat(parts)


def train_local_model(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    local_epochs: int,
    device: torch.device,
) -> Tuple[float, float]:
    """Return (sample-weighted loss, hardware-accurate wall-clock training seconds)."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    # Required for accurate CUDA timing: finish any prior CUDA work BEFORE t0.
    sync_cuda(device)
    t0 = time.perf_counter()

    for _ in range(int(local_epochs)):
        for images, labels in loader:
            images = images.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(device, non_blocking=(device.type == "cuda"))
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            n = int(labels.size(0))
            total_loss += float(loss.detach().item()) * n
            total_samples += n

    # Required for accurate CUDA timing: wait for all kernels BEFORE t1.
    sync_cuda(device)
    elapsed = time.perf_counter() - t0
    return total_loss / max(1, total_samples), float(elapsed)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=(device.type == "cuda"))
        labels = labels.to(device, non_blocking=(device.type == "cuda"))
        outputs = model(images)
        loss = criterion(outputs, labels)
        n = int(labels.size(0))
        total_loss += float(loss.detach().item()) * n
        total_correct += int((outputs.argmax(dim=1) == labels).sum().item())
        total_samples += n
    sync_cuda(device)
    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


def timed_evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    # Keep evaluation timing separate from algorithmic round/training timings.
    sync_cuda(device)
    t0 = time.perf_counter()
    loss, acc = evaluate_model(model, loader, criterion, device)
    elapsed = time.perf_counter() - t0
    return loss, acc, float(elapsed)


def validate_mixing_matrix(
    W: torch.Tensor,
    n: int,
    aggregation_mode: str = "metropolis",
) -> torch.Tensor:
    """Validate a received mixing matrix without changing it.

    All supported modes must be finite, non-negative, and row-stochastic.
    Metropolis-Hastings is additionally symmetric and column-stochastic.
    Sample-aware and hybrid weighting are generally directional after row
    normalization, so symmetry is intentionally not required for those modes.
    """
    W = torch.as_tensor(W, dtype=torch.float64, device="cpu")
    if tuple(W.shape) != (n, n):
        raise ValueError(f"Invalid W shape={tuple(W.shape)}, expected {(n, n)}")
    if not torch.isfinite(W).all():
        raise ValueError("W contains NaN or Inf values")
    if torch.min(W).item() < -1e-10:
        raise ValueError("W contains negative weights")

    tol = 1e-6
    row_sums = W.sum(dim=1)
    row_error = torch.max(torch.abs(row_sums - 1.0)).item()
    if row_error > tol:
        raise ValueError(
            "W rows do not sum to 1 within tolerance: "
            f"max_error={row_error:.3e}, row_sums={row_sums.tolist()}"
        )

    mode = str(aggregation_mode).strip().lower()
    if mode not in {"metropolis", "sample_aware", "hybrid"}:
        raise ValueError(f"Unknown aggregation_mode={aggregation_mode!r}")

    if mode == "metropolis":
        symmetry_error = torch.max(torch.abs(W - W.T)).item()
        if symmetry_error > tol:
            raise ValueError(
                "Metropolis W is not symmetric within tolerance: "
                f"max_error={symmetry_error:.3e}"
            )
        col_sums = W.sum(dim=0)
        col_error = torch.max(torch.abs(col_sums - 1.0)).item()
        if col_error > tol:
            raise ValueError(
                "Metropolis W columns do not sum to 1 within tolerance: "
                f"max_error={col_error:.3e}"
            )

    return W


def aggregate_neighbor_models(
    own_state: Dict[str, torch.Tensor],
    peer_models: Dict[int, Dict[str, torch.Tensor]],
    weights_row: torch.Tensor,
    row_index: int,
    ordered_peer_ids: Sequence[int],
) -> Dict[str, torch.Tensor]:
    weights = weights_row.detach().cpu().double()
    active = [j for j in range(len(ordered_peer_ids)) if float(weights[j]) > 1e-12]
    if abs(sum(float(weights[j]) for j in active) - 1.0) > 1e-6:
        raise ValueError("Active mixing weights do not sum to 1")

    out: Dict[str, torch.Tensor] = {}
    for key, own_tensor in own_state.items():
        if not torch.is_floating_point(own_tensor):
            out[key] = own_tensor.clone()
            continue
        acc = torch.zeros_like(own_tensor, dtype=torch.float32, device="cpu")
        for j in active:
            w = float(weights[j])
            if j == row_index:
                source = own_state
            else:
                physical_id = int(ordered_peer_ids[j])
                if physical_id not in peer_models:
                    raise KeyError(f"Missing model from active peer {physical_id}")
                source = peer_models[physical_id]
            if key not in source:
                raise KeyError(f"Peer state is missing parameter/buffer {key!r}")
            acc.add_(source[key].detach().cpu().float(), alpha=w)
        out[key] = acc.to(dtype=own_tensor.dtype)
    return out


# -----------------------------------------------------------------------------
# Local topology inbox: the PC manager posts W(t) to this client's HTTP server.
# -----------------------------------------------------------------------------

class TopologyInbox:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._values: Dict[Tuple[str, int], torch.Tensor] = {}

    def put(self, run_id: str, round_num: int, W: torch.Tensor) -> None:
        with self._lock:
            self._values[(str(run_id), int(round_num))] = W.detach().cpu().clone()

    def get(self, run_id: str, round_num: int) -> Optional[torch.Tensor]:
        with self._lock:
            W = self._values.get((str(run_id), int(round_num)))
            return None if W is None else W.detach().cpu().clone()



def install_topology_routes(server: DFLServer, inbox: TopologyInbox) -> None:
    """Add manager -> client topology-control routes without changing model P2P semantics."""

    @server.app.post("/control/topology")
    def control_topology():
        run_id = request.args.get("run_id", default="", type=str)
        round_num = request.args.get("round", type=int)
        if not run_id or round_num is None:
            return jsonify({"error": "run_id and round are required"}), 400
        raw = request.get_data(cache=False)
        if not raw:
            return jsonify({"error": "empty topology payload"}), 400
        try:
            try:
                W = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            except TypeError:
                W = torch.load(io.BytesIO(raw), map_location="cpu")
            W = torch.as_tensor(W, dtype=torch.float32, device="cpu")
            inbox.put(run_id, int(round_num), W)
            return jsonify({
                "ok": True,
                "peer_id": int(server.peer_id),
                "run_id": str(run_id),
                "round": int(round_num),
                "shape": list(W.shape),
            })
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @server.app.get("/control/topology/status")
    def control_topology_status():
        run_id = request.args.get("run_id", default="", type=str)
        round_num = request.args.get("round", type=int)
        if not run_id or round_num is None:
            return jsonify({"error": "run_id and round are required"}), 400
        W = inbox.get(run_id, int(round_num))
        return jsonify({
            "peer_id": int(server.peer_id),
            "run_id": str(run_id),
            "round": int(round_num),
            "ready": W is not None,
        })


class ManagerFinalizationSignal:
    """Thread-safe manager -> client final-round acknowledgement."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._acks: Dict[str, int] = {}

    def mark(self, run_id: str, final_round: int) -> None:
        with self._lock:
            self._acks[str(run_id)] = max(
                int(final_round),
                int(self._acks.get(str(run_id), 0)),
            )

    def acknowledged(self, run_id: str, final_round: int) -> bool:
        with self._lock:
            return int(self._acks.get(str(run_id), 0)) >= int(final_round)


def install_finalization_routes(
    server: DFLServer,
    signal: ManagerFinalizationSignal,
) -> None:
    """Keep the client HTTP server alive until the manager has final data."""

    @server.app.post("/control/finalize")
    def control_finalize():
        data = request.get_json(silent=True) or {}
        run_id = str(data.get("run_id", ""))
        final_round = data.get("final_round")
        if not run_id or final_round is None:
            return jsonify({"error": "run_id and final_round are required"}), 400
        try:
            final_round = int(final_round)
        except (TypeError, ValueError):
            return jsonify({"error": "final_round must be an integer"}), 400
        signal.mark(run_id, final_round)
        return jsonify({
            "ok": True,
            "peer_id": int(server.peer_id),
            "run_id": run_id,
            "final_round": final_round,
        })


async def wait_for_manager_finalization(
    signal: ManagerFinalizationSignal,
    run_id: str,
    final_round: int,
    timeout_seconds: float,
    poll_seconds: float,
) -> None:
    deadline = time.perf_counter() + float(timeout_seconds)
    while time.perf_counter() < deadline:
        if signal.acknowledged(run_id, final_round):
            return
        await asyncio.sleep(float(poll_seconds))
    raise TimeoutError(
        f"Timed out waiting for manager finalization acknowledgement "
        f"for run={run_id} final_round={final_round}"
    )


async def wait_for_topology(
    inbox: TopologyInbox,
    run_id: str,
    round_num: int,
    timeout_seconds: float,
    poll_seconds: float,
) -> torch.Tensor:
    deadline = time.perf_counter() + float(timeout_seconds)
    while time.perf_counter() < deadline:
        W = inbox.get(run_id, round_num)
        if W is not None:
            return W
        await asyncio.sleep(float(poll_seconds))
    raise TimeoutError(f"Timed out waiting for W for run={run_id} round={round_num}")


# -----------------------------------------------------------------------------
# Lightweight hardware/network probe
# -----------------------------------------------------------------------------

async def measure_cross_clique_rtt(
    http: DFLClient,
    peers: Dict[int, Dict[str, Any]],
    client_id: int,
    samples: int,
) -> Dict[str, Optional[float]]:
    """Probe only potential inter-clique candidates, and only when called."""
    local_clique = int(peers[client_id]["clique_id"])
    targets = [
        pid for pid in sorted(peers)
        if pid != client_id and int(peers[pid]["clique_id"]) != local_clique
    ]
    result: Dict[str, Optional[float]] = {}
    for pid in targets:
        values: List[float] = []
        for _ in range(max(1, int(samples))):
            t0 = time.perf_counter()
            health = await http.get_health(pid, request_timeout=3.0)
            elapsed = time.perf_counter() - t0
            if health is not None:
                values.append(float(elapsed))
            await asyncio.sleep(0.01)
        result[str(pid)] = float(statistics.median(values)) if values else None
    return result


# -----------------------------------------------------------------------------
# Main client
# -----------------------------------------------------------------------------

async def run_client(client_id: int, peer_config_path: str, experiment_config_path: str) -> None:
    configure_torch_runtime()
    config = load_config(peer_config_path, experiment_config_path)
    peers: Dict[int, Dict[str, Any]] = config["peers"]
    ordered_peer_ids = sorted(peers)

    if ordered_peer_ids != list(range(1, 9)):
        raise ValueError(f"Expected physical peer IDs exactly 1..8; found {ordered_peer_ids}")
    if client_id not in peers:
        raise ValueError(f"Client {client_id} missing from peer configuration")

    experiment_cfg = config.get("experiment", {})
    startup_cfg = config.get("startup", {})
    training_cfg = config.get("training", {})
    topology_cfg = config.get("topology", {})
    aggregation_cfg = config.get("aggregation", {})
    evaluation_cfg = config.get("evaluation", {})
    network_cfg = config.get("network", {})

    aggregation_mode = str(
        aggregation_cfg.get("mode", "metropolis")
    ).strip().lower()

    aggregation_alpha = aggregation_cfg.get("alpha", None)
    if aggregation_alpha is not None:

          aggregation_alpha = float(aggregation_alpha)
    run_id = str(experiment_cfg.get("run_id", "proposed_seed42"))
    method = str(experiment_cfg.get("method", "proposed_dclique"))

    if method != "proposed_dclique":
        raise ValueError(
            f"This client implements proposed_dclique only. Got method={method!r}. "
            "Baseline strategies will use the same runtime in separate method modules."
        )

    seed = int(experiment_cfg.get("seed", 42))
    coordinator_id = int(startup_cfg.get("coordinator_client_id", 1))
    log_dir = str(startup_cfg.get("log_dir", "logs"))
    checkpoint_dir = str(startup_cfg.get("checkpoint_dir", "checkpoints"))
    strict_device = bool(startup_cfg.get("strict_expected_device", True))

    logger, log_path = configure_logging(client_id, log_dir)

    if aggregation_mode == "hybrid":
          logger.info(
                "AGGREGATION CONFIG | mode=%s | alpha=%.3f",
                 aggregation_mode,
                 aggregation_alpha,
    )
    else:
        logger.info(
            "AGGREGATION CONFIG | mode=%s",
            aggregation_mode,
        )
    local_cfg = peers[client_id]

    # Stable random seeds for local shuffling and model-independent operations.
    random.seed(seed + client_id)
    np.random.seed(seed + client_id)
    torch.manual_seed(seed + client_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + client_id)

    device, device_info = detect_device(client_id, local_cfg, strict_device, logger)
    warmup_cuda_with_retry(device, logger, retries=2)

    print("=" * 80)
    print(f"Client ID             : {client_id}")
    print(f"Clique                : {local_cfg.get('clique_id')}")
    print(f"Configured Hardware   : {device_info['hardware_type']}")
    print(f"Expected Device       : {device_info['expected_device']}")
    print(f"Selected Device       : {device}")
    print(f"Selected Device Type  : {torch.device(device).type}")
    print(f"PyTorch               : {device_info['torch_version']}")
    print(f"Compiled CUDA         : {device_info['torch_compiled_cuda']}")
    print(f"CUDA Available        : {device_info['cuda_available']}")
    print(f"GPU                    : {device_info['gpu_name'] or 'None (CPU)'}")
    print(f"Log File              : {log_path}")
    print("=" * 80)

    # Load ALL datasets BEFORE the synchronized release so disk I/O does not distort start fairness.
    # Only train_ds is used for optimization. Global validation/test are read-only evaluation sets.
    train_ds, val_ds, train_path, val_path = load_local_datasets(local_cfg)
    (
        global_val_ds, global_test_ds, global_val_path, global_test_path,
        global_val_sha256, global_test_sha256,
    ) = load_global_evaluation_datasets(local_cfg, evaluation_cfg)
    batch_size = int(training_cfg.get("batch_size", 64))
    eval_batch_size = int(training_cfg.get("eval_batch_size", 256))
    local_epochs = int(training_cfg.get("local_epochs", 3))
    learning_rate = float(training_cfg.get("learning_rate", 0.01))
    num_rounds = int(training_cfg.get("num_rounds", 2))
    round_timeout = float(training_cfg.get("round_timeout_seconds", 900))
    finalization_timeout = float(training_cfg.get("finalization_timeout_seconds", 180))
    update_interval = int(topology_cfg.get("update_interval", 5))
    rtt_samples = int(topology_cfg.get("rtt_samples", 3))
    aggregation_mode = str(aggregation_cfg.get("mode", "metropolis")).strip().lower()
    if aggregation_mode not in {"metropolis", "sample_aware", "hybrid"}:
        raise ValueError(
            "aggregation.mode must be metropolis, sample_aware, or hybrid"
        )

    generator = torch.Generator().manual_seed(seed + 1000 + client_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=False,  # conservative for Tegra unified-memory systems
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    global_val_loader = DataLoader(
        global_val_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    global_test_loader = DataLoader(
        global_test_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    logger.info(
        "Data ready | train=%d local_val=%d global_val=%d global_test=%d | "
        "train_path=%s local_val_path=%s global_val_path=%s global_test_path=%s",
        len(train_ds), len(val_ds), len(global_val_ds), len(global_test_ds),
        train_path, val_path, global_val_path, global_test_path,
    )
    logger.info(
        "GLOBAL EVALUATION DATA VERIFIED | global_val_sha256=%s | global_test_sha256=%s",
        global_val_sha256, global_test_sha256,
    )

    # Start local P2P server. Add topology inbox BEFORE the Flask thread starts.
    topology_inbox = TopologyInbox()
    finalization_signal = ManagerFinalizationSignal()
    server = DFLServer(peer_id=client_id, config=config, port=int(local_cfg["port"]))
    install_topology_routes(server, topology_inbox)
    install_finalization_routes(server, finalization_signal)
    server.set_runtime_info(json_safe(device_info))
    server.run_threaded()
    await asyncio.sleep(1.0)

    http = DFLClient(
        peer_id=client_id,
        peer_config=peers,
        timeout=float(network_cfg.get("timeout_seconds", 10)),
        retry_attempts=int(network_cfg.get("retry_attempts", 2)),
        poll_interval=float(network_cfg.get("poll_interval_seconds", 0.10)),
    )

    startup_timeout = float(startup_cfg.get("startup_health_timeout_seconds", 180))
    init_timeout = float(startup_cfg.get("initialization_timeout_seconds", 180))
    barrier_timeout = float(startup_cfg.get("barrier_timeout_seconds", 180))
    poll_seconds = float(network_cfg.get("poll_interval_seconds", 0.10))

    logger.info("Waiting for all 8 peer servers...")
    await http.wait_for_all_peers_healthy(timeout=startup_timeout, request_timeout=2.0)
    logger.info("All eight peer servers are healthy")

    # Exact shared initialization. Client 1 coordinates checkpoint bytes only;
    # this is NOT model aggregation and happens once before Round 1.
    if client_id == coordinator_id:
        init_payload, init_hash = create_initial_checkpoint_bytes(seed)
        server.set_initial_checkpoint(init_payload, init_hash, seed, run_id)
        logger.info(
            "Published exact shared initialization | seed=%d sha256=%s bytes=%d",
            seed, init_hash, len(init_payload),
        )

    payload, expected_hash, init_seed = await http.download_initial_checkpoint(
        coordinator_peer_id=coordinator_id,
        run_id=run_id,
        timeout=init_timeout,
    )
    if int(init_seed) != seed:
        raise RuntimeError(f"Initialization seed mismatch: local={seed} coordinator={init_seed}")

    model, ckpt_path, init_hash, ckpt_size = persist_and_load_initial_model(
        payload=payload,
        expected_sha256=expected_hash,
        seed=seed,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )
    assert_model_device_type(model, device)

    logger.info(
        "INITIALIZATION VERIFIED | sha256=%s | checkpoint=%s | size=%d | param_device=%s",
        init_hash, ckpt_path, ckpt_size, next(model.parameters()).device,
    )
    print("\nEXACT INITIAL MODEL VERIFIED")
    print(f"  Absolute checkpoint : {ckpt_path}")
    print(f"  File size           : {ckpt_size:,} bytes")
    print(f"  SHA256              : {init_hash}")
    print(f"  Parameter device    : {next(model.parameters()).device}")

    log_root = Path(log_dir).expanduser().resolve()
    startup_csv = log_root / f"client_{client_id}_startup.csv"
    round_csv = log_root / f"client_{client_id}_round_metrics.csv"
    peer_csv = log_root / f"client_{client_id}_peer_metrics.csv"

    append_csv(
        startup_csv,
        {
            "utc": utc_now_iso(), "event": "initialization_verified", "run_id": run_id,
            "client_id": client_id, "hardware_type": device_info["hardware_type"],
            "expected_device": device_info["expected_device"], "selected_device": str(device),
            "torch_version": device_info["torch_version"],
            "compiled_cuda": device_info["torch_compiled_cuda"],
            "cuda_available": device_info["cuda_available"], "gpu_name": device_info["gpu_name"],
            "init_sha256": init_hash, "checkpoint_path": str(ckpt_path),
            "checkpoint_size_bytes": ckpt_size,
            "model_parameter_device": str(next(model.parameters()).device),
        },
        STARTUP_FIELDS,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Fetch W(1) BEFORE registering ready. This guarantees that after the common
    # release there is no topology-download delay before local Round-1 training.
    W_round = validate_mixing_matrix(
        await wait_for_topology(
            topology_inbox, run_id, 1, round_timeout, poll_seconds
        ),
        len(ordered_peer_ids),
        aggregation_mode=aggregation_mode,
    )
    logger.info(
        "Round-1 topology already cached before startup barrier | aggregation=%s",
        aggregation_mode,
    )

    ready_meta = json_safe({
        **device_info,
        "model_parameter_device": str(next(model.parameters()).device),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_size_bytes": ckpt_size,
    })
    await http.register_startup_ready(
        coordinator_peer_id=coordinator_id,
        run_id=run_id,
        init_sha256=init_hash,
        device_info=ready_meta,
    )

    logger.info("Waiting at synchronized Round-1 start barrier")
    start_meta = await http.wait_for_synchronized_start(
        coordinator_peer_id=coordinator_id,
        run_id=run_id,
        timeout=barrier_timeout,
    )
    actual_start_utc = datetime.fromtimestamp(
        float(start_meta["actual_start_unix"]), tz=timezone.utc
    ).isoformat()
    logger.info(
        "SYNCHRONIZED ROUND-1 RELEASE | utc=%s | init_sha256=%s",
        actual_start_utc, init_hash,
    )
    append_csv(
        startup_csv,
        {
            "utc": utc_now_iso(), "event": "round1_start_released", "run_id": run_id,
            "client_id": client_id, "hardware_type": device_info["hardware_type"],
            "expected_device": device_info["expected_device"], "selected_device": str(device),
            "torch_version": device_info["torch_version"],
            "compiled_cuda": device_info["torch_compiled_cuda"],
            "cuda_available": device_info["cuda_available"], "gpu_name": device_info["gpu_name"],
            "init_sha256": init_hash, "checkpoint_path": str(ckpt_path),
            "checkpoint_size_bytes": ckpt_size,
            "actual_start_utc": actual_start_utc,
            "status_request_rtt_seconds": start_meta.get("status_request_rtt_seconds"),
            "corrected_wait_seconds": start_meta.get("corrected_wait_seconds"),
        },
        STARTUP_FIELDS,
    )

    # IMPORTANT: Round 1 training starts immediately after the barrier. No HTTP
    # report is sent before the first local training call.
    start_report_payload = json_safe({
        **start_meta,
        "actual_start_utc": actual_start_utc,
        "init_sha256": init_hash,
        "selected_device": str(device),
    })

    row_index = ordered_peer_ids.index(client_id)

    for round_num in range(1, num_rounds + 1):
        round_wall_t0 = time.perf_counter()

        if round_num > 1:
            W_round = validate_mixing_matrix(
                await wait_for_topology(
                    topology_inbox, run_id, round_num, round_timeout, poll_seconds
                ),
                len(ordered_peer_ids),
                aggregation_mode=aggregation_mode,
            )

        weights_i = W_round[row_index]
        active_indices = [
            j for j in range(len(ordered_peer_ids))
            if float(weights_i[j]) > 1e-12
        ]
        active_peer_ids = [
            int(ordered_peer_ids[j]) for j in active_indices if j != row_index
        ]

        logger.info(
            "ROUND %d START | SAMPLE-AWARE | device=%s | active_peers=%s | weights=%s",
            round_num, device, active_peer_ids,
            [round(float(weights_i[j]), 8) for j in active_indices],
        )

        before = get_parameter_snapshot(model)
        train_loss, training_seconds = train_local_model(
            model, train_loader, optimizer, criterion, local_epochs, device
        )
        assert_model_device_type(model, device)

        update_vector = compute_update_vector(before, model)
        server.publish_update(update_vector, round_num, run_id)

        pre_loss, pre_acc = evaluate_model(model, val_loader, criterion, device)
        pre_state = cpu_state_dict(model)
        server.publish_pre_model(pre_state, round_num, run_id)

        neighbor_models, comm = await http.wait_and_fetch_active_models(
            active_peer_ids=active_peer_ids,
            round_num=round_num,
            run_id=run_id,
            timeout=round_timeout,
        )

        post_state = aggregate_neighbor_models(
            own_state=pre_state,
            peer_models=neighbor_models,
            weights_row=weights_i,
            row_index=row_index,
            ordered_peer_ids=ordered_peer_ids,
        )
        model.load_state_dict(post_state, strict=True)
        assert_model_device_type(model, device)
        sync_cuda(device)

        post_loss, post_acc = evaluate_model(model, val_loader, criterion, device)

        # Only on topology-update rounds, probe all potential cross-clique
        # candidates using tiny health requests (not full model transfers).
        do_probe = (round_num % update_interval == 0)
        peer_rtt_seconds: Dict[str, Optional[float]] = {}
        if do_probe:
            peer_rtt_seconds = await measure_cross_clique_rtt(
                http=http,
                peers=peers,
                client_id=client_id,
                samples=rtt_samples,
            )

        # Freeze the algorithmic round timing BEFORE common global evaluation so
        # research-only evaluation does not inflate reported training/communication time.
        round_wall_seconds = time.perf_counter() - round_wall_t0

        global_val_loss = None
        global_val_acc = None
        global_val_eval_seconds = 0.0
        if bool(evaluation_cfg.get("global_validation_every_round", True)):
            global_val_loss, global_val_acc, global_val_eval_seconds = timed_evaluate_model(
                model, global_val_loader, criterion, device
            )

        global_test_loss = None
        global_test_acc = None
        global_test_eval_seconds = 0.0
        if (
            round_num == num_rounds
            and bool(evaluation_cfg.get("global_test_final_round_only", True))
        ):
            global_test_loss, global_test_acc, global_test_eval_seconds = timed_evaluate_model(
                model, global_test_loader, criterion, device
            )

        per_peer_transfer = comm.get("per_peer", {}) or {}
        for peer_id, meta in per_peer_transfer.items():
            append_csv(
                peer_csv,
                {
                    "utc": utc_now_iso(), "run_id": run_id, "round": round_num,
                    "client_id": client_id, "peer_id": int(peer_id),
                    "ready_wait_seconds": meta.get("ready_wait_seconds"),
                    "download_seconds": meta.get("download_seconds"),
                    "total_peer_wait_seconds": meta.get("total_peer_wait_seconds"),
                    "bytes_received": meta.get("bytes_received"),
                    "status_polls": meta.get("status_polls"),
                },
                PEER_FIELDS,
            )

        report = json_safe({
            "run_id": run_id,
            "method": method,
            "round": round_num,
            "client_id": client_id,
            "hardware_type": device_info["hardware_type"],
            "selected_device": str(device),
            "selected_device_type": torch.device(device).type,
            "gpu_name": device_info["gpu_name"],
            "torch_version": device_info["torch_version"],
            "compiled_cuda": device_info["torch_compiled_cuda"],
            "init_sha256": init_hash,
            "local_train_samples": len(train_ds),
            "local_val_samples": len(val_ds),
            "global_val_samples": len(global_val_ds),
            "global_test_samples": len(global_test_ds),
            "global_val_sha256": global_val_sha256,
            "global_test_sha256": global_test_sha256,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "local_train_loss": train_loss,
            "local_training_seconds": training_seconds,
            "pre_local_val_loss": pre_loss,
            "pre_local_val_accuracy": pre_acc,
            "post_local_val_loss": post_loss,
            "post_local_val_accuracy": post_acc,
            "accuracy_gain": post_acc - pre_acc,
            "global_val_loss": global_val_loss,
            "global_val_accuracy": global_val_acc,
            "global_val_eval_seconds": global_val_eval_seconds,
            "global_test_loss": global_test_loss,
            "global_test_accuracy": global_test_acc,
            "global_test_eval_seconds": global_test_eval_seconds,
            "active_peer_ids": active_peer_ids,
            "waiting_time_seconds": comm.get("waiting_time_seconds", 0.0),
            "total_bytes_received": comm.get("total_bytes_received", 0),
            "per_peer_transfer": per_peer_transfer,
            "peer_rtt_seconds": peer_rtt_seconds,
            "topology_update_probe": do_probe,
            "round_wall_seconds": round_wall_seconds,
            "reported_utc": utc_now_iso(),
        })

        # Guard explicitly before the report is exposed through Flask/jsonify.
        safe_json_dumps(report)
        server.set_round_report(report, round_num, run_id)
        server.mark_round_complete(round_num, run_id)

        append_csv(
            round_csv,
            {
                "utc": utc_now_iso(), "run_id": run_id, "method": method,
                "round": round_num, "client_id": client_id,
                "hardware_type": device_info["hardware_type"], "device": str(device),
                "gpu_name": device_info["gpu_name"], "init_sha256": init_hash,
                "local_train_samples": len(train_ds), "local_val_samples": len(val_ds),
                "global_val_samples": len(global_val_ds), "global_test_samples": len(global_test_ds),
                "global_val_sha256": global_val_sha256, "global_test_sha256": global_test_sha256,
                "local_epochs": local_epochs, "batch_size": batch_size,
                "learning_rate": learning_rate, "local_train_loss": train_loss,
                "local_training_seconds": training_seconds,
                "pre_local_val_loss": pre_loss, "pre_local_val_accuracy": pre_acc,
                "post_local_val_loss": post_loss, "post_local_val_accuracy": post_acc,
                "accuracy_gain": post_acc - pre_acc,
                "global_val_loss": global_val_loss, "global_val_accuracy": global_val_acc,
                "global_val_eval_seconds": global_val_eval_seconds,
                "global_test_loss": global_test_loss, "global_test_accuracy": global_test_acc,
                "global_test_eval_seconds": global_test_eval_seconds,
                "waiting_time_seconds": comm.get("waiting_time_seconds", 0.0),
                "total_bytes_received": comm.get("total_bytes_received", 0),
                "active_peer_ids": ";".join(str(x) for x in active_peer_ids),
                "topology_update_probe": do_probe,
                "peer_rtt_seconds": safe_json_dumps(peer_rtt_seconds, sort_keys=True),
                "round_wall_seconds": round_wall_seconds,
            },
            ROUND_FIELDS,
        )

        logger.info(
            "ROUND %d COMPLETE | train=%.4fs loss=%.5f | LOCAL PRE=%.2f%% POST=%.2f%% | "
            "GLOBAL VAL=%s | GLOBAL TEST=%s | wait=%.4fs bytes=%d | probe=%s",
            round_num, training_seconds, train_loss, pre_acc * 100.0, post_acc * 100.0,
            "n/a" if global_val_acc is None else f"{global_val_acc * 100.0:.2f}%",
            "n/a" if global_test_acc is None else f"{global_test_acc * 100.0:.2f}%",
            float(comm.get("waiting_time_seconds", 0.0)),
            int(comm.get("total_bytes_received", 0)), do_probe,
        )

        # Report the common start only after Round 1 local work is safely launched/
        # completed, so the report itself cannot delay the synchronized training start.
        if round_num == 1:
            try:
                result = await http.report_synchronized_start(
                    coordinator_peer_id=coordinator_id,
                    run_id=run_id,
                    start_meta=start_report_payload,
                )
                logger.info("Reported synchronized start after Round 1 | response=%s", result)
            except Exception as exc:
                logger.warning("Could not report synchronized start: %s", exc)

    # Do NOT exit immediately after the final local round. The manager still has
    # to download this client's final update vector and report. Because Flask runs
    # in a daemon thread, returning from run_client would terminate the HTTP server
    # and create a race (ConnectionRefusedError on the manager).
    logger.info(
        "LOCAL TRAINING FINISHED | waiting for manager finalization ACK | "
        "run=%s final_round=%d timeout=%.1fs",
        run_id, num_rounds, finalization_timeout,
    )
    await wait_for_manager_finalization(
        signal=finalization_signal,
        run_id=run_id,
        final_round=num_rounds,
        timeout_seconds=finalization_timeout,
        poll_seconds=poll_seconds,
    )
    logger.info("MANAGER FINALIZATION ACK RECEIVED | safe to stop HTTP server")

    logger.info("TRAINING COMPLETE | rounds=%d | round_csv=%s | peer_csv=%s", num_rounds, round_csv, peer_csv)
    print("\n" + "=" * 80)
    print(f"Client {client_id} training completed")
    print(f"Device            : {device}")
    print(f"Initial SHA256    : {init_hash}")
    print(f"Initial checkpoint: {ckpt_path}")
    print(f"Log               : {log_path}")
    print(f"Round CSV         : {round_csv}")
    print(f"Peer CSV          : {peer_csv}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--peer_config", default="config/peer_config.yaml")
    parser.add_argument("--experiment_config", default="config/experiment_config.yaml")
    args = parser.parse_args()
    try:
        asyncio.run(run_client(args.client_id, args.peer_config, args.experiment_config))
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception:
        logging.exception("Fatal hardware DFL client error")
        raise


if __name__ == "__main__":
    main()
