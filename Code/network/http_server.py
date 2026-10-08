import copy
import io
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import torch
from flask import Flask, jsonify, request


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cpu_clone(obj: Any) -> Any:
    """Detach/clone tensors so later local training cannot mutate stored snapshots."""
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _cpu_clone(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cpu_clone(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_cpu_clone(v) for v in obj)
    return copy.deepcopy(obj)


class DFLServer:
    """
    Per-device HTTP server for the physical D-Clique experiment.

    IMPORTANT SEMANTICS
    -------------------
    * This server does NOT aggregate models.
    * A client publishes its own PRE-aggregation model locally.
    * Neighbor clients directly download that PRE model.
    * The manager may download update vectors / reports for topology learning,
      but it does not perform FedAvg.

    Data are namespaced by run_id so several baselines can be executed
    sequentially with rounds restarting at 1 without stale-model collisions.
    """

    def __init__(
        self,
        peer_id: int,
        config: Dict[str, Any],
        port: Optional[int] = None,
    ):
        self.peer_id = int(peer_id)
        self.config = config
        self.peers = self._normalize_peers(config)

        if self.peer_id not in self.peers:
            raise KeyError(f"Peer {self.peer_id} is missing from peer configuration.")

        configured_port = int(self.peers[self.peer_id]["port"])
        self.port = configured_port if port is None else int(port)

        self.app = Flask(f"dclique_peer_{self.peer_id}")
        self.logger = logging.getLogger(f"HTTPServer-{self.peer_id}")
        self._lock = threading.RLock()

        # (run_id, round_num) -> payload
        self.pre_models: Dict[Tuple[str, int], Dict[str, torch.Tensor]] = {}
        self.updates: Dict[Tuple[str, int], Any] = {}
        self.round_reports: Dict[Tuple[str, int], Dict[str, Any]] = {}

        # run_id -> latest completed round
        self.completed_rounds: Dict[str, int] = {}
        self.active_run_id: Optional[str] = None

        # Startup coordination state. Client 1 (configurable) acts only as a
        # barrier/checkpoint coordinator; it does NOT aggregate DFL models.
        self.runtime_info: Dict[str, Any] = {}
        self.initial_checkpoints: Dict[str, Dict[str, Any]] = {}
        self.startup_ready: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self.startup_started: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self.start_schedules: Dict[str, Dict[str, Any]] = {}

        self.server_start_perf = time.perf_counter()
        self.server_start_utc = _utc_now()

        network_cfg = config.get("network", {})
        self.max_model_bytes = int(
            network_cfg.get("max_model_bytes", 100 * 1024 * 1024)
        )
        self.max_rounds_in_memory = int(
            network_cfg.get("max_rounds_in_memory", 10)
        )

        self._register_routes()

    @staticmethod
    def _normalize_peers(config: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        raw = config.get("peers")
        if raw is None:
            # Temporary backward compatibility with an earlier device_map JSON.
            raw = config.get("clients")
        if raw is None:
            raise KeyError("Configuration must contain a 'peers' mapping.")

        return {int(k): dict(v) for k, v in raw.items()}

    def _peer_name(self) -> str:
        cfg = self.peers[self.peer_id]
        return str(cfg.get("name", cfg.get("device_name", f"peer_{self.peer_id}")))

    def _peer_hardware(self) -> str:
        cfg = self.peers[self.peer_id]
        return str(cfg.get("hardware_type", cfg.get("device_type", "unknown")))

    def _cleanup_old_rounds_locked(self, run_id: str) -> None:
        if self.max_rounds_in_memory <= 0:
            return

        round_nums = sorted(
            {
                rnd
                for rid, rnd in (
                    list(self.pre_models.keys())
                    + list(self.updates.keys())
                    + list(self.round_reports.keys())
                )
                if rid == run_id
            }
        )

        keep = set(round_nums[-self.max_rounds_in_memory :])

        self.pre_models = {
            key: value
            for key, value in self.pre_models.items()
            if key[0] != run_id or key[1] in keep
        }
        self.updates = {
            key: value
            for key, value in self.updates.items()
            if key[0] != run_id or key[1] in keep
        }
        self.round_reports = {
            key: value
            for key, value in self.round_reports.items()
            if key[0] != run_id or key[1] in keep
        }

    def _register_routes(self) -> None:
        @self.app.get("/health")
        def health():
            run_id = request.args.get("run_id", default=None, type=str)

            with self._lock:
                if run_id:
                    completed = int(self.completed_rounds.get(run_id, 0))
                    pre_rounds = sorted(
                        rnd for rid, rnd in self.pre_models.keys() if rid == run_id
                    )
                else:
                    completed = max(self.completed_rounds.values(), default=0)
                    pre_rounds = sorted({rnd for _, rnd in self.pre_models.keys()})

                return jsonify(
                    {
                        "status": "healthy",
                        "peer_id": self.peer_id,
                        "peer_name": self._peer_name(),
                        "hardware_type": self._peer_hardware(),
                        "runtime": dict(self.runtime_info),
                        "active_run_id": self.active_run_id,
                        "completed_round": completed,
                        "published_pre_rounds": pre_rounds,
                        "server_started_utc": self.server_start_utc,
                        "uptime_seconds": time.perf_counter() - self.server_start_perf,
                    }
                )

        @self.app.get("/info")
        def info():
            cfg = dict(self.peers[self.peer_id])
            # Do not expose future SSH secrets/passwords through HTTP.
            for secret_key in ("password", "ssh_key_path"):
                cfg.pop(secret_key, None)

            return jsonify(
                {
                    "peer_id": self.peer_id,
                    "peer_name": self._peer_name(),
                    "config": cfg,
                }
            )

        @self.app.get("/startup/initial_checkpoint")
        def startup_initial_checkpoint():
            run_id = request.args.get("run_id", default="", type=str)
            if not run_id:
                return jsonify({"error": "run_id is required"}), 400

            with self._lock:
                item = self.initial_checkpoints.get(run_id)
                if item is None:
                    return jsonify({"error": "initial checkpoint not ready"}), 404
                payload = item["payload"]
                sha256 = item["sha256"]
                seed = item["seed"]

            return payload, 200, {
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(payload)),
                "X-DFL-Init-SHA256": str(sha256),
                "X-DFL-Seed": str(seed),
                "X-DFL-Run-ID": run_id,
            }

        @self.app.post("/startup/ready")
        def startup_ready_route():
            data = request.get_json(silent=True) or {}
            run_id = str(data.get("run_id", ""))
            client_id = int(data.get("client_id", -1))
            init_sha256 = str(data.get("init_sha256", ""))

            if not run_id or client_id not in self.peers or not init_sha256:
                return jsonify({"error": "run_id, valid client_id, init_sha256 required"}), 400

            with self._lock:
                checkpoint = self.initial_checkpoints.get(run_id)
                if checkpoint is not None and init_sha256 != checkpoint["sha256"]:
                    return jsonify({
                        "error": "initialization hash mismatch",
                        "expected": checkpoint["sha256"],
                        "received": init_sha256,
                    }), 409

                ready_map = self.startup_ready.setdefault(run_id, {})
                ready_map[client_id] = {
                    "client_id": client_id,
                    "init_sha256": init_sha256,
                    "device_info": data.get("device_info", {}),
                    "ready_received_utc": _utc_now(),
                    "ready_received_perf": time.perf_counter(),
                }

                all_ready = len(ready_map) == len(self.peers)
                if all_ready and run_id not in self.start_schedules:
                    startup_cfg = self.config.get("startup", {})
                    lead = float(startup_cfg.get("start_lead_seconds", 8.0))
                    now_perf = time.perf_counter()
                    self.start_schedules[run_id] = {
                        "scheduled_perf": now_perf + lead,
                        "scheduled_utc_created": _utc_now(),
                        "lead_seconds": lead,
                    }

                return jsonify({
                    "ok": True,
                    "run_id": run_id,
                    "ready_ids": sorted(ready_map),
                    "all_ready": all_ready,
                })

        @self.app.get("/startup/status")
        def startup_status_route():
            run_id = request.args.get("run_id", default="", type=str)
            if not run_id:
                return jsonify({"error": "run_id is required"}), 400

            with self._lock:
                ready_map = self.startup_ready.get(run_id, {})
                schedule = self.start_schedules.get(run_id)
                checkpoint = self.initial_checkpoints.get(run_id)

                response = {
                    "run_id": run_id,
                    "ready_ids": sorted(ready_map),
                    "ready_count": len(ready_map),
                    "expected_count": len(self.peers),
                    "all_ready": len(ready_map) == len(self.peers),
                    "init_sha256": None if checkpoint is None else checkpoint["sha256"],
                    "start_scheduled": schedule is not None,
                }

                if schedule is not None:
                    response["start_in_seconds"] = max(
                        0.0, schedule["scheduled_perf"] - time.perf_counter()
                    )
                    response["lead_seconds"] = schedule["lead_seconds"]

                return jsonify(response)

        @self.app.post("/startup/started")
        def startup_started_route():
            data = request.get_json(silent=True) or {}
            run_id = str(data.get("run_id", ""))
            client_id = int(data.get("client_id", -1))
            if not run_id or client_id not in self.peers:
                return jsonify({"error": "run_id and valid client_id required"}), 400

            with self._lock:
                started = self.startup_started.setdefault(run_id, {})
                started[client_id] = {
                    **dict(data),
                    "coordinator_received_utc": _utc_now(),
                    "coordinator_received_perf": time.perf_counter(),
                }
                return jsonify({
                    "ok": True,
                    "started_ids": sorted(started),
                    "all_started": len(started) == len(self.peers),
                })

        @self.app.get("/startup/started/status")
        def startup_started_status_route():
            run_id = request.args.get("run_id", default="", type=str)
            if not run_id:
                return jsonify({"error": "run_id is required"}), 400
            with self._lock:
                started = dict(self.startup_started.get(run_id, {}))
                return jsonify({
                    "run_id": run_id,
                    "started_ids": sorted(started),
                    "started_count": len(started),
                    "expected_count": len(self.peers),
                    "all_started": len(started) == len(self.peers),
                    "reports": started,
                })

        @self.app.get("/model/status")
        def model_status():
            run_id = request.args.get("run_id", default="", type=str)
            round_num = request.args.get("round", type=int)

            if not run_id or round_num is None:
                return jsonify({"error": "run_id and round are required"}), 400

            key = (run_id, int(round_num))
            with self._lock:
                ready = key in self.pre_models

            return jsonify(
                {
                    "peer_id": self.peer_id,
                    "run_id": run_id,
                    "round": int(round_num),
                    "ready": bool(ready),
                }
            )

        @self.app.get("/model/download")
        def download_model():
            run_id = request.args.get("run_id", default="", type=str)
            round_num = request.args.get("round", type=int)

            if not run_id or round_num is None:
                return jsonify({"error": "run_id and round are required"}), 400

            key = (run_id, int(round_num))

            with self._lock:
                state_dict = self.pre_models.get(key)
                if state_dict is None:
                    return jsonify(
                        {
                            "error": "PRE-aggregation model not ready",
                            "peer_id": self.peer_id,
                            "run_id": run_id,
                            "round": int(round_num),
                        }
                    ), 404

                buffer = io.BytesIO()
                torch.save(state_dict, buffer)
                model_bytes = buffer.getvalue()

            return model_bytes, 200, {
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(model_bytes)),
                "X-DFL-Peer-ID": str(self.peer_id),
                "X-DFL-Run-ID": run_id,
                "X-DFL-Round": str(int(round_num)),
                "X-DFL-Stage": "pre",
            }

        @self.app.get("/update/download")
        def download_update():
            """
            Manager-only/control-plane use:
            download this client's local update vector for topology scoring.
            No aggregation happens here.
            """
            run_id = request.args.get("run_id", default="", type=str)
            round_num = request.args.get("round", type=int)

            if not run_id or round_num is None:
                return jsonify({"error": "run_id and round are required"}), 400

            key = (run_id, int(round_num))

            with self._lock:
                update_obj = self.updates.get(key)
                if update_obj is None:
                    return jsonify({"error": "update not ready"}), 404

                buffer = io.BytesIO()
                torch.save(update_obj, buffer)
                payload = buffer.getvalue()

            return payload, 200, {
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(payload)),
                "X-DFL-Peer-ID": str(self.peer_id),
                "X-DFL-Run-ID": run_id,
                "X-DFL-Round": str(int(round_num)),
            }

        @self.app.get("/report")
        def get_report():
            """
            Manager may collect real timing/metric reports from each client.
            """
            run_id = request.args.get("run_id", default="", type=str)
            round_num = request.args.get("round", type=int)

            if not run_id or round_num is None:
                return jsonify({"error": "run_id and round are required"}), 400

            key = (run_id, int(round_num))
            with self._lock:
                report = self.round_reports.get(key)

            if report is None:
                return jsonify({"error": "report not ready"}), 404

            return jsonify(report)

        @self.app.get("/round/status")
        def round_status():
            run_id = request.args.get("run_id", default="", type=str)
            round_num = request.args.get("round", type=int)

            if not run_id or round_num is None:
                return jsonify({"error": "run_id and round are required"}), 400

            key = (run_id, int(round_num))
            with self._lock:
                return jsonify(
                    {
                        "peer_id": self.peer_id,
                        "run_id": run_id,
                        "round": int(round_num),
                        "pre_model_ready": key in self.pre_models,
                        "update_ready": key in self.updates,
                        "report_ready": key in self.round_reports,
                        "round_complete": int(
                            self.completed_rounds.get(run_id, 0)
                        ) >= int(round_num),
                    }
                )

    def set_runtime_info(self, info: Dict[str, Any]) -> None:
        with self._lock:
            self.runtime_info = dict(info)

    def set_initial_checkpoint(
        self,
        payload: bytes,
        sha256: str,
        seed: int,
        run_id: str,
    ) -> None:
        run_id = str(run_id)
        with self._lock:
            self.active_run_id = run_id
            self.initial_checkpoints[run_id] = {
                "payload": bytes(payload),
                "sha256": str(sha256),
                "seed": int(seed),
            }
        self.logger.info(
            "Initial checkpoint published | run=%s seed=%s sha256=%s bytes=%s",
            run_id, seed, sha256, len(payload),
        )

    def publish_pre_model(
        self,
        state_dict: Dict[str, torch.Tensor],
        round_num: int,
        run_id: str,
    ) -> None:
        """
        Called LOCALLY by client.py immediately after local training.
        This is the model that active neighbors are allowed to download.
        """
        run_id = str(run_id)
        round_num = int(round_num)
        key = (run_id, round_num)

        with self._lock:
            self.active_run_id = run_id
            self.pre_models[key] = _cpu_clone(state_dict)
            self._cleanup_old_rounds_locked(run_id)

        self.logger.info(
            "Published PRE model | peer=%s run=%s round=%s",
            self.peer_id,
            run_id,
            round_num,
        )

    def publish_update(
        self,
        update_obj: Any,
        round_num: int,
        run_id: str,
    ) -> None:
        """
        Store the local parameter update used by the manager for statistical
        topology scoring. This is NOT a global/FedAvg upload.
        """
        run_id = str(run_id)
        round_num = int(round_num)

        with self._lock:
            self.active_run_id = run_id
            self.updates[(run_id, round_num)] = _cpu_clone(update_obj)
            self._cleanup_old_rounds_locked(run_id)

    def set_round_report(
        self,
        report: Dict[str, Any],
        round_num: int,
        run_id: str,
    ) -> None:
        run_id = str(run_id)
        round_num = int(round_num)

        payload = dict(report)
        payload.setdefault("peer_id", self.peer_id)
        payload.setdefault("run_id", run_id)
        payload.setdefault("round", round_num)
        payload.setdefault("reported_utc", _utc_now())

        with self._lock:
            self.active_run_id = run_id
            self.round_reports[(run_id, round_num)] = payload
            self._cleanup_old_rounds_locked(run_id)

    def mark_round_complete(self, round_num: int, run_id: str) -> None:
        run_id = str(run_id)
        round_num = int(round_num)

        with self._lock:
            self.active_run_id = run_id
            self.completed_rounds[run_id] = max(
                round_num,
                int(self.completed_rounds.get(run_id, 0)),
            )

    def reset_run(self, run_id: str) -> None:
        """
        Optional cleanup before re-running the same run_id.
        Normally every baseline gets a unique run_id.
        """
        run_id = str(run_id)
        with self._lock:
            self.pre_models = {
                k: v for k, v in self.pre_models.items() if k[0] != run_id
            }
            self.updates = {
                k: v for k, v in self.updates.items() if k[0] != run_id
            }
            self.round_reports = {
                k: v for k, v in self.round_reports.items() if k[0] != run_id
            }
            self.completed_rounds.pop(run_id, None)

    def run(self, host: str = "0.0.0.0", debug: bool = False) -> None:
        self.logger.info(
            "Starting D-Clique HTTP server | peer=%s host=%s port=%s",
            self.peer_id,
            host,
            self.port,
        )
        self.app.run(
            host=host,
            port=self.port,
            debug=debug,
            use_reloader=False,
            threaded=True,
        )

    def run_threaded(
        self,
        host: str = "0.0.0.0",
        daemon: bool = True,
    ) -> threading.Thread:
        thread = threading.Thread(
            target=self.run,
            args=(host,),
            daemon=daemon,
            name=f"dclique-http-{self.peer_id}",
        )
        thread.start()
        return thread
