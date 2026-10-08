import asyncio
import io
import logging
import time
from typing import Any, Dict, Iterable, Optional, Tuple

import aiohttp
import torch


class DFLClient:
    """
    Async peer-to-peer HTTP client for physical D-Clique DFL.

    Startup:
        wait_for_all_peers_healthy()

    Per round:
        wait_and_fetch_active_models(active_peer_ids, round_num, run_id)

    IMPORTANT:
    - Startup health checks ALL configured devices.
    - Round model waiting checks ONLY the active W-neighbors supplied by client.py.
    - Durations use time.perf_counter(); synchronized device clocks are not needed.
    """

    def __init__(
        self,
        peer_id: int,
        peer_config: Dict[Any, Dict[str, Any]],
        timeout: float = 30.0,
        retry_attempts: int = 2,
        poll_interval: float = 0.10,
    ):
        self.peer_id = int(peer_id)
        self.peer_config = {int(k): dict(v) for k, v in peer_config.items()}

        if self.peer_id not in self.peer_config:
            raise KeyError(f"Peer {self.peer_id} missing from peer configuration.")

        self.timeout_seconds = float(timeout)
        self.retry_attempts = max(1, int(retry_attempts))
        self.poll_interval = max(0.02, float(poll_interval))
        self.logger = logging.getLogger(f"HTTPClient-{self.peer_id}")

    def _url(self, peer_id: int, endpoint: str) -> str:
        peer_id = int(peer_id)
        cfg = self.peer_config[peer_id]
        return f"http://{cfg['ip']}:{int(cfg['port'])}{endpoint}"

    def _validate_peer_ids(self, peer_ids: Iterable[int]) -> list[int]:
        result = []
        seen = set()

        for value in peer_ids:
            peer_id = int(value)

            if peer_id == self.peer_id:
                # Own model is already local; never download it over HTTP.
                continue

            if peer_id not in self.peer_config:
                raise KeyError(f"Unknown peer id {peer_id}")

            if peer_id not in seen:
                seen.add(peer_id)
                result.append(peer_id)

        return result

    async def get_health(
        self,
        target_peer_id: int,
        run_id: Optional[str] = None,
        request_timeout: float = 2.0,
    ) -> Optional[Dict[str, Any]]:
        params = {}
        if run_id:
            params["run_id"] = str(run_id)

        timeout = aiohttp.ClientTimeout(total=float(request_timeout))

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    self._url(target_peer_id, "/health"),
                    params=params,
                ) as response:
                    if response.status != 200:
                        return None

                    data = await response.json()
                    if data.get("status") != "healthy":
                        return None

                    return data

        except Exception:
            return None

    async def check_all_peers_health(
        self,
        include_self: bool = True,
        request_timeout: float = 2.0,
    ) -> Dict[int, Optional[Dict[str, Any]]]:
        peer_ids = sorted(self.peer_config.keys())

        if not include_self:
            peer_ids = [pid for pid in peer_ids if pid != self.peer_id]

        results = await asyncio.gather(
            *[
                self.get_health(
                    pid,
                    request_timeout=request_timeout,
                )
                for pid in peer_ids
            ],
            return_exceptions=True,
        )

        health: Dict[int, Optional[Dict[str, Any]]] = {}

        for pid, result in zip(peer_ids, results):
            if isinstance(result, Exception):
                health[pid] = None
            else:
                health[pid] = result

        return health

    async def wait_for_all_peers_healthy(
        self,
        timeout: float = 180.0,
        request_timeout: float = 2.0,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Startup-only barrier. Training does not begin until all configured
        Raspberry Pi / Jetson peer servers are reachable.
        """
        start = time.perf_counter()
        last_missing = None

        while True:
            health = await self.check_all_peers_health(
                include_self=True,
                request_timeout=request_timeout,
            )

            missing = sorted(
                pid for pid, status in health.items() if status is None
            )

            if not missing:
                elapsed = time.perf_counter() - start
                self.logger.info(
                    "All %d peers healthy after %.3fs",
                    len(health),
                    elapsed,
                )
                return {
                    pid: status
                    for pid, status in health.items()
                    if status is not None
                }

            if missing != last_missing:
                self.logger.info("Waiting for peer health: missing=%s", missing)
                last_missing = missing

            if time.perf_counter() - start >= float(timeout):
                raise TimeoutError(
                    "Startup health barrier timed out. "
                    f"Unreachable peers: {missing}"
                )

            await asyncio.sleep(0.5)

    async def is_pre_model_ready(
        self,
        target_peer_id: int,
        round_num: int,
        run_id: str,
    ) -> bool:
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    self._url(target_peer_id, "/model/status"),
                    params={
                        "run_id": str(run_id),
                        "round": int(round_num),
                    },
                ) as response:
                    if response.status != 200:
                        return False
                    data = await response.json()
                    return bool(data.get("ready", False))
        except Exception:
            return False

    async def fetch_pre_model_once(
        self,
        target_peer_id: int,
        round_num: int,
        run_id: str,
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], int, float]:
        """
        Returns:
            (state_dict_or_none, bytes_received, download_seconds)
        """
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        url = self._url(target_peer_id, "/model/download")

        for attempt in range(1, self.retry_attempts + 1):
            transfer_start = time.perf_counter()

            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(
                        url,
                        params={
                            "run_id": str(run_id),
                            "round": int(round_num),
                        },
                    ) as response:
                        if response.status == 404:
                            return None, 0, 0.0

                        if response.status != 200:
                            if attempt < self.retry_attempts:
                                await asyncio.sleep(self.poll_interval)
                                continue
                            return None, 0, 0.0

                        payload = await response.read()
                        download_seconds = time.perf_counter() - transfer_start

                        state_dict = torch.load(
                            io.BytesIO(payload),
                            map_location="cpu",
                        )

                        if not isinstance(state_dict, dict):
                            raise TypeError(
                                f"Peer {target_peer_id} returned a non-state_dict model."
                            )

                        return state_dict, len(payload), download_seconds

            except Exception as exc:
                if attempt >= self.retry_attempts:
                    self.logger.warning(
                        "Model fetch failed | peer=%s run=%s round=%s error=%s",
                        target_peer_id,
                        run_id,
                        round_num,
                        exc,
                    )
                    return None, 0, 0.0

                await asyncio.sleep(self.poll_interval)

        return None, 0, 0.0

    async def _wait_for_one_active_peer(
        self,
        target_peer_id: int,
        round_num: int,
        run_id: str,
        deadline_perf: float,
    ) -> Tuple[int, Dict[str, torch.Tensor], Dict[str, Any]]:
        peer_wait_start = time.perf_counter()
        status_polls = 0

        while True:
            now = time.perf_counter()
            if now >= deadline_perf:
                raise TimeoutError(
                    f"Timed out waiting for active peer {target_peer_id} "
                    f"(run={run_id}, round={round_num})"
                )

            ready = await self.is_pre_model_ready(
                target_peer_id=target_peer_id,
                round_num=round_num,
                run_id=run_id,
            )
            status_polls += 1

            if not ready:
                await asyncio.sleep(self.poll_interval)
                continue

            ready_detected_perf = time.perf_counter()

            state_dict, bytes_received, download_seconds = (
                await self.fetch_pre_model_once(
                    target_peer_id=target_peer_id,
                    round_num=round_num,
                    run_id=run_id,
                )
            )

            # Race-safe: if status was ready but download briefly failed,
            # continue polling until deadline.
            if state_dict is None:
                await asyncio.sleep(self.poll_interval)
                continue

            finished_perf = time.perf_counter()

            metrics = {
                "peer_id": int(target_peer_id),
                "ready_wait_seconds": (
                    ready_detected_perf - peer_wait_start
                ),
                "download_seconds": float(download_seconds),
                "total_peer_wait_seconds": (
                    finished_perf - peer_wait_start
                ),
                "bytes_received": int(bytes_received),
                "status_polls": int(status_polls),
            }

            return int(target_peer_id), state_dict, metrics

    async def wait_and_fetch_active_models(
        self,
        active_peer_ids: Iterable[int],
        round_num: int,
        run_id: str,
        timeout: float = 180.0,
    ) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[str, Any]]:
        """
        Wait for and download ONLY this client's active W-neighbors.

        Call this immediately after local training and local PRE-model publication.

        waiting_time_seconds is therefore measured as:
            own PRE model ready -> last required active peer model downloaded

        Returns:
            peer_models:
                {physical_peer_id: state_dict}

            timing:
                aggregate waiting and per-peer transfer/readiness details.
        """
        active_peers = self._validate_peer_ids(active_peer_ids)

        overall_start = time.perf_counter()

        if not active_peers:
            return {}, {
                "active_peer_ids": [],
                "num_active_peers": 0,
                "waiting_time_seconds": 0.0,
                "total_bytes_received": 0,
                "per_peer": {},
            }

        deadline = overall_start + float(timeout)

        tasks = [
            self._wait_for_one_active_peer(
                target_peer_id=pid,
                round_num=int(round_num),
                run_id=str(run_id),
                deadline_perf=deadline,
            )
            for pid in active_peers
        ]

        results = await asyncio.gather(*tasks)
        overall_end = time.perf_counter()

        peer_models: Dict[int, Dict[str, torch.Tensor]] = {}
        per_peer: Dict[int, Dict[str, Any]] = {}

        for pid, state_dict, metrics in results:
            peer_models[int(pid)] = state_dict
            per_peer[int(pid)] = metrics

        total_bytes = sum(
            int(meta["bytes_received"]) for meta in per_peer.values()
        )

        timing = {
            "active_peer_ids": active_peers,
            "num_active_peers": len(active_peers),
            "waiting_time_seconds": overall_end - overall_start,
            "total_bytes_received": total_bytes,
            "per_peer": per_peer,
        }

        self.logger.info(
            "Active-neighbor models ready | peer=%s run=%s round=%s "
            "neighbors=%s wait=%.4fs bytes=%s",
            self.peer_id,
            run_id,
            round_num,
            active_peers,
            timing["waiting_time_seconds"],
            total_bytes,
        )

        return peer_models, timing

    async def download_initial_checkpoint(
        self,
        coordinator_peer_id: int,
        run_id: str,
        timeout: float = 180.0,
    ) -> Tuple[bytes, str, int]:
        """Download the exact serialized initialization created once by coordinator."""
        deadline = time.perf_counter() + float(timeout)
        url = self._url(int(coordinator_peer_id), "/startup/initial_checkpoint")

        while time.perf_counter() < deadline:
            request_timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            try:
                async with aiohttp.ClientSession(timeout=request_timeout) as session:
                    async with session.get(url, params={"run_id": str(run_id)}) as response:
                        if response.status == 404:
                            await asyncio.sleep(self.poll_interval)
                            continue
                        if response.status != 200:
                            text = await response.text()
                            raise RuntimeError(
                                f"Initialization download failed: HTTP {response.status}: {text}"
                            )
                        payload = await response.read()
                        sha256 = str(response.headers.get("X-DFL-Init-SHA256", ""))
                        seed = int(response.headers.get("X-DFL-Seed", "0"))
                        if not sha256:
                            raise RuntimeError("Coordinator did not provide initialization SHA256")
                        return payload, sha256, seed
            except RuntimeError:
                raise
            except Exception:
                await asyncio.sleep(self.poll_interval)

        raise TimeoutError(
            f"Timed out downloading initial checkpoint from Client {coordinator_peer_id}"
        )

    async def register_startup_ready(
        self,
        coordinator_peer_id: int,
        run_id: str,
        init_sha256: str,
        device_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        url = self._url(int(coordinator_peer_id), "/startup/ready")
        payload = {
            "run_id": str(run_id),
            "client_id": int(self.peer_id),
            "init_sha256": str(init_sha256),
            "device_info": dict(device_info),
        }
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                data = await response.json(content_type=None)
                if response.status != 200:
                    raise RuntimeError(
                        f"Startup ready rejected by coordinator: HTTP {response.status}: {data}"
                    )
                return data

    async def wait_for_synchronized_start(
        self,
        coordinator_peer_id: int,
        run_id: str,
        timeout: float = 180.0,
    ) -> Dict[str, Any]:
        """
        Wait until all clients are ready, then honor one coordinator release time.

        The coordinator returns a remaining duration on its own monotonic clock.
        We subtract half the measured HTTP RTT as a simple one-way-delay estimate,
        avoiding dependence on perfectly synchronized wall clocks for the barrier.
        """
        deadline = time.perf_counter() + float(timeout)
        url = self._url(int(coordinator_peer_id), "/startup/status")
        last_ready_ids = None

        while time.perf_counter() < deadline:
            req_start = time.perf_counter()
            try:
                request_timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
                async with aiohttp.ClientSession(timeout=request_timeout) as session:
                    async with session.get(url, params={"run_id": str(run_id)}) as response:
                        data = await response.json(content_type=None)
                        req_end = time.perf_counter()
                        if response.status != 200:
                            await asyncio.sleep(self.poll_interval)
                            continue

                        ready_ids = tuple(data.get("ready_ids", []))
                        if ready_ids != last_ready_ids:
                            self.logger.info(
                                "Startup barrier | ready=%s/%s ids=%s",
                                data.get("ready_count", 0),
                                data.get("expected_count", len(self.peer_config)),
                                list(ready_ids),
                            )
                            last_ready_ids = ready_ids

                        if data.get("start_scheduled"):
                            rtt = req_end - req_start
                            remaining = float(data.get("start_in_seconds", 0.0))
                            corrected_wait = max(0.0, remaining - 0.5 * rtt)
                            release_received_utc = time.time()
                            self.logger.info(
                                "Synchronized start scheduled | remaining=%.6fs rtt=%.6fs corrected_wait=%.6fs",
                                remaining, rtt, corrected_wait,
                            )
                            await asyncio.sleep(corrected_wait)
                            actual_start_perf = time.perf_counter()
                            actual_start_unix = time.time()
                            return {
                                "coordinator_reported_remaining_seconds": remaining,
                                "status_request_rtt_seconds": rtt,
                                "corrected_wait_seconds": corrected_wait,
                                "release_received_unix": release_received_utc,
                                "actual_start_perf": actual_start_perf,
                                "actual_start_unix": actual_start_unix,
                            }
            except Exception as exc:
                self.logger.debug("Startup status poll failed: %s", exc)

            await asyncio.sleep(self.poll_interval)

        raise TimeoutError("Timed out waiting for synchronized startup release")

    async def report_synchronized_start(
        self,
        coordinator_peer_id: int,
        run_id: str,
        start_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        url = self._url(int(coordinator_peer_id), "/startup/started")
        payload = {
            "run_id": str(run_id),
            "client_id": int(self.peer_id),
            **dict(start_meta),
        }
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                data = await response.json(content_type=None)
                if response.status != 200:
                    raise RuntimeError(
                        f"Could not report synchronized start: HTTP {response.status}: {data}"
                    )
                return data

    async def fetch_started_status(
        self,
        coordinator_peer_id: int,
        run_id: str,
    ) -> Optional[Dict[str, Any]]:
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        url = self._url(int(coordinator_peer_id), "/startup/started/status")
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params={"run_id": str(run_id)}) as response:
                    if response.status != 200:
                        return None
                    return await response.json()
        except Exception:
            return None

    async def fetch_update(
        self,
        target_peer_id: int,
        round_num: int,
        run_id: str,
    ) -> Optional[Any]:
        """
        Control-plane helper intended for the manager/topology learner.
        """
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    self._url(target_peer_id, "/update/download"),
                    params={
                        "run_id": str(run_id),
                        "round": int(round_num),
                    },
                ) as response:
                    if response.status != 200:
                        return None

                    payload = await response.read()
                    return torch.load(
                        io.BytesIO(payload),
                        map_location="cpu",
                    )
        except Exception:
            return None

    async def fetch_round_report(
        self,
        target_peer_id: int,
        round_num: int,
        run_id: str,
    ) -> Optional[Dict[str, Any]]:
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    self._url(target_peer_id, "/report"),
                    params={
                        "run_id": str(run_id),
                        "round": int(round_num),
                    },
                ) as response:
                    if response.status != 200:
                        return None
                    return await response.json()
        except Exception:
            return None
