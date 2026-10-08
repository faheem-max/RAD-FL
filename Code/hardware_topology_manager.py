#!/usr/bin/env python3
"""PC-side topology control plane for hardware_training_client.py.

The manager:
  * never receives/aggregates PRE models for FedAvg,
  * publishes W(t),
  * reads local update vectors and timing reports,
  * runs the same DynamicTopologyManager used in the PC simulation.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import time
from pathlib import Path
from typing import Any, Dict

import aiohttp
import numpy as np
import torch
import yaml

from topology import DynamicTopologyManager


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def peer_url(peers, pid, endpoint):
    cfg = peers[int(pid)]
    return f"http://{cfg['ip']}:{int(cfg['port'])}{endpoint}"


def torch_bytes(obj):
    bio = io.BytesIO()
    torch.save(obj, bio)
    return bio.getvalue()


async def wait_all_healthy(peers, timeout_seconds):
    deadline = time.perf_counter() + float(timeout_seconds)
    while time.perf_counter() < deadline:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as s:
            async def one(pid):
                try:
                    async with s.get(peer_url(peers, pid, '/health')) as r:
                        return pid, (await r.json()) if r.status == 200 else None
                except Exception:
                    return pid, None
            rows = await asyncio.gather(*(one(pid) for pid in sorted(peers)))
        found = {pid: data for pid, data in rows if data is not None}
        if len(found) == len(peers):
            return found
        print('Waiting for clients:', sorted(set(peers) - set(found)))
        await asyncio.sleep(0.5)
    raise TimeoutError('Timed out waiting for all device HTTP servers')


async def post_topology_to_all(peers, run_id, round_num, W, timeout_seconds=30):
    # Preserve float64 over the wire. Metropolis-Hastings weights such as 1/6
    # are not exactly representable in float32; down-casting before validation can
    # make a mathematically stochastic row appear to sum to 1.00000003.
    raw = torch_bytes(torch.as_tensor(W, dtype=torch.float64, device='cpu'))
    timeout = aiohttp.ClientTimeout(total=float(timeout_seconds))
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async def one(pid):
            async with s.post(
                peer_url(peers, pid, '/control/topology'),
                params={'run_id': str(run_id), 'round': int(round_num)},
                data=raw,
                headers={'Content-Type': 'application/octet-stream'},
            ) as r:
                text = await r.text()
                if r.status != 200:
                    raise RuntimeError(f'W post failed to client {pid}: HTTP {r.status}: {text}')
                return pid
        await asyncio.gather(*(one(pid) for pid in sorted(peers)))


async def post_finalize_to_all(
    peers, run_id, final_round, timeout_seconds=60, retry_interval=0.5
):
    """Acknowledge that the manager has safely collected the final round."""
    pending = set(int(pid) for pid in peers)
    deadline = time.perf_counter() + float(timeout_seconds)
    payload = {'run_id': str(run_id), 'final_round': int(final_round)}

    while pending and time.perf_counter() < deadline:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async def one(pid):
                try:
                    async with s.post(
                        peer_url(peers, pid, '/control/finalize'),
                        json=payload,
                    ) as r:
                        return pid, (r.status == 200)
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                    return pid, False

            rows = await asyncio.gather(*(one(pid) for pid in sorted(pending)))

        for pid, ok in rows:
            if ok:
                pending.discard(pid)

        if pending:
            await asyncio.sleep(float(retry_interval))

    if pending:
        raise TimeoutError(
            f'Could not deliver finalization ACK to clients {sorted(pending)}'
        )


async def fetch_update_report(peers, pid, run_id, round_num):
    """Fetch one client's update/report; transient network errors mean 'not ready yet'."""
    timeout = aiohttp.ClientTimeout(total=30)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(
                peer_url(peers, pid, '/update/download'),
                params={'run_id': run_id, 'round': int(round_num)},
            ) as r:
                if r.status != 200:
                    return None
                raw = await r.read()
                try:
                    update = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
                except TypeError:
                    update = torch.load(io.BytesIO(raw), map_location='cpu')
            async with s.get(
                peer_url(peers, pid, '/report'),
                params={'run_id': run_id, 'round': int(round_num)},
            ) as r:
                if r.status != 200:
                    return None
                report = await r.json()
        return update, report
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
        # During distributed execution a peer may be briefly busy/restarting its
        # connection. The outer polling loop retries until round_timeout.
        return None


async def wait_round_payloads(peers, run_id, round_num, timeout_seconds):
    deadline = time.perf_counter() + float(timeout_seconds)
    while time.perf_counter() < deadline:
        rows = await asyncio.gather(*(
            fetch_update_report(peers, pid, run_id, round_num) for pid in sorted(peers)
        ))
        if all(row is not None for row in rows):
            updates = {}
            reports = {}
            for pid, row in zip(sorted(peers), rows):
                updates[pid], reports[pid] = row
            return updates, reports
        await asyncio.sleep(0.5)
    missing = [pid for pid, row in zip(sorted(peers), rows) if row is None]
    raise TimeoutError(f'Round {round_num}: missing update/report from clients {missing}')


def build_hardware_profiles(ordered_peer_ids, reports, update_round):
    profiles: Dict[int, Dict[str, Any]] = {}
    for i, pid in enumerate(ordered_peer_ids):
        profiles[i] = {
            'device_type': reports[pid].get('hardware_type', 'unknown'),
            'compute_time': float(reports[pid].get('local_training_seconds', 0.0)),
            'latency_matrix': {},
        }

    for i, pid_i in enumerate(ordered_peer_ids):
        rtts = reports[pid_i].get('peer_rtt_seconds', {}) or {}
        for j, pid_j in enumerate(ordered_peer_ids):
            if i == j:
                profiles[i]['latency_matrix'][j] = 0.0
                continue
            value = rtts.get(str(pid_j))
            if value is None:
                # On non-update rounds topology.py does not consume latency. On
                # an update round this conservative fallback makes an unmeasured
                # link unattractive rather than pretending it is fast.
                value = 1.0 if update_round else 0.05
            profiles[i]['latency_matrix'][j] = float(value)
    return profiles


def topology_edges(W, ids):
    W = np.asarray(W)
    return [
        [int(ids[i]), int(ids[j])]
        for i in range(W.shape[0])
        for j in range(i + 1, W.shape[1])
        if W[i, j] > 0
    ]


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


async def main_async(peer_config_path, experiment_config_path, results_root):
    peer_doc = load_yaml(peer_config_path)
    exp_doc = load_yaml(experiment_config_path)
    peers = {int(k): dict(v) for k, v in peer_doc['peers'].items()}
    ids = sorted(peers)
    if ids != list(range(1, 9)):
        raise ValueError(f'Expected physical IDs 1..8, got {ids}')
    cliques = [int(peers[pid]['clique_id']) for pid in ids]
    if cliques != [1,1,1,1,2,2,2,2]:
        raise ValueError(f'Expected cliques [1,1,1,1,2,2,2,2], got {cliques}')

    e = exp_doc.get('experiment', {})
    t = exp_doc.get('training', {})
    q = exp_doc.get('topology', {})
    a = exp_doc.get('aggregation', {})
    startup = exp_doc.get('startup', {})
    run_id = str(e.get('run_id', 'proposed_seed42'))
    num_rounds = int(t.get('num_rounds', 2))
    round_timeout = float(t.get('round_timeout_seconds', 900))

    out = Path(results_root).expanduser().resolve() / run_id
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'experiment_config_used.json', exp_doc)
    save_json(out / 'peer_config_used.json', {str(k): v for k,v in peers.items()})

    print(f'Hardware topology manager | run={run_id}')
    health = await wait_all_healthy(peers, startup.get('startup_health_timeout_seconds', 240))
    for pid in ids:
        runtime = health[pid].get('runtime', {}) or {}
        print(f"  Client {pid}: {health[pid].get('hardware_type')} | device={runtime.get('selected_device')} | gpu={runtime.get('gpu_name')}")

    aggregation_mode = str(a.get('mode', 'metropolis')).strip().lower()
    if aggregation_mode not in {'metropolis', 'sample_aware', 'hybrid'}:
        raise ValueError(
            "aggregation.mode must be metropolis, sample_aware, or hybrid"
        )
    aggregation_alpha = float(a.get('alpha', 0.5))
    if aggregation_alpha < 0.0:
        raise ValueError('aggregation.alpha must be >= 0')

    # Sample counts are required by sample-aware and hybrid aggregation. They
    # remain fixed; only their normalization changes when the active topology changes.
    sample_counts = None
    if aggregation_mode in {'sample_aware', 'hybrid'}:
        missing_counts = [pid for pid in ids if 'local_train_samples' not in peers[pid]]
        if missing_counts:
            raise ValueError(
                f"Missing local_train_samples in peer_config for clients {missing_counts}"
            )
        sample_counts = [int(peers[pid]['local_train_samples']) for pid in ids]
        if any(n <= 0 for n in sample_counts):
            raise ValueError(f"All local_train_samples must be positive, got {sample_counts}")

    print(f'Aggregation mode: {aggregation_mode.upper()}')
    if sample_counts is not None:
        print('Local train samples:', dict(zip(ids, sample_counts)))
    if aggregation_mode == 'hybrid':
        print(f'Hybrid alpha: {aggregation_alpha}')

    mgr = DynamicTopologyManager(
        num_clients=8,
        num_cliques=int(q.get('num_cliques', 2)),
        k_inter=int(q.get('k_inter', 2)),
        mu=float(q.get('mu', 0.60)),
        beta_g=float(q.get('beta_g', 0.20)),
        straggler_quantile=float(q.get('straggler_quantile', 0.85)),
        delta_max=float(q.get('delta_max', 0.15)),
        update_interval=int(q.get('update_interval', 5)),
        sample_counts=sample_counts,
        aggregation_mode=aggregation_mode,
        aggregation_alpha=aggregation_alpha,
    )

    aggregation_meta = {
        'aggregation_mode': aggregation_mode,
        'aggregation_alpha': aggregation_alpha if aggregation_mode == 'hybrid' else None,
        'sample_counts': dict(zip(ids, sample_counts)) if sample_counts is not None else None,
    }

    current_W = np.asarray(mgr.current_W, dtype=np.float64)
    await post_topology_to_all(peers, run_id, 1, current_W)
    save_json(out / 'topology_round_001.json', {
        'round': 1,
        **aggregation_meta,
        'edges': topology_edges(current_W, ids),
    })
    torch.save(torch.tensor(current_W, dtype=torch.float64), out / 'W_round_001.pt')
    print('Round 1 topology posted:', topology_edges(current_W, ids))

    for round_num in range(1, num_rounds + 1):
        physical_updates, reports = await wait_round_payloads(
            peers, run_id, round_num, round_timeout
        )
        client_updates = {i: physical_updates[pid] for i, pid in enumerate(ids)}
        is_update_round = (round_num % int(q.get('update_interval', 5)) == 0)
        hardware_profiles = build_hardware_profiles(ids, reports, is_update_round)

        save_json(out / f'round_{round_num:03d}_reports.json', {
            'round': round_num,
            'reports': {str(pid): reports[pid] for pid in ids},
        })
        global_val_values = [
            float(reports[p]['global_val_accuracy'])
            for p in ids if reports[p].get('global_val_accuracy') is not None
        ]
        global_test_values = [
            float(reports[p]['global_test_accuracy'])
            for p in ids if reports[p].get('global_test_accuracy') is not None
        ]
        eval_text = (
            f" | mean_global_val={np.mean(global_val_values) * 100.0:.2f}%"
            if global_val_values else ""
        )
        if global_test_values:
            eval_text += f" | mean_global_test={np.mean(global_test_values) * 100.0:.2f}%"
        print(
            f"Round {round_num:02d} complete | "
            f"max_train={max(float(reports[p]['local_training_seconds']) for p in ids):.4f}s | "
            f"mean_wait={np.mean([float(reports[p]['waiting_time_seconds']) for p in ids]):.4f}s"
            f"{eval_text}"
        )

        if round_num >= num_rounds:
            # All final update vectors and reports are now safely held by the
            # manager. Release every client so its daemon HTTP server may stop.
            await post_finalize_to_all(
                peers, run_id, round_num,
                timeout_seconds=min(120.0, max(30.0, round_timeout)),
            )
            print(f'Finalization ACK delivered to all clients for Round {round_num:02d}')
            break

        next_W = mgr.update_topology(
            round_num=round_num,
            client_updates=client_updates,
            hardware_profiles=hardware_profiles,
        )
        current_W = np.asarray(next_W, dtype=np.float64)
        next_round = round_num + 1
        await post_topology_to_all(peers, run_id, next_round, current_W)
        torch.save(torch.tensor(current_W, dtype=torch.float64), out / f'W_round_{next_round:03d}.pt')
        save_json(out / f'topology_round_{next_round:03d}.json', {
            'round': next_round,
            **aggregation_meta,
            'edges': topology_edges(current_W, ids),
            'diagnostics': mgr.get_diagnostics(),
        })
        if mgr.get_diagnostics().get('topology_updated'):
            print(f'  Topology updated for Round {next_round}: {topology_edges(current_W, ids)}')

    print('Manager complete. Results:', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--peer_config', default='config/peer_config.yaml')
    ap.add_argument('--experiment_config', default='config/experiment_config.yaml')
    ap.add_argument('--results_root', default='results')
    args = ap.parse_args()
    asyncio.run(main_async(args.peer_config, args.experiment_config, args.results_root))


if __name__ == '__main__':
    main()
