from typing import Dict, List
import streamlit as st
import simpy
import random
import numpy as np
import math
import pandas as pd
from io import BytesIO  # for downloads
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# =============================
# Utilities
# =============================
def exp_time(mean):
    if mean <= 0:
        return 0.0
    return random.expovariate(1.0/mean)

def normal_time(mean, cv=0.4):
    if mean <= 0:
        return 0.0
    sd = max(1e-6, mean * cv)
    return max(0.0, random.gauss(mean, sd))

def speed_multiplier_from_cv(cv):
    if cv <= 0:
        return 1.0
    sigma = math.sqrt(math.log(1 + cv**2))
    mu = -0.5 * sigma**2
    return np.random.lognormal(mean=mu, sigma=sigma)

def draw_service_time(role_for_dist, mean, dist_map, cv_task):
    dist = dist_map.get(role_for_dist, "exponential")
    base = normal_time(mean, cv=0.4) if dist == "normal" else exp_time(mean)
    return base * speed_multiplier_from_cv(cv_task)

def pct(x):
    try:
        return f"{100*float(x):.1f}%"
    except Exception:
        return "0.0%"

# =============================
# Work schedule helpers
# =============================
MIN_PER_HOUR = 60
DAY_MIN = 24 * MIN_PER_HOUR

def is_open(t_min, open_minutes):
    return (t_min % DAY_MIN) < open_minutes

def minutes_until_close(t_min, open_minutes):
    t_mod = t_min % DAY_MIN
    return max(0.0, open_minutes - t_mod)

def minutes_until_open(t_min, open_minutes):
    t_mod = t_min % DAY_MIN
    if t_mod < open_minutes:
        return 0.0
    return DAY_MIN - t_mod

def effective_open_minutes(sim_minutes, open_minutes):
    full_days = int(sim_minutes // DAY_MIN)
    remainder = sim_minutes % DAY_MIN
    return full_days * open_minutes + min(open_minutes, remainder)

# =============================
# Roles / constants
# =============================
ROLES = ["Front Desk", "Nurse", "Provider", "Back Office"]
DONE = "Done"

# =============================
# Metrics
# =============================
class Metrics:
    def __init__(self):
        self.time_stamps = []
        self.queues = {r: [] for r in ROLES}
        self.waits = {r: [] for r in ROLES}
        self.taps = {r: 0 for r in ROLES}
        self.completed = 0
        self.arrivals_total = 0
        self.arrivals_by_role = {r: 0 for r in ROLES}
        self.service_time_sum = {r: 0.0 for r in ROLES}
        # loop counters
        self.loop_fd_insufficient = 0
        self.loop_nurse_insufficient = 0
        self.loop_provider_insufficient = 0
        self.loop_backoffice_insufficient = 0
        # event log: (time_min, task_id, step_code, note, arrival_time_min)
        self.events = []
        self.task_arrival_time: Dict[str, float] = {}
        self.task_completion_time: Dict[str, float] = {}

    def log(self, t, name, step, note="", arrival_t=None):
        self.events.append((t, name, step, note, arrival_t if arrival_t is not None else self.task_arrival_time.get(name)))

# =============================
# Step labels (for logs; no charts)
# =============================
STEP_LABELS = {
    "ARRIVE": "Task arrived",
    "FD_QUEUE": "Front Desk: queued",
    "FD_DONE": "Front Desk: completed",
    "FD_INSUFF": "Front Desk: missing info",
    "FD_RETRY_QUEUE": "Front Desk: re-queued (info)",
    "FD_RETRY_DONE": "Front Desk: re-done (info)",
    "NU_QUEUE": "Nurse: queued",
    "NU_DONE": "Nurse: completed",
    "NU_INSUFF": "Nurse: missing info",
    "NU_RECHECK_QUEUE": "Nurse: re-check queued",
    "NU_RECHECK_DONE": "Nurse: re-check completed",
    "PR_QUEUE": "Provider: queued",
    "PR_DONE": "Provider: completed",
    "PR_INSUFF": "Provider: rework needed",
    "PR_RECHECK_QUEUE": "Provider: recheck queued",
    "PR_RECHECK_DONE": "Provider: recheck done",
    "BO_QUEUE": "Back Office: queued",
    "BO_DONE": "Back Office: completed",
    "BO_INSUFF": "Back Office: rework needed",
    "BO_RECHECK_QUEUE": "Back Office: recheck queued",
    "BO_RECHECK_DONE": "Back Office: recheck done",
    "DONE": "Task resolved"
}
def pretty_step(code):
    return STEP_LABELS.get(code, code)

# =============================
# System
# =============================
# =============================
# Availability schedule generator
# =============================
def generate_availability_schedule(sim_minutes: int, role: str, minutes_per_hour: int, seed_offset: int = 0) -> set:
    """
    Generate a set of minute timestamps when a role is available.
    For each hour, randomly select 'minutes_per_hour' minutes to be available.
    Returns a set of available minute timestamps.
    """
    if minutes_per_hour >= 60:
        # If available all the time, return all minutes
        return set(range(int(sim_minutes)))
    
    if minutes_per_hour <= 0:
        # If never available, return empty set
        return set()
    
    # Use deterministic random based on role name for reproducibility
    local_random = random.Random(hash(role) + seed_offset)
    
    available_minutes = set()
    total_hours = int(np.ceil(sim_minutes / 60.0))
    
    for hour in range(total_hours):
        hour_start = hour * 60
        hour_end = min((hour + 1) * 60, sim_minutes)
        hour_length = hour_end - hour_start
        
        # Number of minutes available in this hour
        available_in_hour = min(minutes_per_hour, hour_length)
        
        # Randomly select which minutes in this hour are available
        all_minutes_in_hour = list(range(hour_start, hour_end))
        selected = local_random.sample(all_minutes_in_hour, available_in_hour)
        available_minutes.update(selected)
    
    return available_minutes

class CHCSystem:
    def __init__(self, env, params, metrics, seed_offset=0):
        self.env = env
        self.p = params
        self.m = metrics

        # Capacities
        self.fd_cap = params["frontdesk_cap"]
        self.nu_cap = params["nurse_cap"]
        self.pr_cap = params["provider_cap"]
        self.bo_cap = params["backoffice_cap"]

        # Resources (None if capacity==0 → skip stage gracefully)
        self.frontdesk = simpy.Resource(env, capacity=self.fd_cap) if self.fd_cap > 0 else None
        self.nurse     = simpy.Resource(env, capacity=self.nu_cap) if self.nu_cap > 0 else None
        self.provider  = simpy.Resource(env, capacity=self.pr_cap) if self.pr_cap > 0 else None
        self.backoffice= simpy.Resource(env, capacity=self.bo_cap) if self.bo_cap > 0 else None

        self.role_to_res = {
            "Front Desk": self.frontdesk,
            "Nurse": self.nurse,
            "Provider": self.provider,
            "Back Office": self.backoffice
        }
        
        # Generate availability schedules for each role
        avail_params = params.get("availability_per_hour", {
            "Front Desk": 60, "Nurse": 60, "Provider": 60, "Back Office": 60
        })
        self.availability = {
            "Front Desk": generate_availability_schedule(
                params["sim_minutes"], "Front Desk", avail_params.get("Front Desk", 60), seed_offset
            ),
            "Nurse": generate_availability_schedule(
                params["sim_minutes"], "Nurse", avail_params.get("Nurse", 60), seed_offset
            ),
            "Provider": generate_availability_schedule(
                params["sim_minutes"], "Provider", avail_params.get("Provider", 60), seed_offset
            ),
            "Back Office": generate_availability_schedule(
                params["sim_minutes"], "Back Office", avail_params.get("Back Office", 60), seed_offset
            ),
        }

    def scheduled_service(self, resource, role_account, mean_time, role_for_dist=None):
        """
        Respect clinic hours AND role availability; if resource is None (capacity 0), service is skipped.
        """
        if resource is None or mean_time <= 1e-12:
            return
        if role_for_dist is None:
            role_for_dist = role_account

        remaining = draw_service_time(role_for_dist, mean_time, self.p["dist_role"], self.p["cv_speed"])
        remaining += max(0.0, self.p["emr_overhead"].get(role_account, 0.0))

        # micro-optimization: local reference
        open_minutes = self.p["open_minutes"]
        available_set = self.availability.get(role_account, set())

        while remaining > 1e-9:
            current_min = int(self.env.now)
            
            # Check if clinic is open
            if not is_open(self.env.now, open_minutes):
                yield self.env.timeout(minutes_until_open(self.env.now, open_minutes))
                continue
            
            # Check if role is available at this minute
            if len(available_set) > 0 and current_min not in available_set:
                # Role not available, wait 1 minute and check again
                yield self.env.timeout(1)
                continue
            
            # Role is available and clinic is open - do work
            window = minutes_until_close(self.env.now, open_minutes)
            
            # Find how long the role stays available
            if len(available_set) > 0:
                avail_window = 1
                check_min = current_min + 1
                while check_min in available_set and avail_window < window and avail_window < remaining:
                    avail_window += 1
                    check_min += 1
                work_chunk = min(remaining, window, avail_window)
            else:
                work_chunk = min(remaining, window)
            
            with resource.request() as req:
                t_req = self.env.now
                yield req
                self.m.waits[role_account].append(self.env.now - t_req)
                self.m.taps[role_account] += 1
                yield self.env.timeout(work_chunk)
                self.m.service_time_sum[role_account] += work_chunk
            remaining -= work_chunk

# =============================
# Routing helpers
# =============================
def sample_next_role(route_row: Dict[str, float]) -> str:
    """Normalize a row and sample next step from ROLES + DONE."""
    keys = tuple(route_row.keys())
    vals = np.fromiter((max(0.0, float(route_row[k])) for k in keys), dtype=float)
    s = vals.sum()
    if s <= 0:
        return DONE
    probs = vals / s
    return random.choices(keys, weights=probs, k=1)[0]

# =============================
# Workflows per role
# =============================
def handle_role(env, task_id, s: CHCSystem, role: str):
    """Run one role (with loops), then return the next role or DONE."""
    if role not in ROLES:
        return DONE

    res = s.role_to_res[role]

    if role == "Front Desk":
        if res is not None:
            s.m.log(env.now, task_id, "FD_QUEUE", "")
            yield from s.scheduled_service(res, "Front Desk", s.p["svc_frontdesk"])
            s.m.log(env.now, task_id, "FD_DONE", "")
            # Loops
            fd_loops = 0
            while (fd_loops < s.p["max_fd_loops"]) and (random.random() < s.p["p_fd_insuff"]):
                fd_loops += 1
                s.m.loop_fd_insufficient += 1
                s.m.log(env.now, task_id, "FD_INSUFF", f"Missing info loop #{fd_loops}")
                yield env.timeout(s.p["fd_loop_delay"])
                s.m.log(env.now, task_id, "FD_RETRY_QUEUE", f"Loop #{fd_loops}")
                yield from s.scheduled_service(res, "Front Desk", s.p["svc_frontdesk"])
                s.m.log(env.now, task_id, "FD_RETRY_DONE", f"Loop #{fd_loops}")

    elif role == "Nurse":
        if res is not None:
            s.m.log(env.now, task_id, "NU_QUEUE", "")
            if random.random() < s.p["p_protocol"]:
                yield from s.scheduled_service(res, "Nurse", s.p["svc_nurse_protocol"], role_for_dist="NurseProtocol")
            else:
                yield from s.scheduled_service(res, "Nurse", s.p["svc_nurse"])
                s.m.log(env.now, task_id, "NU_DONE", "")
            # Loops
            nurse_loops = 0
            while (nurse_loops < s.p["max_nurse_loops"]) and (random.random() < s.p["p_nurse_insuff"]):
                nurse_loops += 1
                s.m.loop_nurse_insufficient += 1
                s.m.log(env.now, task_id, "NU_INSUFF", f"Back to FD loop #{nurse_loops}")
                if s.role_to_res["Front Desk"] is not None:
                    s.m.log(env.now, task_id, "FD_QUEUE", f"After nurse loop #{nurse_loops}")
                    yield from s.scheduled_service(s.role_to_res["Front Desk"], "Front Desk", s.p["svc_frontdesk"])
                    s.m.log(env.now, task_id, "FD_DONE", f"After nurse loop #{nurse_loops}")
                s.m.log(env.now, task_id, "NU_RECHECK_QUEUE", f"Loop #{nurse_loops}")
                yield from s.scheduled_service(res, "Nurse", max(0.0, 0.5 * s.p["svc_nurse"]))
                s.m.log(env.now, task_id, "NU_RECHECK_DONE", f"Loop #{nurse_loops}")

    elif role == "Provider":
        if res is not None:
            s.m.log(env.now, task_id, "PR_QUEUE", "")
            yield from s.scheduled_service(res, "Provider", s.p["svc_provider"])
            s.m.log(env.now, task_id, "PR_DONE", "")
            # Loops
            provider_loops = 0
            while (provider_loops < s.p["max_provider_loops"]) and (random.random() < s.p["p_provider_insuff"]):
                provider_loops += 1
                s.m.loop_provider_insufficient += 1
                s.m.log(env.now, task_id, "PR_INSUFF", f"Provider rework loop #{provider_loops}")
                yield env.timeout(s.p["provider_loop_delay"])
                s.m.log(env.now, task_id, "PR_RECHECK_QUEUE", f"Loop #{provider_loops}")
                yield from s.scheduled_service(res, "Provider", max(0.0, 0.5 * s.p["svc_provider"]))
                s.m.log(env.now, task_id, "PR_RECHECK_DONE", f"Loop #{provider_loops}")

    elif role == "Back Office":
        if res is not None:
            s.m.log(env.now, task_id, "BO_QUEUE", "")
            yield from s.scheduled_service(res, "Back Office", s.p["svc_backoffice"])
            s.m.log(env.now, task_id, "BO_DONE", "")
            # Loops
            bo_loops = 0
            while (bo_loops < s.p["max_backoffice_loops"]) and (random.random() < s.p["p_backoffice_insuff"]):
                bo_loops += 1
                s.m.loop_backoffice_insufficient += 1
                s.m.log(env.now, task_id, "BO_INSUFF", f"Back Office rework loop #{bo_loops}")
                yield env.timeout(s.p["backoffice_loop_delay"])
                s.m.log(env.now, task_id, "BO_RECHECK_QUEUE", f"Loop #{bo_loops}")
                yield from s.scheduled_service(res, "Back Office", max(0.0, 0.5 * s.p["svc_backoffice"]))
                s.m.log(env.now, task_id, "BO_RECHECK_DONE", f"Loop #{bo_loops}")

    # Route to next step using the interaction matrix
    row = s.p["route_matrix"].get(role, {DONE: 1.0})
    nxt = sample_next_role(row)
    return nxt

def task_lifecycle(env, task_id: str, s: CHCSystem, initial_role: str):
    """Run a task through roles until DONE (guard against infinite loops)."""
    s.m.task_arrival_time[task_id] = env.now
    s.m.arrivals_total += 1
    s.m.arrivals_by_role[initial_role] += 1
    s.m.log(env.now, task_id, "ARRIVE", f"Arrived at {initial_role}", arrival_t=env.now)

    role = initial_role
    for _ in range(60):  # generous guard
        nxt = yield from handle_role(env, task_id, s, role)
        if nxt == DONE:
            s.m.completed += 1
            s.m.task_completion_time[task_id] = env.now
            s.m.log(env.now, task_id, "DONE", "Task completed")
            return
        role = nxt

    # Safety fallback
    s.m.completed += 1
    s.m.task_completion_time[task_id] = env.now
    s.m.log(env.now, task_id, "DONE", "Max handoffs reached — forced completion")

def arrival_process_for_role(env, s: CHCSystem, role_name: str, rate_per_hour: int):
    """
    Independent Poisson arrivals to a given role (rate is integer per hour).
    """
    i = 0
    lam = max(0, int(rate_per_hour)) / 60.0
    while True:
        inter = random.expovariate(lam) if lam > 0 else 999999999
        yield env.timeout(inter)
        i += 1
        task_id = f"{role_name[:2].upper()}-{i:05d}"
        env.process(task_lifecycle(env, task_id, s, initial_role=role_name))

def monitor(env, s: CHCSystem):
    while True:
        s.m.time_stamps.append(env.now)
        for r in ROLES:
            res = s.role_to_res[r]
            self_q = len(res.queue) if res is not None else 0
            s.m.queues[r].append(self_q)
        yield env.timeout(1)

# =============================
# Diagram builder (Graphviz DOT)
# =============================
def build_process_graph(p: Dict) -> str:
    def _fmt_pct(x: float) -> str:
        try:
            return f"{100*float(x):.0f}%"
        except:
            return "0%"

    legend_fill = "#E9F7FF"  # top-left legend bg color

    cap = {
        "Front Desk": p.get("frontdesk_cap", 0),
        "Nurse": p.get("nurse_cap", 0),
        "Provider": p.get("provider_cap", 0),
        "Back Office": p.get("backoffice_cap", 0),
    }
    svc = {
        "Front Desk": p.get("svc_frontdesk", 0.0),
        "Nurse": p.get("svc_nurse", 0.0),
        "Provider": p.get("svc_provider", 0.0),
        "Back Office": p.get("svc_backoffice", 0.0),
    }
    svc_proto = p.get("svc_nurse_protocol", None)
    p_protocol = p.get("p_protocol", None)

    loops = {
        "Front Desk":  (p.get("p_fd_insuff", 0.0),        p.get("max_fd_loops", 0)),
        "Nurse":       (p.get("p_nurse_insuff", 0.0),     p.get("max_nurse_loops", 0)),
        "Provider":    (p.get("p_provider_insuff", 0.0),  p.get("max_provider_loops", 0)),
        "Back Office": (p.get("p_backoffice_insuff", 0.0),p.get("max_backoffice_loops", 0)),
    }

    route = p.get("route_matrix", {})
    roles = ["Front Desk", "Nurse", "Provider", "Back Office"]
    DONE = "Done"

    lines = [
        'digraph CHC {',
        '  rankdir=LR;',
        '  fontsize=12;',
        '  graph [size="10,4", nodesep=0.6, ranksep=0.8, overlap=false, pad="0.1,0.1"];',
        '  node [shape=roundrect, style=filled, fillcolor="#F7F7F7", color="#888", fontname="Helvetica", fontsize=10];',
        '  edge [color="#666", arrowsize=0.8, fontname="Helvetica", fontsize=9];'
    ]

    for r in roles:
        fill = "#EFEFEF" if cap.get(r, 0) <= 0 else "#F7F7F7"
        lines.append(f'  "{r}" [label="{r}\\ncap={cap.get(r,0)}\\nsvc≈{svc.get(r,0):.1f} min", fillcolor="{fill}"];')

    if p_protocol is not None and svc_proto is not None and cap.get("Nurse", 0) >= 0:
        proto_label = f'Nurse Protocol\\np={_fmt_pct(p_protocol)}\\nsvc≈{svc_proto:.1f} min'
        lines += [
            f'  "NurseProto" [shape=note, fillcolor="#FFFBE6", color="#B59F3B", label="{proto_label}", fontsize=9];',
            '  "Nurse" -> "NurseProto" [style=dotted, label=" info "];'
        ]

    lines.append('  "Done" [shape=doublecircle, fillcolor="#E8F5E9", color="#5E8D5B"];')

    for src, row in route.items():
        for tgt, prob in row.items():
            try:
                prob_f = float(prob)
            except:
                prob_f = 0.0
            if prob_f > 0:
                lines.append(f'  "{src}" -> "{tgt}" [label="{_fmt_pct(prob_f)}"];')

    for r, (p_loop, max_loops) in loops.items():
        try:
            p_f = float(p_loop)
        except:
            p_f = 0.0
        if p_f > 0:
            lines.append(f'  "{r}" -> "{r}" [style=dashed, color="#999", label="loop {_fmt_pct(p_f)} / max {max_loops}"];')

    # Legend (top-left, pinned)
    lines += [
        f'  legend [shape=box, style="rounded,filled", fillcolor="{legend_fill}", color="#AFC8D8", fontsize=8, '
        '          label="cap = capacity\\nsvc = mean svc time\\n→ routing prob\\n↺ loop prob / max"];',
        '  { rank=min; legend; }',
        '  legend -> "Front Desk" [style=invis, weight=9999];'
    ]

    lines.append('}')
    return "\n".join(lines)

# =============================
# NEW: Run single replication
# =============================
def run_single_replication(p: Dict, seed: int) -> Metrics:
    """Run one replication and return the metrics object."""
    random.seed(seed)
    np.random.seed(seed)
    
    metrics = Metrics()
    env = simpy.Environment()
    system = CHCSystem(env, p, metrics, seed_offset=seed)

    for role in ROLES:
        rate = int(p["arrivals_per_hour_by_role"].get(role, 0))
        env.process(arrival_process_for_role(env, system, role, rate))

    env.process(monitor(env, system))
    env.run(until=p["sim_minutes"])
    
    return metrics

# =============================
# NEW: Aggregate metrics across replications
# =============================
# =============================
# Visualization functions
# =============================
def plot_utilization_heatmap(all_metrics: List[Metrics], p: Dict, active_roles: List[str]):
    """
    Utilization heatmap: roles (y-axis) vs hour of workday (x-axis)
    Shows average utilization across all replications for each hour slot.
    """
    open_minutes = p["open_minutes"]
    hours_in_day = int(np.ceil(open_minutes / 60.0))
    
    # Calculate utilization by hour for each role across replications
    role_hour_utils = {r: np.zeros(hours_in_day) for r in active_roles}
    
    # Better approach: calculate from actual service times
    open_time_per_hour = 60  # minutes
    for metrics in all_metrics:
        for r in active_roles:
            capacity = {
                "Front Desk": p["frontdesk_cap"],
                "Nurse": p["nurse_cap"],
                "Provider": p["provider_cap"],
                "Back Office": p["backoffice_cap"]
            }[r]
            
            if capacity > 0:
                # Distribute service time evenly across open hours (approximation)
                total_service = metrics.service_time_sum[r]
                num_days = p["sim_minutes"] / DAY_MIN
                avg_service_per_day = total_service / max(1, num_days)
                service_per_hour = avg_service_per_day / max(1, hours_in_day)
                
                for h in range(hours_in_day):
                    util = service_per_hour / (capacity * open_time_per_hour)
                    role_hour_utils[r][h] += util
    
    # Average across replications
    num_reps = len(all_metrics)
    for r in active_roles:
        role_hour_utils[r] /= num_reps
    
    # Create heatmap (smaller size)
    fig, ax = plt.subplots(figsize=(8, 3), dpi=80)
    
    # Prepare data matrix
    data = np.array([role_hour_utils[r] for r in active_roles])
    
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1, interpolation='nearest')
    
    # Set ticks - use "Hour 1, Hour 2" format
    ax.set_xticks(np.arange(hours_in_day))
    ax.set_xticklabels([f"Hour {h+1}" for h in range(hours_in_day)])
    ax.set_yticks(np.arange(len(active_roles)))
    ax.set_yticklabels(active_roles)
    
    # Labels
    ax.set_xlabel('Hour of Workday', fontsize=11)
    ax.set_ylabel('Role', fontsize=11)
    ax.set_title('Utilization Heatmap by Role and Hour', fontsize=13, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Utilization', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(active_roles)):
        for j in range(hours_in_day):
            text = ax.text(j, i, f'{data[i, j]:.0%}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    return fig

def plot_queue_over_time(all_metrics: List[Metrics], p: Dict, active_roles: List[str]):
    """
    Line chart showing queue length over simulation time for each role.
    Shows mean ± std across replications.
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=80)
    
    # Collect queue data from all replications
    max_len = max(len(m.time_stamps) for m in all_metrics)
    
    colors = {'Front Desk': '#1f77b4', 'Nurse': '#ff7f0e', 'Provider': '#2ca02c', 'Back Office': '#d62728'}
    
    for role in active_roles:
        # Pad all replication data to same length
        all_queues = []
        for metrics in all_metrics:
            q = metrics.queues[role]
            if len(q) < max_len:
                q = q + [q[-1]] * (max_len - len(q)) if len(q) > 0 else [0] * max_len
            all_queues.append(q[:max_len])
        
        # Calculate mean and std
        queue_array = np.array(all_queues)
        mean_queue = np.mean(queue_array, axis=0)
        std_queue = np.std(queue_array, axis=0)
        
        # Time axis (convert minutes to days for readability)
        time_days = np.array(all_metrics[0].time_stamps[:max_len]) / DAY_MIN
        
        # Plot
        color = colors.get(role, '#333333')
        ax.plot(time_days, mean_queue, label=role, color=color, linewidth=2)
        ax.fill_between(time_days, mean_queue - std_queue, mean_queue + std_queue, 
                        alpha=0.2, color=color)
    
    ax.set_xlabel('Time (days)', fontsize=11)
    ax.set_ylabel('Queue Length (tasks waiting)', fontsize=11)
    ax.set_title('Queue Length Over Time', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_rework_impact(all_metrics: List[Metrics], p: Dict, active_roles: List[str]):
    """
    Stacked bar chart showing original work time vs rework time by role.
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=80)
    
    # Calculate rework vs original time
    original_time = {r: [] for r in active_roles}
    rework_time = {r: [] for r in active_roles}
    
    for metrics in all_metrics:
        # Count loops as proxy for rework
        loop_counts = {
            "Front Desk": metrics.loop_fd_insufficient,
            "Nurse": metrics.loop_nurse_insufficient,
            "Provider": metrics.loop_provider_insufficient,
            "Back Office": metrics.loop_backoffice_insufficient
        }
        
        for role in active_roles:
            total_time = metrics.service_time_sum[role]
            
            # Estimate rework time (loops typically take 50% of original service time)
            loops = loop_counts.get(role, 0)
            svc_time = {
                "Front Desk": p["svc_frontdesk"],
                "Nurse": p["svc_nurse"],
                "Provider": p["svc_provider"],
                "Back Office": p["svc_backoffice"]
            }[role]
            
            estimated_rework = loops * svc_time * 0.5
            estimated_original = total_time - estimated_rework
            
            original_time[role].append(max(0, estimated_original))
            rework_time[role].append(estimated_rework)
    
    # Average across replications
    original_means = [np.mean(original_time[r]) for r in active_roles]
    rework_means = [np.mean(rework_time[r]) for r in active_roles]
    
    x = np.arange(len(active_roles))
    width = 0.6
    
    p1 = ax.bar(x, original_means, width, label='Original Work', color='#3498db')
    p2 = ax.bar(x, rework_means, width, bottom=original_means, label='Rework Time', color='#e74c3c')
    
    ax.set_xlabel('Role', fontsize=11)
    ax.set_ylabel('Total Time Spent (minutes)', fontsize=11)
    ax.set_title('Rework Impact: Original Work vs Rework Time', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(active_roles)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels showing rework proportion
    for i, (orig, rew) in enumerate(zip(original_means, rework_means)):
        total = orig + rew
        if total > 0 and rew > 0:
            pct = 100 * rew / total
            ax.text(i, total + 50, f'{pct:.1f}% rework', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_daily_throughput(all_metrics: List[Metrics], p: Dict):
    """
    Line chart showing daily throughput trend with confidence bands.
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=80)
    
    num_days = max(1, int(p["sim_minutes"] // DAY_MIN))
    
    # Collect daily completion counts from all replications
    daily_completions = []
    for metrics in all_metrics:
        comp_times = metrics.task_completion_time
        daily = []
        for d in range(num_days):
            start_t = d * DAY_MIN
            end_t = (d + 1) * DAY_MIN
            completed = sum(1 for ct in comp_times.values() if start_t <= ct < end_t)
            daily.append(completed)
        daily_completions.append(daily)
    
    # Calculate statistics
    daily_array = np.array(daily_completions)
    mean_daily = np.mean(daily_array, axis=0)
    std_daily = np.std(daily_array, axis=0)
    
    days = np.arange(1, num_days + 1)
    
    ax.plot(days, mean_daily, marker='o', linewidth=2, markersize=8, color='#3498db', label='Mean Completions')
    ax.fill_between(days, mean_daily - std_daily, mean_daily + std_daily, 
                    alpha=0.3, color='#3498db', label='± 1 SD')
    
    ax.set_xlabel('Day', fontsize=11)
    ax.set_ylabel('Tasks Completed', fontsize=11)
    ax.set_title('Daily Throughput Trend', fontsize=13, fontweight='bold')
    ax.set_xticks(days)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_work_vs_wait(all_metrics: List[Metrics], p: Dict, active_roles: List[str]):
    """
    Stacked bar chart showing average work time vs wait time per completed task by role.
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=80)
    
    work_times = {r: [] for r in active_roles}
    wait_times = {r: [] for r in active_roles}
    
    for metrics in all_metrics:
        completed = len(metrics.task_completion_time)
        if completed > 0:
            for role in active_roles:
                avg_work = metrics.service_time_sum[role] / completed
                avg_wait = np.mean(metrics.waits[role]) if len(metrics.waits[role]) > 0 else 0
                work_times[role].append(avg_work)
                wait_times[role].append(avg_wait)
        else:
            for role in active_roles:
                work_times[role].append(0)
                wait_times[role].append(0)
    
    # Average across replications
    work_means = [np.mean(work_times[r]) for r in active_roles]
    wait_means = [np.mean(wait_times[r]) for r in active_roles]
    
    x = np.arange(len(active_roles))
    width = 0.6
    
    p1 = ax.bar(x, work_means, width, label='Active Working Time', color='#2ecc71')
    p2 = ax.bar(x, wait_means, width, bottom=work_means, label='Waiting for Resources', color='#f39c12')
    
    ax.set_xlabel('Role', fontsize=11)
    ax.set_ylabel('Time per Completed Task (minutes)', fontsize=11)
    ax.set_title('Work vs Wait Time by Role', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(active_roles)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add efficiency percentage
    for i, (work, wait) in enumerate(zip(work_means, wait_means)):
        total = work + wait
        if total > 0:
            efficiency = 100 * work / total
            ax.text(i, total + 0.5, f'{efficiency:.0f}% efficient', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def help_icon(help_text: str):
    """Helper function to create an info icon with expandable help text."""
    with st.expander("ℹ️ How is this calculated?"):
        st.caption(help_text)

def aggregate_replications(p: Dict, all_metrics: List[Metrics], active_roles: List[str]):
    """
    Aggregate metrics from multiple replications and return summary DataFrames with mean ± std.
    """
    num_reps = len(all_metrics)
    
    # Helper to format mean ± std
    def fmt_mean_std(values):
        m = np.mean(values)
        s = np.std(values, ddof=1) if len(values) > 1 else 0.0
        return f"{m:.1f} ± {s:.1f}"
    
    def fmt_mean_std_pct(values):
        m = np.mean(values)
        s = np.std(values, ddof=1) if len(values) > 1 else 0.0
        return f"{m:.1f}% ± {s:.1f}%"
    
    # ========== A) Flow time metrics ==========
    flow_avg_list = []
    flow_med_list = []
    same_day_list = []
    time_at_role_lists = {r: [] for r in ROLES}
    
    for metrics in all_metrics:
        comp_times = metrics.task_completion_time
        arr_times = metrics.task_arrival_time
        done_ids = set(comp_times.keys())
        
        if len(done_ids) > 0:
            tt = np.array([comp_times[k] - arr_times.get(k, comp_times[k]) for k in done_ids])
            flow_avg_list.append(float(np.mean(tt)))
            flow_med_list.append(float(np.median(tt)))
            
            for r in ROLES:
                time_at_role_lists[r].append(metrics.service_time_sum[r] / len(done_ids))
            
            same_day = sum(1 for k in done_ids 
                          if int(arr_times.get(k, 0) // DAY_MIN) == int(comp_times[k] // DAY_MIN))
            same_day_list.append(100.0 * same_day / len(done_ids))
        else:
            flow_avg_list.append(0.0)
            flow_med_list.append(0.0)
            same_day_list.append(0.0)
            for r in ROLES:
                time_at_role_lists[r].append(0.0)
    
    flow_df = pd.DataFrame([
        {"Metric": "Average turnaround time (minutes)", "Value": fmt_mean_std(flow_avg_list)},
        {"Metric": "Median turnaround time (minutes)", "Value": fmt_mean_std(flow_med_list)},
        {"Metric": "Same-day completion", "Value": fmt_mean_std_pct(same_day_list)}
    ])
    
    time_at_role_df = pd.DataFrame([
        {"Role": r, "Avg time at role (min) per completed task": fmt_mean_std(time_at_role_lists[r])}
        for r in active_roles
    ])
    
    # ========== B) Queue metrics ==========
    q_avg_lists = {r: [] for r in ROLES}
    q_max_lists = {r: [] for r in ROLES}
    
    for metrics in all_metrics:
        for r in ROLES:
            q_avg_lists[r].append(np.mean(metrics.queues[r]) if len(metrics.queues[r]) > 0 else 0.0)
            q_max_lists[r].append(np.max(metrics.queues[r]) if len(metrics.queues[r]) > 0 else 0)
    
    queue_df = pd.DataFrame([
        {
            "Role": r,
            "Avg queue length": fmt_mean_std(q_avg_lists[r]),
            "Max queue length": fmt_mean_std(q_max_lists[r])
        }
        for r in active_roles
    ])
    
    # ========== C) Rework metrics ==========
    rework_pct_list = []
    loop_counts_lists = {
        "Front Desk": [],
        "Nurse": [],
        "Provider": [],
        "Back Office": []
    }
    
    for metrics in all_metrics:
        rework_tasks = set()
        for t, name, step, note, _arr in metrics.events:
            if step.endswith("INSUFF") or "RECHECK" in step:
                rework_tasks.add(name)
        
        done_ids = set(metrics.task_completion_time.keys())
        rework_pct_list.append(100.0 * len(rework_tasks & done_ids) / max(1, len(done_ids)))
        
        loop_counts_lists["Front Desk"].append(metrics.loop_fd_insufficient)
        loop_counts_lists["Nurse"].append(metrics.loop_nurse_insufficient)
        loop_counts_lists["Provider"].append(metrics.loop_provider_insufficient)
        loop_counts_lists["Back Office"].append(metrics.loop_backoffice_insufficient)
    
    rework_overview_df = pd.DataFrame([
        {"Metric": "% tasks with any rework", "Value": fmt_mean_std_pct(rework_pct_list)}
    ])
    
    total_loops_list = [sum(loop_counts_lists[r][i] for r in ROLES) for i in range(num_reps)]
    loop_origin_df = pd.DataFrame([
        {
            "Role": r,
            "Loop Count": fmt_mean_std(loop_counts_lists[r]),
            "Share": fmt_mean_std_pct([
                100.0 * loop_counts_lists[r][i] / total_loops_list[i] if total_loops_list[i] > 0 else 0.0
                for i in range(num_reps)
            ])
        }
        for r in active_roles
    ])
    
    # ========== D) Throughput (daily averages) ==========
    num_days = max(1, int(p["sim_minutes"] // DAY_MIN))
    daily_arrivals_lists = [[] for _ in range(num_days)]
    daily_completed_lists = [[] for _ in range(num_days)]
    daily_from_prev_lists = [[] for _ in range(num_days)]
    daily_for_next_lists = [[] for _ in range(num_days)]
    
    for metrics in all_metrics:
        arr_times = metrics.task_arrival_time
        comp_times = metrics.task_completion_time
        
        for d in range(num_days):
            start_t = d * DAY_MIN
            end_t = (d + 1) * DAY_MIN
            
            arrivals_today = sum(1 for k, at in arr_times.items() if start_t <= at < end_t)
            completed_today = sum(1 for k, ct in comp_times.items() if start_t <= ct < end_t)
            from_prev = sum(1 for k, at in arr_times.items() 
                          if at < start_t and (k not in comp_times or comp_times[k] >= start_t))
            for_next = sum(1 for k, at in arr_times.items() 
                         if at < end_t and (k not in comp_times or comp_times[k] >= end_t))
            
            daily_arrivals_lists[d].append(arrivals_today)
            daily_completed_lists[d].append(completed_today)
            daily_from_prev_lists[d].append(from_prev)
            daily_for_next_lists[d].append(for_next)
    
    throughput_rows = []
    for d in range(num_days):
        throughput_rows.append({
            "Day": d + 1,
            "Total tasks that day": fmt_mean_std(daily_arrivals_lists[d]),
            "Completed tasks": fmt_mean_std(daily_completed_lists[d]),
            "Tasks from previous day": fmt_mean_std(daily_from_prev_lists[d]),
            "Tasks for next day": fmt_mean_std(daily_for_next_lists[d])
        })
    throughput_full_df = pd.DataFrame(throughput_rows)
    
    # ========== E) Utilization ==========
    open_time_available = effective_open_minutes(p["sim_minutes"], p["open_minutes"])
    denom = {
        "Front Desk": max(1, p["frontdesk_cap"]) * open_time_available,
        "Nurse": max(1, p["nurse_cap"]) * open_time_available,
        "Provider": max(1, p["provider_cap"]) * open_time_available,
        "Back Office": max(1, p["backoffice_cap"]) * open_time_available,
    }
    
    util_lists = {r: [] for r in ROLES}
    for metrics in all_metrics:
        for r in ROLES:
            util_lists[r].append(100.0 * metrics.service_time_sum[r] / max(1, denom[r]))
    
    util_overall_list = [np.mean([util_lists[r][i] for r in ROLES]) for i in range(num_reps)]
    
    util_rows = [{"Role": r, "Utilization": fmt_mean_std_pct(util_lists[r])} for r in active_roles]
    util_rows.append({"Role": "Overall", "Utilization": fmt_mean_std_pct(util_overall_list)})
    util_df = pd.DataFrame(util_rows)
    
    # ========== Summary row for comparison ==========
    summary_row = {
        "Name": "",
        "Avg turnaround (min)": np.mean(flow_avg_list),
        "Median turnaround (min)": np.mean(flow_med_list),
        "Same-day completion (%)": np.mean(same_day_list),
        "Rework (% of completed)": np.mean(rework_pct_list),
        "Utilization overall (%)": np.mean(util_overall_list),
    }
    summary_df = pd.DataFrame([summary_row])
    
    return {
        "flow_df": flow_df,
        "time_at_role_df": time_at_role_df,
        "queue_df": queue_df,
        "rework_overview_df": rework_overview_df,
        "loop_origin_df": loop_origin_df,
        "throughput_full_df": throughput_full_df,
        "util_df": util_df,
        "summary_df": summary_df
    }

# =============================
# Streamlit UI (2-step wizard)
# =============================
st.set_page_config(page_title="CHC Workflow Simulator", layout="wide")
st.title("CHC Workflow Simulator")

if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 1
if "results" not in st.session_state:
    st.session_state["results"] = None
if "design" not in st.session_state:
    st.session_state["design"] = None
if "design_saved" not in st.session_state:
    st.session_state.design_saved = False
# NEW: track if user has run the simulation (to toggle process preview dropdown)
if "ran" not in st.session_state:
    st.session_state.ran = False
# NEW: store saved interventions/runs
if "saved_runs" not in st.session_state:
    st.session_state.saved_runs = []  # list of dicts

def go_next():
    st.session_state.wizard_step = min(2, st.session_state.wizard_step + 1)
def go_back():
    st.session_state.wizard_step = max(1, st.session_state.wizard_step - 1)

def _init_ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

def prob_input(label: str, key: str, default: float = 0.0, help: str | None = None, disabled: bool = False) -> float:
    if key not in st.session_state:
        st.session_state[key] = f"{float(default):.2f}"
    raw = st.text_input(label, value=st.session_state[key], key=key, help=help, disabled=disabled)
    try:
        val = float(str(raw).replace(",", "."))
    except ValueError:
        val = float(default)
    val = max(0.0, min(1.0, val))
    st.caption(f"{val:.2f}")
    return val

# ---- Excel engine detector ----
def _excel_engine():
    """Return a working pandas ExcelWriter engine or None if unavailable."""
    try:
        import xlsxwriter  # noqa: F401
        return "xlsxwriter"
    except Exception:
        try:
            import openpyxl  # noqa: F401
            return "openpyxl"
        except Exception:
            return None

# ---- Download helpers (engine-aware) ----
def _workbook_from_run(run: dict, engine: str | None = None) -> dict:
    """
    Build a file for a single run.
    Returns dict: {"bytes": ..., "mime": ..., "ext": "..."}.
    Prefers XLSX; falls back to ZIP of CSVs if no Excel engine.
    """
    if engine is None:
        engine = _excel_engine()

    if engine:
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine=engine) as xw:
            run["flow_df"].to_excel(xw, index=False, sheet_name="Flow")
            run["time_at_role_df"].to_excel(xw, index=False, sheet_name="TimeAtRole")
            run["queue_df"].to_excel(xw, index=False, sheet_name="Queue")
            run["rework_overview_df"].to_excel(xw, index=False, sheet_name="Rework")
            run["loop_origin_df"].to_excel(xw, index=False, sheet_name="ReworkByRole")
            run["throughput_full_df"].to_excel(xw, index=False, sheet_name="ThroughputDaily")
            run["util_df"].to_excel(xw, index=False, sheet_name="Utilization")
            run["summary_df"].to_excel(xw, index=False, sheet_name="Summary")
            # NEW: Add events dataframe if present
            if "events_df" in run:
                run["events_df"].to_excel(xw, index=False, sheet_name="RunLog")
        return {
            "bytes": bio.getvalue(),
            "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "ext": "xlsx",
        }
    else:
        # Fallback: ZIP of CSVs (no extra dependencies)
        import zipfile
        bio = BytesIO()
        with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
            def _w(name, df):
                z.writestr(f"{name}.csv", df.to_csv(index=False))
            _w("Flow", run["flow_df"])
            _w("TimeAtRole", run["time_at_role_df"])
            _w("Queue", run["queue_df"])
            _w("Rework", run["rework_overview_df"])
            _w("ReworkByRole", run["loop_origin_df"])
            _w("ThroughputDaily", run["throughput_full_df"])
            _w("Utilization", run["util_df"])
            _w("Summary", run["summary_df"])
            if "events_df" in run:
                _w("RunLog", run["events_df"])
        return {"bytes": bio.getvalue(), "mime": "application/zip", "ext": "zip"}

def _all_runs_workbook(runs: list, engine: str | None = None) -> dict:
    """
    Build a combined file for all saved runs.
    Returns dict: {"bytes": ..., "mime": ..., "ext": "..."}.
    """
    if engine is None:
        engine = _excel_engine()

    if engine:
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine=engine) as xw:
            if runs:
                comp = pd.concat([r["summary_df"] for r in runs], ignore_index=True)
                comp.to_excel(xw, index=False, sheet_name="SummaryAll")
            for r in runs:
                name = r["name"][:28]  # Excel sheet name limit
                r["summary_df"].to_excel(xw, index=False, sheet_name=f"{name}_Summary")
                r["flow_df"].to_excel(xw, index=False, sheet_name=f"{name}_Flow")
                r["queue_df"].to_excel(xw, index=False, sheet_name=f"{name}_Queue")
                r["util_df"].to_excel(xw, index=False, sheet_name=f"{name}_Util")
        return {
            "bytes": bio.getvalue(),
            "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "ext": "xlsx",
        }
    else:
        # Fallback: ZIP of CSVs, one folder per run (by name)
        import zipfile
        bio = BytesIO()
        with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
            if runs:
                comp = pd.concat([r["summary_df"] for r in runs], ignore_index=True)
                z.writestr("SummaryAll.csv", comp.to_csv(index=False))
            for r in runs:
                base = r["name"]
                def _w(name, df):
                    z.writestr(f"{base}/{name}.csv", df.to_csv(index=False))
                _w("Summary", r["summary_df"])
                _w("Flow", r["flow_df"])
                _w("Queue", r["queue_df"])
                _w("Utilization", r["util_df"])
        return {"bytes": bio.getvalue(), "mime": "application/zip", "ext": "zip"}

def _runlog_workbook(events_df: pd.DataFrame, engine: str | None = None) -> dict:
    """
    Build a single-sheet Excel file with the DES run log.
    Returns dict: {"bytes": ..., "mime": ..., "ext": "..."}
    """
    if engine is None:
        engine = _excel_engine()

    from io import BytesIO
    if engine:
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine=engine) as xw:
            events_df.to_excel(xw, index=False, sheet_name="RunLog")
        return {
            "bytes": bio.getvalue(),
            "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "ext": "xlsx",
        }
    else:
        # Require Excel engine (matches your preference to avoid CSV zips)
        st.error(
            "Excel export requires an engine. Add `xlsxwriter` (recommended) or `openpyxl` "
            "to your environment and rerun."
        )
        return {"bytes": b"", "mime": "application/octet-stream", "ext": "xlsx"}


# -------- STEP 1: DESIGN --------
if st.session_state.wizard_step == 1:

    st.markdown("""
    ### 🏥 **Design Your Clinic**
    Customize staffing, arrivals, times, loops, and routing. **Staffing below updates the page live**;
    other inputs are saved when you click **Save**.
    """)

    # Staffing (live)
    st.markdown("### 👥 Staffing (on duty)")
    cStaff1, cStaff2, cStaff3, cStaff4 = st.columns(4)
    with cStaff1:
        st.session_state.fd_cap = st.number_input("Front Desk staff", 0, 50, _init_ss("fd_cap", 3), 1, "%d",
                                                  help="Number of front desk staff simultaneously available.")
    with cStaff2:
        st.session_state.nurse_cap = st.number_input("Nurses / MAs", 0, 50, _init_ss("nurse_cap", 2), 1, "%d",
                                                     help="Number of nurses/medical assistants on duty.")
    with cStaff3:
        st.session_state.provider_cap = st.number_input("Providers", 0, 50, _init_ss("provider_cap", 1), 1, "%d",
                                                        help="Number of providers on duty.")
    with cStaff4:
        st.session_state.bo_cap = st.number_input("Back Office staff", 0, 50, _init_ss("backoffice_cap", 1), 1, "%d",
                                                  help="Number of back-office staff on duty.")

    fd_off = (st.session_state.fd_cap == 0)
    nu_off = (st.session_state.nurse_cap == 0)
    pr_off = (st.session_state.provider_cap == 0)
    bo_off = (st.session_state.bo_cap == 0)

    cap_map = {
        "Front Desk": st.session_state.fd_cap,
        "Nurse": st.session_state.nurse_cap,
        "Provider": st.session_state.provider_cap,
        "Back Office": st.session_state.bo_cap,
    }

    with st.form("design_form", clear_on_submit=False):
        # Full-width controls
        st.markdown("### Simulation horizon & variability")
        sim_days = st.number_input("Days to simulate", 1, 30, _init_ss("sim_days", 5), 1, "%d",
                                   help="Number of 24-hour days to include in the simulation.")
        open_hours = st.number_input("Hours open per day", 1, 24, _init_ss("open_hours", 10), 1, "%d",
                                     help="Clinic operating hours per day during which work can be performed.")
        cv_speed = st.slider("Task speed variability (CV)", 0.0, 0.8, _init_ss("cv_speed", 0.25), 0.05,
                             help="How variable individual task times are around their mean (coefficient of variation).")

        st.markdown("### Arrivals per hour at each role")
        cA1, cA2, cA3, cA4 = st.columns(4)
        with cA1:
            arr_fd = st.number_input("→ Front Desk", 0, 500, _init_ss("arr_fd", 15), 1, "%d",
                                     help="Average number of tasks arriving to the Front Desk each hour.",
                                     disabled=fd_off)
        with cA2:
            arr_nu = st.number_input("→ Nurse / MAs", 0, 500, _init_ss("arr_nu", 20), 1, "%d",
                                     help="Average number of tasks arriving directly to the Nurse/MA queue per hour.",
                                     disabled=nu_off)
        with cA3:
            arr_pr = st.number_input("→ Provider", 0, 500, _init_ss("arr_pr", 10), 1, "%d",
                                     help="Average number of tasks arriving directly to the Provider per hour.",
                                     disabled=pr_off)
        with cA4:
            arr_bo = st.number_input("→ Back Office", 0, 500, _init_ss("arr_bo", 5), 1, "%d",
                                     help="Average number of tasks arriving directly to the Back Office per hour.",
                                     disabled=bo_off)

        st.markdown("### Role availability (minutes per hour)")
        st.caption("How many minutes each hour each role is available to work on these tasks.")
        cAv1, cAv2, cAv3, cAv4 = st.columns(4)
        with cAv1:
            avail_fd = st.number_input("Front Desk", 0, 60, _init_ss("avail_fd", 45), 1, "%d",
                                    help="Minutes per hour Front Desk staff are available.",
                                    disabled=fd_off)
        with cAv2:
            avail_nu = st.number_input("Nurse / MAs", 0, 60, _init_ss("avail_nu", 20), 1, "%d",
                                   help="Minutes per hour Nurses are available.",
                                   disabled=nu_off)
        with cAv3:
            avail_pr = st.number_input("Providers", 0, 60, _init_ss("avail_pr", 30), 1, "%d",
                                   help="Minutes per hour Providers are available.",
                                   disabled=pr_off)
        with cAv4:
            avail_bo = st.number_input("Back Office", 0, 60, _init_ss("avail_bo", 45), 1, "%d",
                                   help="Minutes per hour Back Office staff are available.",
                                   disabled=bo_off)

        with st.expander("Additional (optional) — service times, loops & interaction matrix", expanded=False):
            st.markdown("#### Service times (mean minutes)")
            cS1, cS2 = st.columns(2)
            with cS1:
                svc_frontdesk = st.slider("Front Desk", 0.0, 30.0, _init_ss("svc_frontdesk", 3.0), 0.5,
                                          help="Average time to process a task at the Front Desk.",
                                          disabled=fd_off)
                svc_nurse_protocol = st.slider("Nurse Protocol", 0.0, 30.0, _init_ss("svc_nurse_protocol", 2.0), 0.5,
                                               help="Average time when a task is handled entirely by standing nurse protocol.",
                                               disabled=nu_off)
                svc_nurse = st.slider("Nurse (non-protocol)", 0.0, 40.0, _init_ss("svc_nurse", 4.0), 0.5,
                                      help="Average time for a standard nurse/MA task (non-protocol).",
                                      disabled=nu_off)
            with cS2:
                svc_provider = st.slider("Provider", 0.0, 60.0, _init_ss("svc_provider", 6.0), 0.5,
                                         help="Average time for a provider to complete their part of a task.",
                                         disabled=pr_off)
                svc_backoffice = st.slider("Back Office", 0.0, 60.0, _init_ss("svc_backoffice", 5.0), 0.5,
                                           help="Average time for back-office processing.",
                                           disabled=bo_off)
                p_protocol = st.slider("Probability Nurse resolves via protocol", 0.0, 1.0, _init_ss("p_protocol", 0.40), 0.05,
                                       help="Chance that the nurse protocol resolves the task without needing the provider.",
                                       disabled=nu_off)

            st.markdown("#### Loops (rework probabilities, caps, and delays)")
            cL1, cL2 = st.columns(2)
            with cL1:
                p_fd_insuff = st.slider("Front Desk: probability of missing info", 0.0, 1.0,
                                        _init_ss("p_fd_insuff", 0.15), 0.01,
                                        help="Per pass: probability that Front Desk triggers a rework cycle.",
                                        disabled=fd_off)
                max_fd_loops = st.number_input("Front Desk: max loops", 0, 10,
                                               _init_ss("max_fd_loops", 2), 1, "%d",
                                               help="Cap on the number of FD rework cycles per task.",
                                               disabled=fd_off)
                fd_loop_delay = st.slider("Front Desk: rework delay (min)", 0.0, 60.0,
                                          _init_ss("fd_loop_delay", 5.0), 0.5,
                                          help="Delay between discovering missing info and retrying at FD.",
                                          disabled=fd_off)

                p_nurse_insuff = st.slider("Nurse: probability of insufficient info", 0.0, 1.0,
                                           _init_ss("p_nurse_insuff", 0.10), 0.01,
                                           help="Per pass: probability that Nurse work requires rework.",
                                           disabled=nu_off)
                max_nurse_loops = st.number_input("Nurse: max loops", 0, 10,
                                                  _init_ss("max_nurse_loops", 2), 1, "%d",
                                                  help="Cap on the number of Nurse rework cycles per task.",
                                                  disabled=nu_off)
            with cL2:
                p_provider_insuff = st.slider("Provider: probability of rework needed", 0.0, 1.0,
                                              _init_ss("p_provider_insuff", 0.08), 0.01,
                                              help="Per pass: probability that Provider requires rework.",
                                              disabled=pr_off)
                max_provider_loops = st.number_input("Provider: max loops", 0, 10,
                                                     _init_ss("max_provider_loops", 2), 1, "%d",
                                                     help="Cap on the number of Provider rework cycles per task.",
                                                     disabled=pr_off)
                provider_loop_delay = st.slider("Provider: rework delay (min)", 0.0, 60.0,
                                                _init_ss("provider_loop_delay", 5.0), 0.5,
                                                help="Delay before Provider rechecks the task.",
                                                disabled=pr_off)

                p_backoffice_insuff = st.slider("Back Office: probability of rework needed", 0.0, 1.0,
                                                _init_ss("p_backoffice_insuff", 0.05), 0.01,
                                                help="Per pass: probability that Back Office requires rework.",
                                                disabled=bo_off)
                max_backoffice_loops = st.number_input("Back Office: max loops", 0, 10,
                                                       _init_ss("max_backoffice_loops", 2), 1, "%d",
                                                       help="Cap on the number of Back Office rework cycles per task.",
                                                       disabled=bo_off)
                backoffice_loop_delay = st.slider("Back Office: rework delay (min)", 0.0, 60.0,
                                                  _init_ss("backoffice_loop_delay", 5.0), 0.5,
                                                  help="Delay before Back Office rechecks the task.",
                                                  disabled=bo_off)

            st.markdown("#### Interaction matrix — Routing Probabilities")
            st.caption("Self-routing is disabled. You cannot route to roles with 0 capacity.")

            route: Dict[str, Dict[str, float]] = {}
            def route_row_ui(from_role: str, defaults: Dict[str, float], disabled_source: bool = False) -> Dict[str, float]:
                st.markdown(f"**{from_role} →**")
                targets = [r for r in ROLES if r != from_role] + [DONE]
                cols = st.columns(len(targets))
                row: Dict[str, float] = {}
                for i, tgt in enumerate(targets):
                    tgt_disabled = disabled_source or (tgt in ROLES and cap_map[tgt] == 0)
                    label_name = "Done" if tgt == DONE else tgt
                    key_name = f"r_{from_role}_{'done' if tgt==DONE else label_name.replace(' ','_').lower()}"
                    default_val = float(defaults.get(tgt, 0.0))
                    with cols[i]:
                        val = prob_input(
                            f"to {label_name} ({from_role})",
                            key=key_name,
                            default=(0.0 if tgt_disabled else default_val),
                            help="Routing probability from this role to the target.",
                            disabled=tgt_disabled
                        )
                        if tgt_disabled:
                            val = 0.0
                    row[tgt] = val
                return row

            # Non-zero, realistic defaults
            route["Front Desk"]  = route_row_ui("Front Desk",  {"Nurse": 0.50, "Provider": 0.10, "Back Office": 0.10, DONE: 0.30}, disabled_source=fd_off)
            route["Nurse"]       = route_row_ui("Nurse",       {"Provider": 0.40, "Back Office": 0.20, DONE: 0.40}, disabled_source=nu_off)
            route["Provider"]    = route_row_ui("Provider",    {"Back Office": 0.30, DONE: 0.70}, disabled_source=pr_off)
            route["Back Office"] = route_row_ui("Back Office", {"Front Desk": 0.10, "Nurse": 0.10, "Provider": 0.10, DONE: 0.70}, disabled_source=bo_off)

        saved = st.form_submit_button("Save", use_container_width=True)

        if saved:
            open_minutes = int(open_hours * MIN_PER_HOUR)
            sim_minutes = int(sim_days * DAY_MIN)

            # sanitize routing
            for r in ROLES:
                if r in route:
                    route[r].pop(r, None)
            for r in ROLES:
                if r in route:
                    for tgt in list(route[r].keys()):
                        if tgt in ROLES and cap_map[tgt] == 0:
                            route[r][tgt] = 0.0

            st.session_state["design"] = dict(
                sim_minutes=sim_minutes,
                open_minutes=open_minutes,
                frontdesk_cap=st.session_state.fd_cap,
                nurse_cap=st.session_state.nurse_cap,
                provider_cap=st.session_state.provider_cap,
                backoffice_cap=st.session_state.bo_cap,
                arrivals_per_hour_by_role={
                    "Front Desk": int(arr_fd),
                    "Nurse":      int(arr_nu),
                    "Provider":   int(arr_pr),
                    "Back Office":int(arr_bo),
                },
                availability_per_hour={
                    "Front Desk": int(avail_fd),
                    "Nurse":      int(avail_nu),
                    "Provider":   int(avail_pr),
                    "Back Office":int(avail_bo),
                },
                svc_frontdesk=svc_frontdesk, svc_nurse_protocol=svc_nurse_protocol, svc_nurse=svc_nurse,
                svc_provider=svc_provider,   svc_backoffice=svc_backoffice,
                dist_role={"Front Desk":"normal","NurseProtocol":"normal","Nurse":"exponential","Provider":"exponential","Back Office":"exponential"},
                cv_speed=cv_speed,
                emr_overhead={"Front Desk":0.5,"Nurse":0.5,"NurseProtocol":0.5,"Provider":0.5,"Back Office":0.5},
                p_fd_insuff=p_fd_insuff, max_fd_loops=max_fd_loops, fd_loop_delay=fd_loop_delay,
                p_nurse_insuff=p_nurse_insuff, max_nurse_loops=max_nurse_loops,
                p_provider_insuff=p_provider_insuff, max_provider_loops=max_provider_loops, provider_loop_delay=provider_loop_delay,
                p_backoffice_insuff=p_backoffice_insuff, max_backoffice_loops=max_backoffice_loops, backoffice_loop_delay=backoffice_loop_delay,
                p_protocol=p_protocol,
                route_matrix=route
            )
            st.session_state.design_saved = True
            st.success("Configuration saved.")

    if st.session_state.design_saved:
        st.button("Continue →", on_click=go_next, type="primary", use_container_width=True)
    else:
        st.info("Click **Save** to enable Continue.")
        st.button("Continue →", disabled=True, use_container_width=True)

# -------- STEP 2: RUN & RESULTS --------
elif st.session_state.wizard_step == 2:
    st.subheader("Step 2 — Run & Results")
    st.button("← Back to Design", on_click=go_back)

    if not st.session_state["design"]:
        st.info("Use **Continue** on Step 1 first.")
        st.stop()

    # Process diagram (always collapsed but openable)
    dot = build_process_graph(st.session_state["design"])
    with st.expander("📋 Process Preview (click to expand)", expanded=False):
        st.caption("Live view of staffing, routing, nurse protocol, and loop settings based on your saved design.")
        st.graphviz_chart(dot, use_container_width=False)

    seed = st.number_input("Random seed", 0, 999999, 42, 1, "%d", help="Seed for reproducibility.")
    num_replications = st.number_input("Number of replications", 1, 1000, 30, 1, "%d", 
                                       help="Number of independent simulation runs to average over.")
    run = st.button("Run Simulation", type="primary", use_container_width=True)

    if run:
        # mark as ran → collapse diagram on re-render
        st.session_state.ran = True

        p = st.session_state["design"]
        
        # Determine active roles
        active_roles_caps = [
            ("Provider",    p["provider_cap"]),
            ("Front Desk",  p["frontdesk_cap"]),
            ("Nurse",       p["nurse_cap"]),
            ("Back Office", p["backoffice_cap"]),
        ]
        active_roles = [r for r, cap in active_roles_caps if cap > 0]
        
        # Run replications with progress bar
        all_metrics = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for rep in range(num_replications):
            status_text.text(f"Running replication {rep + 1} of {num_replications}...")
            metrics = run_single_replication(p, seed + rep)
            all_metrics.append(metrics)
            progress_bar.progress((rep + 1) / num_replications)
        
        status_text.text(f"Completed {num_replications} replications!")
        
        # Aggregate results
        agg_results = aggregate_replications(p, all_metrics, active_roles)
        
        flow_df = agg_results["flow_df"]
        time_at_role_df = agg_results["time_at_role_df"]
        queue_df = agg_results["queue_df"]
        rework_overview_df = agg_results["rework_overview_df"]
        loop_origin_df = agg_results["loop_origin_df"]
        throughput_full_df = agg_results["throughput_full_df"]
        util_df = agg_results["util_df"]
        summary_df = agg_results["summary_df"]
        
        # Build events dataframe with columns for each replication
        # Create a combined dataframe where each replication is a separate column set
        all_events_data = []
        for rep_idx, metrics in enumerate(all_metrics):
            for t, name, step, note, arr in metrics.events:
                all_events_data.append({
                    "Replication": rep_idx + 1,
                    "Time (min)": float(t),
                    "Task": name,
                    "Step": step,
                    "Step label": pretty_step(step),
                    "Note": note,
                    "Arrival time (min)": (float(arr) if arr is not None else None),
                    "Day": int(t // DAY_MIN)
                })
        events_df = pd.DataFrame(all_events_data)
        
        # ── RENDER ────────────────────────────────────────────────────────────
        st.markdown(f"## 📊 Simulation Results")
        st.caption(f"Averaged over {num_replications} independent replications")
        
        st.markdown("---")
        
        # ============================================================
        # SECTION 1: SYSTEM PERFORMANCE
        # ============================================================
        st.markdown("## 📈 System Performance")
        st.caption("How well is the clinic handling incoming work?")
        
        # Daily Throughput
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("### Daily Throughput Trend")
        with col2:
            help_icon(
                "**Calculation:** Counts the number of tasks completed each day across all replications. "
                "The line shows the mean daily completions, and the shaded area represents ±1 standard deviation. "
                "\n\n**Interpretation:** A declining trend suggests the system is falling behind; "
                "an increasing or stable trend indicates the system is keeping up with demand."
            )
        fig_throughput = plot_daily_throughput(all_metrics, p)
        st.pyplot(fig_throughput, use_container_width=False)
        plt.close(fig_throughput)
        
        st.markdown("")
        
        # Queue Length Over Time
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("### Queue Length Over Time")
        with col2:
            help_icon(
                "**Calculation:** Tracks the number of tasks waiting in each role's queue at every minute of the simulation. "
                "Shows mean ± standard deviation across replications. "
                "\n\n**Interpretation:** Persistent high queues indicate bottlenecks. "
                "Growing queues suggest the system cannot keep up with arrivals. "
                "Different colored lines represent different roles."
            )
        fig_queue = plot_queue_over_time(all_metrics, p, active_roles)
        st.pyplot(fig_queue, use_container_width=False)
        plt.close(fig_queue)
        
        st.markdown("---")
        
        # ============================================================
        # SECTION 2: BURNOUT & WORKLOAD INDICATORS
        # ============================================================
        st.markdown("## 🔥 Burnout & Workload Indicators")
        st.caption("Which roles are at risk of being overwhelmed?")
        
        # Utilization Heatmap
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("### Utilization Heatmap")
        with col2:
            help_icon(
                "**Calculation:** For each hour of the workday, calculates what percentage of available time "
                "each role spends actively working on tasks (service time ÷ capacity × available minutes). "
                "Averaged across all replications and all days. "
                "\n\n**Interpretation:** "
                "🔴 Red (>80%): High burnout risk — constantly busy with little breathing room. "
                "🟡 Yellow (50-80%): Moderate load. "
                "🟢 Green (<50%): Underutilized. "
                "\n\nConsistent red zones indicate chronic overwork and high burnout risk."
            )
        fig_heatmap = plot_utilization_heatmap(all_metrics, p, active_roles)
        st.pyplot(fig_heatmap, use_container_width=False)
        plt.close(fig_heatmap)
        
        st.markdown("")
        
        # Work vs Wait
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("### Work vs Wait Time by Role")
        with col2:
            help_icon(
                "**Calculation:** For each completed task, calculates the average time spent: "
                "(1) actively working (green), and (2) waiting for resources to become available (orange). "
                "Averaged across all tasks and replications. "
                "\n\n**Interpretation:** "
                "High work time = exhaustion from constant activity. "
                "High wait time = frustration from delays and idle time. "
                "Both contribute to burnout. Efficiency % shows work/(work+wait)."
            )
        fig_work_wait = plot_work_vs_wait(all_metrics, p, active_roles)
        st.pyplot(fig_work_wait, use_container_width=False)
        plt.close(fig_work_wait)
        
        st.markdown("")
        
        # Rework Impact
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("### Rework Impact")
        with col2:
            help_icon(
                "**Calculation:** Separates total time spent into original work (blue) vs. rework loops (red). "
                "Rework time is estimated as: (number of loops) × (50% of service time). "
                "Percentage shows rework time / total time. "
                "\n\n**Interpretation:** "
                "High rework percentages indicate frequent errors, missing information, or poor handoffs. "
                "Rework is especially frustrating and contributes significantly to staff burnout."
            )
        fig_rework = plot_rework_impact(all_metrics, p, active_roles)
        st.pyplot(fig_rework, use_container_width=False)
        plt.close(fig_rework)
        
        st.markdown("---")

        # ---- Download run log ----
        st.markdown("## 💾 Download Data")
        
        with st.spinner("Producing run log for download..."):
            runlog_pkg = _runlog_workbook(events_df, engine=_excel_engine())
        
        st.download_button(
            "Download Run Log (Excel)",
            data=runlog_pkg["bytes"],
            file_name=f"RunLog_{num_replications}reps.{runlog_pkg['ext']}",
            mime=runlog_pkg["mime"],
            use_container_width=True,
            type="primary"
        )
