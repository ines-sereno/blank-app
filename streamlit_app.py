from typing import Dict, List
import streamlit as st
import simpy
import random
import numpy as np
import math
import pandas as pd
from io import BytesIO
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, Patch

# =============================
# Utilities
# =============================
def exp_time(mean):
    return 0.0 if mean <= 0 else random.expovariate(1.0/mean)

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

# =============================
# Work schedule helpers
# =============================
MIN_PER_HOUR = 60
DAY_MIN = 24 * MIN_PER_HOUR

def is_open(t_min, open_minutes):
    return (t_min % DAY_MIN) < open_minutes

def minutes_until_close(t_min, open_minutes):
    return max(0.0, open_minutes - (t_min % DAY_MIN))

def minutes_until_open(t_min, open_minutes):
    t_mod = t_min % DAY_MIN
    return 0.0 if t_mod < open_minutes else DAY_MIN - t_mod

def effective_open_minutes(sim_minutes, open_minutes):
    full_days = int(sim_minutes // DAY_MIN)
    remainder = sim_minutes % DAY_MIN
    return full_days * open_minutes + min(open_minutes, remainder)

# =============================
# Roles / constants
# =============================
ROLES = ["Front Desk", "Nurse", "Providers", "Back Office"]
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
        self.loop_fd_insufficient = 0
        self.loop_nurse_insufficient = 0
        self.loop_provider_insufficient = 0
        self.loop_backoffice_insufficient = 0
        self.events = []
        self.task_arrival_time: Dict[str, float] = {}
        self.task_completion_time: Dict[str, float] = {}

    def log(self, t, name, step, note="", arrival_t=None):
        self.events.append((t, name, step, note, arrival_t if arrival_t is not None else self.task_arrival_time.get(name)))

# =============================
# Step labels
# =============================
STEP_LABELS = {
    "ARRIVE": "Task arrived", "FD_QUEUE": "Front Desk: queued", "FD_DONE": "Front Desk: completed",
    "FD_INSUFF": "Front Desk: missing info", "FD_RETRY_QUEUE": "Front Desk: re-queued (info)",
    "FD_RETRY_DONE": "Front Desk: re-done (info)", "NU_QUEUE": "Nurse: queued", "NU_DONE": "Nurse: completed",
    "NU_INSUFF": "Nurse: missing info", "NU_RECHECK_QUEUE": "Nurse: re-check queued",
    "NU_RECHECK_DONE": "Nurse: re-check completed", "PR_QUEUE": "Providers: queued", "PR_DONE": "Providers: completed",
    "PR_INSUFF": "Providers: rework needed", "PR_RECHECK_QUEUE": "Providers: recheck queued",
    "PR_RECHECK_DONE": "Providers: recheck done", "BO_QUEUE": "Back Office: queued", "BO_DONE": "Back Office: completed",
    "BO_INSUFF": "Back Office: rework needed", "BO_RECHECK_QUEUE": "Back Office: recheck queued",
    "BO_RECHECK_DONE": "Back Office: recheck done", "DONE": "Task resolved"
}

def pretty_step(code):
    return STEP_LABELS.get(code, code)

# =============================
# Availability schedule generator
# =============================
def generate_availability_schedule(sim_minutes: int, role: str, minutes_per_hour: int, seed_offset: int = 0) -> set:
    if minutes_per_hour >= 60:
        return set(range(int(sim_minutes)))
    if minutes_per_hour <= 0:
        return set()
    
    local_random = random.Random(hash(role) + seed_offset)
    available_minutes = set()
    total_hours = int(np.ceil(sim_minutes / 60.0))
    
    for hour in range(total_hours):
        hour_start = hour * 60
        hour_end = min((hour + 1) * 60, sim_minutes)
        hour_length = hour_end - hour_start
        available_in_hour = min(minutes_per_hour, hour_length)
        all_minutes_in_hour = list(range(hour_start, hour_end))
        selected = local_random.sample(all_minutes_in_hour, available_in_hour)
        available_minutes.update(selected)
    
    return available_minutes

# =============================
# System
# =============================
class CHCSystem:
    def __init__(self, env, params, metrics, seed_offset=0):
        self.env = env
        self.p = params
        self.m = metrics

        self.fd_cap = params["frontdesk_cap"]
        self.nu_cap = params["nurse_cap"]
        self.pr_cap = params["provider_cap"]
        self.bo_cap = params["backoffice_cap"]

        self.frontdesk = simpy.Resource(env, capacity=self.fd_cap) if self.fd_cap > 0 else None
        self.nurse = simpy.Resource(env, capacity=self.nu_cap) if self.nu_cap > 0 else None
        self.provider = simpy.Resource(env, capacity=self.pr_cap) if self.pr_cap > 0 else None
        self.backoffice = simpy.Resource(env, capacity=self.bo_cap) if self.bo_cap > 0 else None

        self.role_to_res = {
            "Front Desk": self.frontdesk, "Nurse": self.nurse,
            "Providers": self.provider, "Back Office": self.backoffice
        }
        
        avail_params = params.get("availability_per_hour", {"Front Desk": 60, "Nurse": 60, "Providers": 60, "Back Office": 60})
        self.availability = {
            role: generate_availability_schedule(params["sim_minutes"], role, avail_params.get(role, 60), seed_offset)
            for role in ROLES
        }

    def scheduled_service(self, resource, role_account, mean_time, role_for_dist=None):
        if resource is None or mean_time <= 1e-12:
            return
        if role_for_dist is None:
            role_for_dist = role_account

        remaining = draw_service_time(role_for_dist, mean_time, self.p["dist_role"], self.p["cv_speed"])
        remaining += max(0.0, self.p["emr_overhead"].get(role_account, 0.0))

        open_minutes = self.p["open_minutes"]
        available_set = self.availability.get(role_account, set())

        while remaining > 1e-9:
            current_min = int(self.env.now)
            
            if not is_open(self.env.now, open_minutes):
                yield self.env.timeout(minutes_until_open(self.env.now, open_minutes))
                continue
            
            if len(available_set) > 0 and current_min not in available_set:
                yield self.env.timeout(1)
                continue
            
            window = minutes_until_close(self.env.now, open_minutes)
            
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
    if role not in ROLES:
        return DONE

    res = s.role_to_res[role]

    if role == "Front Desk":
        if res is not None:
            s.m.log(env.now, task_id, "FD_QUEUE", "")
            yield from s.scheduled_service(res, "Front Desk", s.p["svc_frontdesk"])
            s.m.log(env.now, task_id, "FD_DONE", "")
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

    elif role == "Providers":
        if res is not None:
            s.m.log(env.now, task_id, "PR_QUEUE", "")
            yield from s.scheduled_service(res, "Providers", s.p["svc_provider"])
            s.m.log(env.now, task_id, "PR_DONE", "")
            provider_loops = 0
            while (provider_loops < s.p["max_provider_loops"]) and (random.random() < s.p["p_provider_insuff"]):
                provider_loops += 1
                s.m.loop_provider_insufficient += 1
                s.m.log(env.now, task_id, "PR_INSUFF", f"Providers rework loop #{provider_loops}")
                yield env.timeout(s.p["provider_loop_delay"])
                s.m.log(env.now, task_id, "PR_RECHECK_QUEUE", f"Loop #{provider_loops}")
                yield from s.scheduled_service(res, "Providers", max(0.0, 0.5 * s.p["svc_provider"]))
                s.m.log(env.now, task_id, "PR_RECHECK_DONE", f"Loop #{provider_loops}")

    elif role == "Back Office":
        if res is not None:
            s.m.log(env.now, task_id, "BO_QUEUE", "")
            yield from s.scheduled_service(res, "Back Office", s.p["svc_backoffice"])
            s.m.log(env.now, task_id, "BO_DONE", "")
            bo_loops = 0
            while (bo_loops < s.p["max_backoffice_loops"]) and (random.random() < s.p["p_backoffice_insuff"]):
                bo_loops += 1
                s.m.loop_backoffice_insufficient += 1
                s.m.log(env.now, task_id, "BO_INSUFF", f"Back Office rework loop #{bo_loops}")
                yield env.timeout(s.p["backoffice_loop_delay"])
                s.m.log(env.now, task_id, "BO_RECHECK_QUEUE", f"Loop #{bo_loops}")
                yield from s.scheduled_service(res, "Back Office", max(0.0, 0.5 * s.p["svc_backoffice"]))
                s.m.log(env.now, task_id, "BO_RECHECK_DONE", f"Loop #{bo_loops}")

    row = s.p["route_matrix"].get(role, {DONE: 1.0})
    nxt = sample_next_role(row)
    return nxt

def task_lifecycle(env, task_id: str, s: CHCSystem, initial_role: str):
    s.m.task_arrival_time[task_id] = env.now
    s.m.arrivals_total += 1
    s.m.arrivals_by_role[initial_role] += 1
    s.m.log(env.now, task_id, "ARRIVE", f"Arrived at {initial_role}", arrival_t=env.now)

    role = initial_role
    for _ in range(60):
        nxt = yield from handle_role(env, task_id, s, role)
        if nxt == DONE:
            s.m.completed += 1
            s.m.task_completion_time[task_id] = env.now
            s.m.log(env.now, task_id, "DONE", "Task completed")
            return
        role = nxt

    s.m.completed += 1
    s.m.task_completion_time[task_id] = env.now
    s.m.log(env.now, task_id, "DONE", "Max handoffs reached – forced completion")

def arrival_process_for_role(env, s: CHCSystem, role_name: str, rate_per_hour: int):
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
# Run single replication
# =============================
def run_single_replication(p: Dict, seed: int) -> Metrics:
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
# Burnout calculation (MBI + JD-R)
# =============================
def calculate_burnout(all_metrics: List[Metrics], p: Dict, active_roles: List[str]) -> Dict:
    """
    Refined burnout model with non-overlapping dimensions and non-linear effects:
      - EE: Utilization (with threshold effects) + AvailabilityStress
      - DP: ReworkPct + TaskSwitching (from queue volatility)
      - RA: SameDayCompletion + TaskThroughput
    Each subscale maps to 0–100, then Overall uses user-defined weights.
    """
    rank_to_weight = {1: 0.5, 2: 0.3, 3: 0.2}
    burnout_weights = p.get("burnout_weights", {"ee_rank": 1, "dp_rank": 2, "ra_rank": 3})
    w_ee = rank_to_weight[burnout_weights["ee_rank"]]
    w_dp = rank_to_weight[burnout_weights["dp_rank"]]
    w_ra = rank_to_weight[burnout_weights["ra_rank"]]
    
    burnout_scores = {}

    open_time_available = effective_open_minutes(p["sim_minutes"], p["open_minutes"])
    num_days = max(1, p["sim_minutes"] / DAY_MIN)

    for role in active_roles:
        capacity = {
            "Front Desk": p["frontdesk_cap"],
            "Nurse": p["nurse_cap"],
            "Providers": p["provider_cap"],
            "Back Office": p["backoffice_cap"]
        }[role]
        if capacity == 0:
            continue

        util_list = []
        rework_pct_list = []
        queue_volatility_list = []
        completion_rate_list = []
        throughput_rate_list = []

        for metrics in all_metrics:
            # Utilization (0–1)
            total_service = metrics.service_time_sum[role]
            denom = capacity * open_time_available
            util = total_service / max(1, denom)
            util_list.append(min(1.0, util))

            # ReworkPct (0–1)
            loop_counts = {
                "Front Desk": metrics.loop_fd_insufficient,
                "Nurse": metrics.loop_nurse_insufficient,
                "Providers": metrics.loop_provider_insufficient,
                "Back Office": metrics.loop_backoffice_insufficient
            }
            loops = loop_counts.get(role, 0)
            svc_time = {
                "Front Desk": p["svc_frontdesk"],
                "Nurse": p["svc_nurse"],
                "Providers": p["svc_provider"],
                "Back Office": p["svc_backoffice"]
            }[role]
            estimated_rework = loops * max(0.0, svc_time) * 0.5
            rework_pct = (estimated_rework / max(1, total_service)) if total_service > 0 else 0.0
            rework_pct_list.append(min(1.0, rework_pct))

            # Queue Volatility (0–1)
            queue_lengths = metrics.queues[role]
            if len(queue_lengths) > 1:
                q_mean = np.mean(queue_lengths)
                q_std = np.std(queue_lengths)
                q_cv = (q_std / max(1e-6, q_mean)) if q_mean > 0 else 0.0
                queue_volatility_list.append(min(1.0, q_cv))
            else:
                queue_volatility_list.append(0.0)

            # Same-day completion rate (0–1)
            done_ids = set(metrics.task_completion_time.keys())
            if len(done_ids) > 0:
                same_day = sum(
                    1 for k in done_ids
                    if int(metrics.task_arrival_time.get(k, 0) // DAY_MIN) ==
                       int(metrics.task_completion_time[k] // DAY_MIN)
                )
                completion_rate_list.append(same_day / len(done_ids))
            else:
                completion_rate_list.append(0.0)

            # Throughput rate (tasks/day)
            tasks_completed = len(done_ids)
            throughput_rate_list.append(tasks_completed / num_days)

        # Average metrics across replications
        avg_util = float(np.mean(util_list)) if util_list else 0.0
        avg_rework = float(np.mean(rework_pct_list)) if rework_pct_list else 0.0
        avg_queue_volatility = float(np.mean(queue_volatility_list)) if queue_volatility_list else 0.0
        avg_completion_rate = float(np.mean(completion_rate_list)) if completion_rate_list else 0.0
        avg_throughput = float(np.mean(throughput_rate_list)) if throughput_rate_list else 0.0

        # Availability stress (0–1)
        avail_minutes = p.get("availability_per_hour", {}).get(role, 60)
        avail_stress = (60 - float(avail_minutes)) / 60.0
        avail_stress = min(max(avail_stress, 0.0), 1.0)

        # Non-linear transformations
        def transform_utilization(u):
            if u <= 0.75:
                return u / 0.75 * 0.5
            else:
                excess = (u - 0.75) / 0.25
                return 0.5 + 0.5 * (np.exp(2 * excess) - 1) / (np.exp(2) - 1)
        
        util_transformed = transform_utilization(avg_util)
        rework_transformed = avg_rework ** 1.5
        volatility_transformed = np.sqrt(avg_queue_volatility)
        incompletion = 1.0 - avg_completion_rate
        incompletion_transformed = incompletion ** 0.7
        expected_throughput = p["arrivals_per_hour_by_role"].get(role, 1) * open_time_available / 60.0 / num_days
        throughput_ratio = avg_throughput / max(1e-6, expected_throughput)
        throughput_deficit = max(0.0, 1.0 - throughput_ratio)
        throughput_deficit = min(1.0, throughput_deficit)

        # Burnout subscales
        EE = 100.0 * (0.75 * util_transformed + 0.25 * avail_stress)
        DP = 100.0 * (0.60 * rework_transformed + 0.40 * volatility_transformed)
        RA = 100.0 * (0.55 * incompletion_transformed + 0.45 * throughput_deficit)

        burnout_score = w_ee * EE + w_dp * DP + w_ra * RA

        burnout_scores[role] = {
            "overall": float(burnout_score),
            "emotional_exhaustion": float(EE),
            "depersonalization": float(DP),
            "reduced_accomplishment": float(RA)
        }

    clinic_burnout = np.mean([v["overall"] for v in burnout_scores.values()]) if burnout_scores else 0.0
    return {"by_role": burnout_scores, "overall_clinic": float(clinic_burnout)}

# =============================
# Visualization functions
# =============================
def plot_utilization_by_role(all_metrics: List[Metrics], p: Dict, active_roles: List[str]):
    """
    Bar chart showing utilization by role (mean ± SD across replications).
    """
    fig, ax = plt.subplots(figsize=(6, 3), dpi=80)
    
    open_time_available = effective_open_minutes(p["sim_minutes"], p["open_minutes"])
    
    util_lists = {r: [] for r in active_roles}
    
    for metrics in all_metrics:
        for role in active_roles:
            capacity = {
                "Front Desk": p["frontdesk_cap"],
                "Nurse": p["nurse_cap"],
                "Providers": p["provider_cap"],
                "Back Office": p["backoffice_cap"]
            }[role]
            
            if capacity > 0:
                total_service = metrics.service_time_sum[role]
                avail_minutes = p.get("availability_per_hour", {}).get(role, 60)
                available_capacity = capacity * open_time_available * (avail_minutes / 60.0)
                util = min(1.0, total_service / max(1, available_capacity))
                util_lists[role].append(util)
    
    means = [np.mean(util_lists[r]) * 100 for r in active_roles]
    stds = [np.std(util_lists[r]) * 100 for r in active_roles]
    
    colors = []
    for mean_util in means:
        if mean_util < 50:
            colors.append('#2ecc71')
        elif mean_util < 75:
            colors.append('#f39c12')
        elif mean_util < 90:
            colors.append('#e67e22')
        else:
            colors.append('#e74c3c')
    
    x = np.arange(len(active_roles))
    bars = ax.bar(x, means, color=colors, alpha=0.8, width=0.6)
    ax.errorbar(x, means, yerr=stds, fmt='none', ecolor='black', capsize=5, alpha=0.6)
    
    ax.axhline(y=75, color='orange', linestyle='--', alpha=0.4, linewidth=1.5, label='75% threshold')
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='90% critical')
    
    ax.set_xlabel('Role', fontsize=10)
    ax.set_ylabel('Utilization (%)', fontsize=10)
    ax.set_title('Staff Utilization by Role', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(active_roles, fontsize=9, rotation=15, ha='right')
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{mean:.0f}%\n±{std:.0f}%',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_queue_over_time(all_metrics: List[Metrics], p: Dict, active_roles: List[str]):
    fig, ax = plt.subplots(figsize=(6, 3), dpi=40)
    colors = {'Front Desk': '#1f77b4', 'Nurse': '#ff7f0e', 'Providers': '#2ca02c', 'Back Office': '#d62728'}
    
    num_days = max(1, int(p["sim_minutes"] // DAY_MIN))
    open_minutes = p["open_minutes"]
    
    end_of_day_queues = {role: [] for role in active_roles}
    
    for role in active_roles:
        daily_queues = []
        for metrics in all_metrics:
            role_daily = []
            for day in range(num_days):
                end_of_open_time = day * DAY_MIN + open_minutes
                
                if len(metrics.time_stamps) > 0:
                    closest_idx = min(range(len(metrics.time_stamps)), 
                                    key=lambda i: abs(metrics.time_stamps[i] - end_of_open_time))
                    role_daily.append(metrics.queues[role][closest_idx])
                else:
                    role_daily.append(0)
            daily_queues.append(role_daily)
        
        if daily_queues:
            daily_array = np.array(daily_queues)
            mean_daily = np.mean(daily_array, axis=0)
            std_daily = np.std(daily_array, axis=0)
            end_of_day_queues[role] = (mean_daily, std_daily)
    
    x = np.arange(num_days)
    width = 0.8 / len(active_roles)
    
    for i, role in enumerate(active_roles):
        mean_daily, std_daily = end_of_day_queues[role]
        offset = (i - len(active_roles)/2 + 0.5) * width
        ax.bar(x + offset, mean_daily, width, label=role, color=colors.get(role, '#333333'), 
               alpha=0.8, yerr=std_daily, capsize=3)
    
    for day in range(1, num_days):
        ax.axvline(x=day - 0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Operational Day', fontsize=10)
    ax.set_ylabel('Queue Length (end of day)', fontsize=10)
    ax.set_title('End-of-Day Queue Backlog', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}' for i in range(num_days)])
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    return fig

def plot_daily_throughput(all_metrics: List[Metrics], p: Dict, active_roles: List[str]):
    num_days = max(1, int(p["sim_minutes"] // DAY_MIN))
    
    daily_data = []
    
    for d in range(num_days):
        start_t = d * DAY_MIN
        end_t = start_t + p["open_minutes"]
        
        day_totals = []
        for metrics in all_metrics:
            completed = sum(1 for ct in metrics.task_completion_time.values() if start_t <= ct < end_t)
            day_totals.append(completed)
        
        mean_total = np.mean(day_totals)
        std_total = np.std(day_totals, ddof=1) if len(day_totals) > 1 else 0.0
        
        daily_data.append({
            'Day': f'Day {d+1}',
            'Tasks Completed': f'{mean_total:.1f} ± {std_total:.1f}'
        })
    
    df = pd.DataFrame(daily_data)
    
    return df

def plot_rework_impact(all_metrics: List[Metrics], p: Dict, active_roles: List[str]):
    fig, ax = plt.subplots(figsize=(6, 3), dpi=80)
    original_time = {r: [] for r in active_roles}
    rework_time = {r: [] for r in active_roles}
    
    for metrics in all_metrics:
        loop_counts = {"Front Desk": metrics.loop_fd_insufficient, "Nurse": metrics.loop_nurse_insufficient,
                      "Providers": metrics.loop_provider_insufficient, "Back Office": metrics.loop_backoffice_insufficient}
        
        for role in active_roles:
            total_time = metrics.service_time_sum[role]
            loops = loop_counts.get(role, 0)
            svc_time = {"Front Desk": p["svc_frontdesk"], "Nurse": p["svc_nurse"],
                       "Providers": p["svc_provider"], "Back Office": p["svc_backoffice"]}[role]
            estimated_rework = loops * svc_time * 0.5
            estimated_original = total_time - estimated_rework
            original_time[role].append(max(0, estimated_original))
            rework_time[role].append(estimated_rework)
    
    original_means = [np.mean(original_time[r]) for r in active_roles]
    rework_means = [np.mean(rework_time[r]) for r in active_roles]
    
    original_stds = [np.std(original_time[r]) for r in active_roles]
    rework_stds = [np.std(rework_time[r]) for r in active_roles]
    
    x = np.arange(len(active_roles))
    width = 0.6
    
    bars1 = ax.bar(x, original_means, width, label='Original Work', color='#3498db')
    bars2 = ax.bar(x, rework_means, width, bottom=original_means, label='Rework', color='#e74c3c')
    
    total_means = [original_means[i] + rework_means[i] for i in range(len(active_roles))]
    total_stds = [np.sqrt(original_stds[i]**2 + rework_stds[i]**2) for i in range(len(active_roles))]
    
    ax.errorbar(x, total_means, yerr=total_stds, fmt='none', ecolor='black', capsize=5, alpha=0.6, linewidth=1.5)
    
    ax.set_xlabel('Role', fontsize=10)
    ax.set_ylabel('Time (minutes)', fontsize=10)
    ax.set_title('Rework Impact', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(active_roles, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (orig, rew) in enumerate(zip(original_means, rework_means)):
        total = orig + rew
        if total > 0 and rew > 0:
            pct = 100 * rew / total
            label_height = total + total_stds[i] + 20
            ax.text(i, label_height, f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig

def plot_burnout_scores(burnout_data: Dict, active_roles: List[str]):
    """
    Bar chart showing burnout scores by role + overall clinic with error bars.
    """
    fig, ax = plt.subplots(figsize=(6, 3), dpi=40)
    
    roles_plot = active_roles + ["Overall Clinic"]
    
    role_scores = [burnout_data["by_role"][r]["overall"] for r in active_roles]
    role_scores.append(burnout_data["overall_clinic"])
    
    stds = []
    for role in active_roles:
        subscales = [
            burnout_data["by_role"][role]["emotional_exhaustion"],
            burnout_data["by_role"][role]["depersonalization"],
            burnout_data["by_role"][role]["reduced_accomplishment"]
        ]
        stds.append(np.std(subscales))
    
    stds.append(np.std(role_scores[:-1]))
    
    colors = []
    for score in role_scores:
        if score < 25:
            colors.append('#2ecc71')
        elif score < 50:
            colors.append('#f39c12')
        elif score < 75:
            colors.append('#e67e22')
        else:
            colors.append('#e74c3c')
    
    x = np.arange(len(roles_plot))
    bars = ax.bar(x, role_scores, color=colors, width=0.6)
    
    ax.errorbar(x, role_scores, yerr=stds, fmt='none', ecolor='black', capsize=5, alpha=0.6, linewidth=1.5)
    
    ax.set_xlabel('Role', fontsize=10)
    ax.set_ylabel('Burnout Score (0-100)', fontsize=10)
    ax.set_title('Burnout Index by Role', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(roles_plot, fontsize=9, rotation=15, ha='right')
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, score, std) in enumerate(zip(bars, role_scores, stds)):
        height = bar.get_height()
        label_height = height + std + 2
        ax.text(bar.get_x() + bar.get_width()/2., label_height,
                f'{score:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Low (0-25)'),
        Patch(facecolor='#f39c12', label='Moderate (25-50)'),
        Patch(facecolor='#e67e22', label='High (50-75)'),
        Patch(facecolor='#e74c3c', label='Severe (75-100)')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=5)
    
    plt.tight_layout()
    return fig

def plot_overtime_needed(all_metrics: List[Metrics], p: Dict, active_roles: List[str]):
    """
    Bar chart showing additional hours per day needed to complete all tasks.
    """
    fig, ax = plt.subplots(figsize=(6, 3), dpi=80)
    
    num_days = max(1, p["sim_minutes"] / DAY_MIN)
    open_hours_per_day = p["open_minutes"] / 60.0
    
    overtime_lists = {r: [] for r in active_roles}
    
    for metrics in all_metrics:
        for role in active_roles:
            capacity = {
                "Front Desk": p["frontdesk_cap"],
                "Nurse": p["nurse_cap"],
                "Providers": p["provider_cap"],
                "Back Office": p["backoffice_cap"]
            }[role]
            
            if capacity > 0:
                total_work_needed = metrics.service_time_sum[role]
                avail_minutes_per_hour = p.get("availability_per_hour", {}).get(role, 60)
                capacity_per_day = capacity * p["open_minutes"] * (avail_minutes_per_hour / 60.0)
                total_capacity_available = capacity_per_day * num_days
                overtime_minutes = max(0, total_work_needed - total_capacity_available)
                
                if overtime_minutes > 0:
                    overtime_hours_total = overtime_minutes / 60.0
                    overtime_hours_per_day = overtime_hours_total / num_days
                    overtime_hours_per_person = overtime_hours_per_day / capacity
                else:
                    overtime_hours_per_person = 0.0
                
                overtime_lists[role].append(overtime_hours_per_person)
    
    means = [np.mean(overtime_lists[r]) for r in active_roles]
    stds = [np.std(overtime_lists[r]) for r in active_roles]
    
    colors = []
    for mean_ot in means:
        if mean_ot < 0.5:
            colors.append('#2ecc71')
        elif mean_ot < 1.0:
            colors.append('#f39c12')
        elif mean_ot < 2.0:
            colors.append('#e67e22')
        else:
            colors.append('#e74c3c')
    
    x = np.arange(len(active_roles))
    bars = ax.bar(x, means, color=colors, alpha=0.8, width=0.6)
    
    ax.errorbar(x, means, yerr=stds, fmt='none', ecolor='black', capsize=5, alpha=0.6)
    
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.4, linewidth=1.5, label='0.5 hr/day')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='1.0 hr/day')
    
    ax.set_xlabel('Role', fontsize=10)
    ax.set_ylabel('Additional Hours per Day per Person', fontsize=10)
    ax.set_title('Overtime Needed to Clear Backlog', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(active_roles, fontsize=9, rotation=15, ha='right')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        if height > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2., height + max(0.05, std),
                    f'{mean:.1f}h\n±{std:.1f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., 0.05,
                    '0.0h',
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='gray')
    
    plt.tight_layout()
    return fig

def create_kpi_banner(all_metrics: List[Metrics], p: Dict, burnout_data: Dict, active_roles: List[str]):
    """
    Create a simple one-line banner showing key metrics.
    """
    turnaround_times = []
    for metrics in all_metrics:
        comp_times = metrics.task_completion_time
        arr_times = metrics.task_arrival_time
        done_ids = set(comp_times.keys())
        
        if len(done_ids) > 0:
            tt = [comp_times[k] - arr_times.get(k, comp_times[k]) for k in done_ids]
            turnaround_times.extend(tt)
    
    avg_turnaround = np.mean(turnaround_times) if turnaround_times else 0.0
    
    all_ee = [burnout_data["by_role"][r]["emotional_exhaustion"] for r in active_roles]
    all_dp = [burnout_data["by_role"][r]["depersonalization"] for r in active_roles]
    all_ra = [burnout_data["by_role"][r]["reduced_accomplishment"] for r in active_roles]
    
    avg_ee = np.mean(all_ee) if all_ee else 0.0
    avg_dp = np.mean(all_dp) if all_dp else 0.0
    avg_ra = np.mean(all_ra) if all_ra else 0.0
    overall_burnout = burnout_data["overall_clinic"]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Avg Turnaround", f"{avg_turnaround:.0f} min", 
                 delta=f"{avg_turnaround/60:.1f} hrs", delta_color="off")
    
    with col2:
        st.metric("Overall Burnout", f"{overall_burnout:.1f}")
    
    with col3:
        st.metric("Emotional Exhaustion", f"{avg_ee:.1f}")
    
    with col4:
        st.metric("Depersonalization", f"{avg_dp:.1f}")
    
    with col5:
        st.metric("Reduced Accomplishment", f"{avg_ra:.1f}")

def help_icon(help_text: str):
    with st.expander("How is this calculated?"):
        st.caption(help_text)

def aggregate_replications(p: Dict, all_metrics: List[Metrics], active_roles: List[str]):
    num_reps = len(all_metrics)
    
    def fmt_mean_std(values):
        m = np.mean(values)
        s = np.std(values, ddof=1) if len(values) > 1 else 0.0
        return f"{m:.1f} ± {s:.1f}"
    
    def fmt_mean_std_pct(values):
        m = np.mean(values)
        s = np.std(values, ddof=1) if len(values) > 1 else 0.0
        return f"{m:.1f}% ± {s:.1f}%"
    
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
    
    q_avg_lists = {r: [] for r in ROLES}
    q_max_lists = {r: [] for r in ROLES}
    
    for metrics in all_metrics:
        for r in ROLES:
            q_avg_lists[r].append(np.mean(metrics.queues[r]) if len(metrics.queues[r]) > 0 else 0.0)
            q_max_lists[r].append(np.max(metrics.queues[r]) if len(metrics.queues[r]) > 0 else 0)
    
    queue_df = pd.DataFrame([
        {"Role": r, "Avg queue length": fmt_mean_std(q_avg_lists[r]), "Max queue length": fmt_mean_std(q_max_lists[r])}
        for r in active_roles
    ])
    
    rework_pct_list = []
    loop_counts_lists = {"Front Desk": [], "Nurse": [], "Providers": [], "Back Office": []}
    
    for metrics in all_metrics:
        rework_tasks = set()
        for t, name, step, note, _arr in metrics.events:
            if step.endswith("INSUFF") or "RECHECK" in step:
                rework_tasks.add(name)
        
        done_ids = set(metrics.task_completion_time.keys())
        rework_pct_list.append(100.0 * len(rework_tasks & done_ids) / max(1, len(done_ids)))
        
        loop_counts_lists["Front Desk"].append(metrics.loop_fd_insufficient)
        loop_counts_lists["Nurse"].append(metrics.loop_nurse_insufficient)
        loop_counts_lists["Providers"].append(metrics.loop_provider_insufficient)
        loop_counts_lists["Back Office"].append(metrics.loop_backoffice_insufficient)
    
    rework_overview_df = pd.DataFrame([
        {"Metric": "% tasks with any rework", "Value": fmt_mean_std_pct(rework_pct_list)}
    ])
    
    total_loops_list = [sum(loop_counts_lists[r][i] for r in ROLES) for i in range(num_reps)]
    loop_origin_df = pd.DataFrame([
        {"Role": r, "Loop Count": fmt_mean_std(loop_counts_lists[r]),
         "Share": fmt_mean_std_pct([100.0 * loop_counts_lists[r][i] / total_loops_list[i] 
                                   if total_loops_list[i] > 0 else 0.0 for i in range(num_reps)])}
        for r in active_roles
    ])
    
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
    
    open_time_available = effective_open_minutes(p["sim_minutes"], p["open_minutes"])
    denom = {
        "Front Desk": max(1, p["frontdesk_cap"]) * open_time_available,
        "Nurse": max(1, p["nurse_cap"]) * open_time_available,
        "Providers": max(1, p["provider_cap"]) * open_time_available,
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
        "flow_df": flow_df, "time_at_role_df": time_at_role_df, "queue_df": queue_df,
        "rework_overview_df": rework_overview_df, "loop_origin_df": loop_origin_df,
        "throughput_full_df": throughput_full_df, "util_df": util_df, "summary_df": summary_df
    }

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="HSyE Burnout Grant - DES Model", layout="wide")
st.title("HSyE Burnout Grant - DES Model for Community Health Centers")
st.caption("By Ines Sereno")

if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 1
if "results" not in st.session_state:
    st.session_state["results"] = None
if "design" not in st.session_state:
    st.session_state["design"] = None
if "design_saved" not in st.session_state:
    st.session_state.design_saved = False
if "ran" not in st.session_state:
    st.session_state.ran = False
if "saved_runs" not in st.session_state:
    st.session_state.saved_runs = []

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

def _excel_engine():
    try:
        import xlsxwriter
        return "xlsxwriter"
    except Exception:
        try:
            import openpyxl
            return "openpyxl"
        except Exception:
            return None

def _runlog_workbook(events_df: pd.DataFrame, engine: str | None = None) -> dict:
    if engine is None:
        engine = _excel_engine()
    
    if engine:
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine=engine) as xw:
            events_df.to_excel(xw, index=False, sheet_name="RunLog")
        return {"bytes": bio.getvalue(), "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "ext": "xlsx"}
    else:
        st.error("Excel export requires xlsxwriter or openpyxl")
        return {"bytes": b"", "mime": "application/octet-stream", "ext": "xlsx"}

# -------- STEP 1: DESIGN --------
if st.session_state.wizard_step == 1:
    st.markdown("### Design Your Clinic")
    
    def route_row_ui(from_role: str, defaults: Dict[str, float], disabled_source: bool = False, 
                     fd_cap_val: int = 0, nu_cap_val: int = 0, pr_cap_val: int = 0, bo_cap_val: int = 0) -> Dict[str, float]:
        current_cap_map = {"Front Desk": fd_cap_val, "Nurse": nu_cap_val, "Providers": pr_cap_val, "Back Office": bo_cap_val}
        st.markdown(f"**{from_role} →**")
        targets = [r for r in ROLES if r != from_role] + [DONE]
        cols = st.columns(len(targets))
        row: Dict[str, float] = {}
        for i, tgt in enumerate(targets):
            tgt_disabled = disabled_source or (tgt in ROLES and current_cap_map[tgt] == 0)
            label_name = "Done" if tgt == DONE else tgt
            key_name = f"r_{from_role}_{'done' if tgt==DONE else label_name.replace(' ','_').lower()}"
            default_val = float(defaults.get(tgt, 0.0))
            with cols[i]:
                val = prob_input(f"to {label_name} ({from_role})", key=key_name, 
                               default=(0.0 if tgt_disabled else default_val), disabled=tgt_disabled)
                if tgt_disabled:
                    val = 0.0
            row[tgt] = val
        return row
                         
    with st.form("design_form", clear_on_submit=False):
        st.markdown("### Simulation horizon & variability")
        sim_days = st.number_input("Days to simulate", 1, 30, _init_ss("sim_days", 5), 1, "%d",
                                   help="Number of clinic operating days to simulate")
        open_hours = st.number_input("Hours open per day", 1, 24, _init_ss("open_hours", 10), 1, "%d",
                                      help="Number of hours the clinic is open each day")

        cv_speed_label = st.select_slider(
            "Task speed variability",
            options=["Very Low", "Low", "Moderate", "High", "Very High"],
            value=_init_ss("cv_speed_label", "Moderate"),
            help="Variation in task completion times"
        )
        
        cv_speed_map = {
            "Very Low": 0.1,
            "Low": 0.2,
            "Moderate": 0.3,
            "High": 0.5,
            "Very High": 0.7
        }
        cv_speed = cv_speed_map[cv_speed_label]
        st.caption(f"(Coefficient of Variation: {cv_speed})")

        seed = st.number_input("Random seed", 0, 999999, _init_ss("seed", 42), 1, "%d", 
                               help="Seed for reproducibility")
        num_replications = st.number_input("Number of replications", 1, 1000, _init_ss("num_replications", 30), 1, "%d", 
                                          help="Number of independent simulation runs")

        st.markdown("### Role Configuration")
        st.caption("Configure staffing, arrivals, and availability for each role")
        
        with st.expander("Front Desk", expanded=True):
            cFD1, cFD2, cFD3 = st.columns(3)
            with cFD1:
                fd_cap_form = st.number_input("Staff on duty", 0, 50, _init_ss("fd_cap", 3), 1, "%d", key="fd_cap_input",
                                                           help="Number of front desk staff")
            with cFD2:
                arr_fd = st.number_input("Arrivals per hour", 0, 500, _init_ss("arr_fd", 5), 1, "%d", disabled=(fd_cap_form==0), key="arr_fd_input",
                                         help="Average number of tasks per hour")
            with cFD3:
                avail_fd = st.number_input("Availability (min/hour)", 0, 60, _init_ss("avail_fd", 45), 1, "%d", disabled=(fd_cap_form==0), key="avail_fd_input",
                                           help="Minutes per hour available for work")
        
        with st.expander("Nurse / MAs", expanded=True):
            cNU1, cNU2, cNU3 = st.columns(3)
            with cNU1:
                nu_cap_form = st.number_input("Staff on duty", 0, 50, _init_ss("nurse_cap", 2), 1, "%d", key="nurse_cap_input",
                                                              help="Number of nurses or medical assistants")
            with cNU2:
                arr_nu = st.number_input("Arrivals per hour", 0, 500, _init_ss("arr_nu", 10), 1, "%d", disabled=(nu_cap_form==0), key="arr_nu_input",
                                         help="Average number of tasks per hour")
            with cNU3:
                avail_nu = st.number_input("Availability (min/hour)", 0, 60, _init_ss("avail_nu", 20), 1, "%d", disabled=(nu_cap_form==0), key="avail_nu_input",
                                           help="Minutes per hour available for work")
        
        with st.expander("Providers", expanded=True):
            cPR1, cPR2, cPR3 = st.columns(3)
            with cPR1:
                pr_cap_form = st.number_input("Staff on duty", 0, 50, _init_ss("provider_cap", 1), 1, "%d", key="provider_cap_input",
                                                                 help="Number of providers")
            with cPR2:
                arr_pr = st.number_input("Arrivals per hour", 0, 500, _init_ss("arr_pr", 3), 1, "%d", disabled=(pr_cap_form==0), key="arr_pr_input",
                                         help="Average number of tasks per hour")
            with cPR3:
                avail_pr = st.number_input("Availability (min/hour)", 0, 60, _init_ss("avail_pr", 30), 1, "%d", disabled=(pr_cap_form==0), key="avail_pr_input",
                                           help="Minutes per hour available for work")
        
        with st.expander("Back Office", expanded=True):
            cBO1, cBO2, cBO3 = st.columns(3)
            with cBO1:
                bo_cap_form = st.number_input("Staff on duty", 0, 50, _init_ss("backoffice_cap", 1), 1, "%d", key="bo_cap_input",
                                                           help="Number of back office staff")
            with cBO2:
                arr_bo = st.number_input("Arrivals per hour", 0, 500, _init_ss("arr_bo", 2), 1, "%d", disabled=(bo_cap_form==0), key="arr_bo_input",
                                         help="Average number of tasks per hour")
            with cBO3:
                avail_bo = st.number_input("Availability (min/hour)", 0, 60, _init_ss("avail_bo", 45), 1, "%d", disabled=(bo_cap_form==0), key="avail_bo_input",
                                           help="Minutes per hour available for work")

        with st.expander("Advanced Settings – Service times, loops & routing", expanded=False):
            
            st.markdown("### Burnout Priority Weights")
            st.caption("Rank the burnout dimensions by importance (1 = most important, 3 = least important)")
            cB1, cB2, cB3 = st.columns(3)
            with cB1:
                ee_rank = st.selectbox("Emotional Exhaustion", [1, 2, 3], index=0, key="ee_rank",
                                      help="Rank importance of Emotional Exhaustion")
            with cB2:
                dp_rank = st.selectbox("Depersonalization", [1, 2, 3], index=1, key="dp_rank",
                                      help="Rank importance of Depersonalization")
            with cB3:
                ra_rank = st.selectbox("Reduced Accomplishment", [1, 2, 3], index=2, key="ra_rank",
                                      help="Rank importance of Reduced Accomplishment")

            ranks = [ee_rank, dp_rank, ra_rank]
            if len(set(ranks)) != 3:
                st.error("Each dimension must have a unique rank (1, 2, or 3)")
            else:
                rank_to_weight = {1: 0.5, 2: 0.3, 3: 0.2}
                st.success(f"Weights will be: EE={rank_to_weight[ee_rank]:.1f}, DP={rank_to_weight[dp_rank]:.1f}, RA={rank_to_weight[ra_rank]:.1f}")
            
            st.markdown("---")
            
            with st.expander("Front Desk", expanded=False):
                st.markdown("**Service Time**")
                svc_frontdesk = st.slider("Mean service time (minutes)", 0.0, 30.0, _init_ss("svc_frontdesk", 3.0), 0.5, disabled=(fd_cap_form==0),
                                          help="Average time to complete a task")
                
                st.markdown("**Rework Loops**")
                cFDL1, cFDL2, cFDL3 = st.columns(3)
                with cFDL1:
                    p_fd_insuff = st.slider("Probability of missing info", 0.0, 1.0, _init_ss("p_fd_insuff", 0.15), 0.01, disabled=(fd_cap_form==0), key="fd_p_insuff")
                with cFDL2:
                    max_fd_loops = st.number_input("Max loops", 0, 10, _init_ss("max_fd_loops", 2), 1, "%d", disabled=(fd_cap_form==0), key="fd_max_loops")
                with cFDL3:
                    fd_loop_delay = st.slider("Rework delay (min)", 0.0, 60.0, _init_ss("fd_loop_delay", 5.0), 0.5, disabled=(fd_cap_form==0), key="fd_delay")
                
                st.markdown("**Routing: Where tasks go after Front Desk**")
                fd_route = route_row_ui("Front Desk", {"Nurse": 0.50, "Providers": 0.10, "Back Office": 0.10, DONE: 0.30}, 
                                       disabled_source=(fd_cap_form==0), fd_cap_val=fd_cap_form, nu_cap_val=nu_cap_form, 
                                       pr_cap_val=pr_cap_form, bo_cap_val=bo_cap_form)
            
            with st.expander("Nurse / MAs", expanded=False):
                st.markdown("**Service Times**")
                cNS1, cNS2 = st.columns(2)
                with cNS1:
                    svc_nurse_protocol = st.slider("Protocol service time (minutes)", 0.0, 30.0, _init_ss("svc_nurse_protocol", 2.0), 0.5, disabled=(nu_cap_form==0))
                    p_protocol = st.slider("Probability of using protocol", 0.0, 1.0, _init_ss("p_protocol", 0.40), 0.05, disabled=(nu_cap_form==0))
                with cNS2:
                    svc_nurse = st.slider("Non-protocol service time (minutes)", 0.0, 40.0, _init_ss("svc_nurse", 4.0), 0.5, disabled=(nu_cap_form==0))
                
                st.markdown("**Rework Loops**")
                cNUL1, cNUL2 = st.columns(2)
                with cNUL1:
                    p_nurse_insuff = st.slider("Probability of insufficient info", 0.0, 1.0, _init_ss("p_nurse_insuff", 0.10), 0.01, disabled=(nu_cap_form==0), key="nu_p_insuff")
                with cNUL2:
                    max_nurse_loops = st.number_input("Max loops", 0, 10, _init_ss("max_nurse_loops", 2), 1, "%d", disabled=(nu_cap_form==0), key="nu_max_loops")
                
                st.markdown("**Routing: Where tasks go after Nurse**")
                nu_route = route_row_ui("Nurse", {"Providers": 0.40, "Back Office": 0.20, DONE: 0.40}, 
                                       disabled_source=(nu_cap_form==0), fd_cap_val=fd_cap_form, nu_cap_val=nu_cap_form, 
                                       pr_cap_val=pr_cap_form, bo_cap_val=bo_cap_form)
            
            with st.expander("Providers", expanded=False):
                st.markdown("**Service Time**")
                svc_provider = st.slider("Mean service time (minutes)", 0.0, 60.0, _init_ss("svc_provider", 6.0), 0.5, disabled=(pr_cap_form==0))
                
                st.markdown("**Rework Loops**")
                cPRL1, cPRL2, cPRL3 = st.columns(3)
                with cPRL1:
                    p_provider_insuff = st.slider("Probability of rework needed", 0.0, 1.0, _init_ss("p_provider_insuff", 0.08), 0.01, disabled=(pr_cap_form==0), key="pr_p_insuff")
                with cPRL2:
                    max_provider_loops = st.number_input("Max loops", 0, 10, _init_ss("max_provider_loops", 2), 1, "%d", disabled=(pr_cap_form==0), key="pr_max_loops")
                with cPRL3:
                    provider_loop_delay = st.slider("Rework delay (min)", 0.0, 60.0, _init_ss("provider_loop_delay", 5.0), 0.5, disabled=(pr_cap_form==0), key="pr_delay")
                
                st.markdown("**Routing: Where tasks go after Providers**")
                pr_route = route_row_ui("Providers", {"Back Office": 0.30, DONE: 0.70}, 
                                       disabled_source=(pr_cap_form==0), fd_cap_val=fd_cap_form, nu_cap_val=nu_cap_form, 
                                       pr_cap_val=pr_cap_form, bo_cap_val=bo_cap_form)
            
            with st.expander("Back Office", expanded=False):
                st.markdown("**Service Time**")
                svc_backoffice = st.slider("Mean service time (minutes)", 0.0, 60.0, _init_ss("svc_backoffice", 5.0), 0.5, disabled=(bo_cap_form==0))
                
                st.markdown("**Rework Loops**")
                cBOL1, cBOL2, cBOL3 = st.columns(3)
                with cBOL1:
                    p_backoffice_insuff = st.slider("Probability of rework needed", 0.0, 1.0, _init_ss("p_backoffice_insuff", 0.05), 0.01, disabled=(bo_cap_form==0), key="bo_p_insuff")
                with cBOL2:
                    max_backoffice_loops = st.number_input("Max loops", 0, 10, _init_ss("max_backoffice_loops", 2), 1, "%d", disabled=(bo_cap_form==0), key="bo_max_loops")
                with cBOL3:
                    backoffice_loop_delay = st.slider("Rework delay (min)", 0.0, 60.0, _init_ss("backoffice_loop_delay", 5.0), 0.5, disabled=(bo_cap_form==0), key="bo_delay")
                
                st.markdown("**Routing: Where tasks go after Back Office**")
                bo_route = route_row_ui("Back Office", {"Front Desk": 0.10, "Nurse": 0.10, "Providers": 0.10, DONE: 0.70}, 
                                       disabled_source=(bo_cap_form==0), fd_cap_val=fd_cap_form, nu_cap_val=nu_cap_form, 
                                       pr_cap_val=pr_cap_form, bo_cap_val=bo_cap_form)
            
            route: Dict[str, Dict[str, float]] = {}
            route["Front Desk"] = fd_route
            route["Nurse"] = nu_route
            route["Providers"] = pr_route
            route["Back Office"] = bo_route

        saved = st.form_submit_button("Save", type="primary")

        if saved:
            st.session_state.fd_cap = fd_cap_form
            st.session_state.nurse_cap = nu_cap_form
            st.session_state.provider_cap = pr_cap_form
            st.session_state.bo_cap = bo_cap_form
            
            open_minutes = int(open_hours * MIN_PER_HOUR)
            sim_minutes = int(sim_days * DAY_MIN)

            for r in ROLES:
                if r in route:
                    route[r].pop(r, None)
            for r in ROLES:
                if r in route:
                    for tgt in list(route[r].keys()):
                        if tgt in ROLES and {"Front Desk": fd_cap_form, "Nurse": nu_cap_form, "Providers": pr_cap_form, "Back Office": bo_cap_form}[tgt] == 0:
                            route[r][tgt] = 0.0

            st.session_state["design"] = dict(
                sim_minutes=sim_minutes, open_minutes=open_minutes,
                seed=seed, num_replications=num_replications,
                frontdesk_cap=fd_cap_form, nurse_cap=nu_cap_form,
                provider_cap=pr_cap_form, backoffice_cap=bo_cap_form,
                arrivals_per_hour_by_role={"Front Desk": int(arr_fd), "Nurse": int(arr_nu), 
                                          "Providers": int(arr_pr), "Back Office": int(arr_bo)},
                availability_per_hour={"Front Desk": int(avail_fd), "Nurse": int(avail_nu),
                                      "Providers": int(avail_pr), "Back Office": int(avail_bo)},
                svc_frontdesk=svc_frontdesk, svc_nurse_protocol=svc_nurse_protocol, svc_nurse=svc_nurse,
                svc_provider=svc_provider, svc_backoffice=svc_backoffice,
                dist_role={"Front Desk": "normal", "NurseProtocol": "normal", "Nurse": "exponential",
                          "Providers": "exponential", "Back Office": "exponential"},
                cv_speed=cv_speed,
                emr_overhead={"Front Desk": 0.5, "Nurse": 0.5, "NurseProtocol": 0.5, "Providers": 0.5, "Back Office": 0.5},
                burnout_weights={"ee_rank": ee_rank, "dp_rank": dp_rank, "ra_rank": ra_rank},
                p_fd_insuff=p_fd_insuff, max_fd_loops=max_fd_loops, fd_loop_delay=fd_loop_delay,
                p_nurse_insuff=p_nurse_insuff, max_nurse_loops=max_nurse_loops,
                p_provider_insuff=p_provider_insuff, max_provider_loops=max_provider_loops, provider_loop_delay=provider_loop_delay,
                p_backoffice_insuff=p_backoffice_insuff, max_backoffice_loops=max_backoffice_loops, backoffice_loop_delay=backoffice_loop_delay,
                p_protocol=p_protocol, route_matrix=route
            )
            st.session_state.design_saved = True
            st.success("Configuration saved successfully")

    if st.session_state.design_saved:
        if st.button("Run Simulation", type="primary", use_container_width=True):
            st.session_state.wizard_step = 2
            st.rerun()
        
# -------- STEP 2: RUN & RESULTS --------
elif st.session_state.wizard_step == 2:
    st.markdown("## Running Simulation...")
    st.button("← Back to Design", on_click=go_back)

    if not st.session_state["design"]:
        st.info("Use Save on Step 1 first.")
        st.session_state.wizard_step = 1
        st.rerun()

    p = st.session_state["design"]
    seed = p.get("seed", 42)
    num_replications = p.get("num_replications", 30)
    
    st.info(f"Seed: {seed} | Replications: {num_replications} | Days: {p['sim_minutes'] // DAY_MIN}")
    
    active_roles_caps = [("Providers", p["provider_cap"]), ("Front Desk", p["frontdesk_cap"]),
                        ("Nurse", p["nurse_cap"]), ("Back Office", p["backoffice_cap"])]
    active_roles = [r for r, cap in active_roles_caps if cap > 0]
    
    all_metrics = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for rep in range(num_replications):
        status_text.text(f"Running replication {rep + 1} of {num_replications}...")
        metrics = run_single_replication(p, seed + rep)
        all_metrics.append(metrics)
        progress_bar.progress((rep + 1) / num_replications)
    
    status_text.text(f"Completed {num_replications} replications")
    progress_bar.empty()
    
    agg_results = aggregate_replications(p, all_metrics, active_roles)
    
    flow_df = agg_results["flow_df"]
    time_at_role_df = agg_results["time_at_role_df"]
    queue_df = agg_results["queue_df"]
    rework_overview_df = agg_results["rework_overview_df"]
    loop_origin_df = agg_results["loop_origin_df"]
    throughput_full_df = agg_results["throughput_full_df"]
    util_df = agg_results["util_df"]
    summary_df = agg_results["summary_df"]
    
    burnout_data = calculate_burnout(all_metrics, p, active_roles)
    
    all_events_data = []
    for rep_idx, metrics in enumerate(all_metrics):
        for t, name, step, note, arr in metrics.events:
            all_events_data.append({
                "Replication": rep_idx + 1, "Time (min)": float(t), "Task": name,
                "Step": step, "Step label": pretty_step(step), "Note": note,
                "Arrival time (min)": (float(arr) if arr is not None else None),
                "Day": int(t // DAY_MIN)
            })
    events_df = pd.DataFrame(all_events_data)
    
    st.markdown(f"## Simulation Results")
    st.caption(f"Averaged over {num_replications} independent replications")
    
    create_kpi_banner(all_metrics, p, burnout_data, active_roles)
    
    st.markdown("---")
    
    st.markdown("## System Performance")
    st.caption("How well is the clinic handling incoming work?")
    
    col1, col2 = st.columns(2)
    with col1:
        throughput_df = plot_daily_throughput(all_metrics, p, active_roles)
        st.dataframe(throughput_df, use_container_width=True, hide_index=True)
    
    with col2:
        fig_queue = plot_queue_over_time(all_metrics, p, active_roles)
        st.pyplot(fig_queue, use_container_width=False)
        plt.close(fig_queue)
    
    col1, col2 = st.columns(2)
    with col1:
        help_icon("**Calculation:** Counts tasks completed each day across replications. "
                 "**Interpretation:** Declining = falling behind; stable/increasing = keeping up.")
    with col2:
        help_icon("**Calculation:** Tracks tasks waiting in each queue every minute (mean ± SD). "
                 "**Interpretation:** Persistent high queues = bottlenecks.")
    
    st.markdown("---")
    
    st.markdown("## Burnout & Workload Indicators")
    st.caption("Which roles are at risk of being overwhelmed?")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_burnout = plot_burnout_scores(burnout_data, active_roles)
        st.pyplot(fig_burnout, use_container_width=False)
        plt.close(fig_burnout)

    with col2:
        fig_utilization = plot_utilization_by_role(all_metrics, p, active_roles)
        st.pyplot(fig_utilization, use_container_width=False)
        plt.close(fig_utilization)

    col1, col2 = st.columns(2)
    with col1:
        fig_rework = plot_rework_impact(all_metrics, p, active_roles)
        st.pyplot(fig_rework, use_container_width=False)
        plt.close(fig_rework)

    with col2:
        fig_overtime = plot_overtime_needed(all_metrics, p, active_roles)
        st.pyplot(fig_overtime, use_container_width=False)
        plt.close(fig_overtime)
    
    col1, col2 = st.columns(2)
    with col1:
        help_icon("**Burnout Calculation (Refined):**\n"
                "• **Emotional Exhaustion (EE)** = 100 × (0.75×Utilization* + 0.25×AvailabilityStress)\n"
                "  *Non-linear: <75% utilization grows slowly, >75% accelerates rapidly\n\n"
                "• **Depersonalization (DP)** = 100 × (0.60×ReworkPct* + 0.40×QueueVolatility)\n"
                "  *ReworkPct uses quadratic penalty; QueueVolatility = task switching stress\n\n"
                "• **Reduced Accomplishment (RA)** = 100 × (0.55×Incompletion* + 0.45×ThroughputDeficit)\n"
                "  *Measures actual task completion vs. expected workload\n\n"
                "**Overall = Your custom weights × (EE, DP, RA)**\n\n"
                "**Interpretation:** 0–25 Low, 25–50 Moderate, 50–75 High, 75–100 Severe.")
    with col2:
        help_icon("**Calculation:** Utilization = (Actual work time) ÷ (Staff capacity × Open hours × Availability %)\n\n"
                 "Capped at 100% (can't exceed available time).\n\n"
                 "**Interpretation:**\n"
                 "• Green (<50%) = Underutilized\n"
                 "• Orange (50-75%) = Healthy workload\n"
                 "• Dark Orange (75-90%) = High stress\n"
                 "• Red (>90%) = Critical burnout risk")

    col1, col2 = st.columns(2)
    with col1:
        help_icon("**Rework Calculation:** Original work (blue) vs rework time (red). Rework = loops × 50% of service time. "
                 "**Interpretation:** High rework % = errors, missing info, poor handoffs.")
    with col2:
        help_icon("**Calculation:** (Total work needed - Available capacity) ÷ (Days × Staff count)\n\n"
                 "Measures additional hours per person per day needed to finish all tasks.\n\n"
                 "**Interpretation:**\n"
                 "• 0 hours = Keeping up with workload\n"
                 "• 0.5 hours = 30min overtime daily\n"
                 "• 1+ hours = Serious capacity shortage\n"
                 "• 2+ hours = Critical understaffing")
    
    st.markdown("---")

    st.markdown("## Download Data")
    with st.spinner("Producing run log..."):
        runlog_pkg = _runlog_workbook(events_df, engine=_excel_engine())
    
    st.download_button("Download Run Log (Excel)", data=runlog_pkg["bytes"],
                      file_name=f"RunLog_{num_replications}reps.{runlog_pkg['ext']}",
                      mime=runlog_pkg["mime"], type="primary")
