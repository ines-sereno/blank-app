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
from matplotlib.patches import Rectangle

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
# Diagram builder
# =============================
def build_process_graph(p: Dict) -> str:
    def _fmt_pct(x: float) -> str:
        try:
            return f"{100*float(x):.0f}%"
        except:
            return "0%"

    legend_fill = "#E9F7FF"
    cap = {"Front Desk": p.get("frontdesk_cap", 0), "Nurse": p.get("nurse_cap", 0),
           "Providers": p.get("provider_cap", 0), "Back Office": p.get("backoffice_cap", 0)}
    svc = {"Front Desk": p.get("svc_frontdesk", 0.0), "Nurse": p.get("svc_nurse", 0.0),
           "Providers": p.get("svc_provider", 0.0), "Back Office": p.get("svc_backoffice", 0.0)}
    svc_proto = p.get("svc_nurse_protocol", None)
    p_protocol = p.get("p_protocol", None)
    loops = {
        "Front Desk": (p.get("p_fd_insuff", 0.0), p.get("max_fd_loops", 0)),
        "Nurse": (p.get("p_nurse_insuff", 0.0), p.get("max_nurse_loops", 0)),
        "Providers": (p.get("p_provider_insuff", 0.0), p.get("max_provider_loops", 0)),
        "Back Office": (p.get("p_backoffice_insuff", 0.0), p.get("max_backoffice_loops", 0)),
    }
    route = p.get("route_matrix", {})
    roles = ["Front Desk", "Nurse", "Providers", "Back Office"]

    lines = ['digraph CHC {', '  rankdir=LR;', '  fontsize=12;',
             '  graph [size="10,4", nodesep=0.6, ranksep=0.8, overlap=false, pad="0.1,0.1"];',
             '  node [shape=roundrect, style=filled, fillcolor="#F7F7F7", color="#888", fontname="Helvetica", fontsize=10];',
             '  edge [color="#666", arrowsize=0.8, fontname="Helvetica", fontsize=9];']

    for r in roles:
        fill = "#EFEFEF" if cap.get(r, 0) <= 0 else "#F7F7F7"
        lines.append(f'  "{r}" [label="{r}\\ncap={cap.get(r,0)}\\nsvc≈{svc.get(r,0):.1f} min", fillcolor="{fill}"];')

    if p_protocol is not None and svc_proto is not None and cap.get("Nurse", 0) >= 0:
        proto_label = f'Nurse Protocol\\np={_fmt_pct(p_protocol)}\\nsvc≈{svc_proto:.1f} min'
        lines += [f'  "NurseProto" [shape=note, fillcolor="#FFFBE6", color="#B59F3B", label="{proto_label}", fontsize=9];',
                  '  "Nurse" -> "NurseProto" [style=dotted, label=" info "];']

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

    lines += [f'  legend [shape=box, style="rounded,filled", fillcolor="{legend_fill}", color="#AFC8D8", fontsize=8, '
              '          label="cap = capacity\\nsvc = mean svc time\\n→ routing prob\\n↺ loop prob / max"];',
              '  { rank=min; legend; }', '  legend -> "Front Desk" [style=invis, weight=9999];']
    lines.append('}')
    return "\n".join(lines)

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
    Simplified, non-overlapping burnout model with user-defined priority weights:
      - EE: Utilization + AvailabilityStress
      - DP: ReworkPct + QueuePressure
      - RA: WaitInefficiency + Incompletion
    Each subscale maps to 0–100, then Overall uses user-defined weights.
    """
    # Convert rankings to weights (1st=0.5, 2nd=0.3, 3rd=0.2)
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
        queue_pressure_list = []
        wait_inefficiency_list = []
        completion_rate_list = []

        for metrics in all_metrics:
            # Utilization (0–1)
            total_service = metrics.service_time_sum[role]
            denom = capacity * open_time_available
            util = total_service / max(1, denom)
            util_list.append(min(1.0, util))

            # ReworkPct (0–1): estimated rework time / total service time
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

            # QueuePressure (0–1): avg queue / capacity
            avg_queue = np.mean(metrics.queues[role]) if len(metrics.queues[role]) > 0 else 0.0
            queue_pressure = avg_queue / max(1, capacity)
            queue_pressure_list.append(min(1.0, queue_pressure))

            # WaitInefficiency (0–1): avg wait / (avg wait + avg work)
            avg_work = total_service / max(1, len(metrics.task_completion_time))
            avg_wait = np.mean(metrics.waits[role]) if len(metrics.waits[role]) > 0 else 0.0
            total_time = avg_work + avg_wait
            wait_ineff = (avg_wait / max(1, total_time)) if total_time > 0 else 0.0
            wait_inefficiency_list.append(min(1.0, wait_ineff))

            # Same-day completion rate → Incompletion (0–1)
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

        # Averages across replications
        avg_util = float(np.mean(util_list)) if util_list else 0.0
        avg_rework = float(np.mean(rework_pct_list)) if rework_pct_list else 0.0
        avg_queue_pressure = float(np.mean(queue_pressure_list)) if queue_pressure_list else 0.0
        avg_wait_ineff = float(np.mean(wait_inefficiency_list)) if wait_inefficiency_list else 0.0
        avg_incompletion = 1.0 - (float(np.mean(completion_rate_list)) if completion_rate_list else 0.0)

        # AvailabilityStress (0–1) from minutes available per hour
        avail_minutes = p.get("availability_per_hour", {}).get(role, 60)
        avail_stress = (60 - float(avail_minutes)) / 60.0
        avail_stress = min(max(avail_stress, 0.0), 1.0)

        # ---- New subscales (no shared metrics) ----
        EE = 100.0 * (0.70 * avg_util + 0.30 * avail_stress)                       # load + constraints
        DP = 100.0 * (0.60 * avg_rework + 0.40 * avg_queue_pressure)               # quality friction
        RA = 100.0 * (0.70 * avg_wait_ineff + 0.30 * avg_incompletion)             # outcomes + delays

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
def plot_utilization_heatmap(all_metrics: List[Metrics], p: Dict, active_roles: List[str]):
    open_minutes = p["open_minutes"]
    hours_in_day = int(np.ceil(open_minutes / 60.0))
    role_hour_utils = {r: np.zeros(hours_in_day) for r in active_roles}
    
    open_time_per_hour = 60
    for metrics in all_metrics:
        for r in active_roles:
            capacity = {"Front Desk": p["frontdesk_cap"], "Nurse": p["nurse_cap"],
                       "Providers": p["provider_cap"], "Back Office": p["backoffice_cap"]}[r]
            
            if capacity > 0:
                total_service = metrics.service_time_sum[r]
                num_days = p["sim_minutes"] / DAY_MIN
                avg_service_per_day = total_service / max(1, num_days)
                service_per_hour = avg_service_per_day / max(1, hours_in_day)
                
                for h in range(hours_in_day):
                    util = service_per_hour / (capacity * open_time_per_hour)
                    role_hour_utils[r][h] += util
    
    num_reps = len(all_metrics)
    for r in active_roles:
        role_hour_utils[r] /= num_reps
    
    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=40)
    data = np.array([role_hour_utils[r] for r in active_roles])
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1, interpolation='nearest')
    
    ax.set_xticks(np.arange(hours_in_day))
    ax.set_xticklabels([f"Hour {h+1}" for h in range(hours_in_day)], fontsize=8)
    ax.set_yticks(np.arange(len(active_roles)))
    ax.set_yticklabels(active_roles, fontsize=9)
    ax.set_xlabel('Hour of Workday', fontsize=10)
    ax.set_ylabel('Role', fontsize=10)
    ax.set_title('Utilization Heatmap', fontsize=11, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Utilization', rotation=270, labelpad=15, fontsize=9)
    
    for i in range(len(active_roles)):
        for j in range(hours_in_day):
            ax.text(j, i, f'{data[i, j]:.0%}', ha="center", va="center", color="black", fontsize=7)
    
    plt.tight_layout()
    return fig

def plot_queue_over_time(all_metrics: List[Metrics], p: Dict, active_roles: List[str]):
    fig, ax = plt.subplots(figsize=(6, 3), dpi=40)
    max_len = max(len(m.time_stamps) for m in all_metrics)
    colors = {'Front Desk': '#1f77b4', 'Nurse': '#ff7f0e', 'Providers': '#2ca02c', 'Back Office': '#d62728'}
    
    for role in active_roles:
        all_queues = []
        for metrics in all_metrics:
            q = metrics.queues[role]
            if len(q) < max_len:
                q = q + [q[-1]] * (max_len - len(q)) if len(q) > 0 else [0] * max_len
            all_queues.append(q[:max_len])
        
        queue_array = np.array(all_queues)
        mean_queue = np.mean(queue_array, axis=0)
        std_queue = np.std(queue_array, axis=0)
        time_days = np.array(all_metrics[0].time_stamps[:max_len]) / DAY_MIN
        
        color = colors.get(role, '#333333')
        ax.plot(time_days, mean_queue, label=role, color=color, linewidth=1.5)
        ax.fill_between(time_days, mean_queue - std_queue, mean_queue + std_queue, alpha=0.2, color=color)
    
    ax.set_xlabel('Time (days)', fontsize=10)
    ax.set_ylabel('Queue Length', fontsize=10)
    ax.set_title('Queue Length Over Time', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

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
    
    x = np.arange(len(active_roles))
    width = 0.6
    ax.bar(x, original_means, width, label='Original Work', color='#3498db')
    ax.bar(x, rework_means, width, bottom=original_means, label='Rework', color='#e74c3c')
    
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
            ax.text(i, total + 20, f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig

def plot_daily_throughput(all_metrics: List[Metrics], p: Dict):
    fig, ax = plt.subplots(figsize=(6, 3), dpi=40)
    num_days = max(1, int(p["sim_minutes"] // DAY_MIN))
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
    
    daily_array = np.array(daily_completions)
    mean_daily = np.mean(daily_array, axis=0)
    std_daily = np.std(daily_array, axis=0)
    days = np.arange(1, num_days + 1)
    
    ax.plot(days, mean_daily, marker='o', linewidth=1.5, markersize=6, color='#3498db', label='Mean')
    ax.fill_between(days, mean_daily - std_daily, mean_daily + std_daily, alpha=0.3, color='#3498db', label='± 1 SD')
    
    ax.set_xlabel('Day', fontsize=10)
    ax.set_ylabel('Tasks Completed', fontsize=10)
    ax.set_title('Daily Throughput', fontsize=11, fontweight='bold')
    ax.set_xticks(days)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_burnout_scores(burnout_data: Dict, active_roles: List[str]):
    """
    Bar chart showing burnout scores by role + overall clinic.
    Color-coded by severity.
    """
    fig, ax = plt.subplots(figsize=(6, 3), dpi=40)
    
    roles_plot = active_roles + ["Overall Clinic"]
    scores = [burnout_data["by_role"][r]["overall"] for r in active_roles]
    scores.append(burnout_data["overall_clinic"])
    
    # Color by severity
    colors = []
    for score in scores:
        if score < 25:
            colors.append('#2ecc71')  # Green - Low
        elif score < 50:
            colors.append('#f39c12')  # Orange - Moderate
        elif score < 75:
            colors.append('#e67e22')  # Dark Orange - High
        else:
            colors.append('#e74c3c')  # Red - Severe
    
    x = np.arange(len(roles_plot))
    bars = ax.bar(x, scores, color=colors, width=0.6)
    
    ax.set_xlabel('Role', fontsize=10)
    ax.set_ylabel('Burnout Score (0-100)', fontsize=10)
    ax.set_title('Burnout Index by Role (MBI-based)', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(roles_plot, fontsize=9, rotation=15, ha='right')
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{score:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Low (0-25)'),
        Patch(facecolor='#f39c12', label='Moderate (25-50)'),
        Patch(facecolor='#e67e22', label='High (50-75)'),
        Patch(facecolor='#e74c3c', label='Severe (75-100)')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=5)
    
    plt.tight_layout()
    return fig

def help_icon(help_text: str):
    with st.expander("ℹ️ How is this calculated?"):
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
st.set_page_config(page_title="HSyE Burnout Grant - DES Model for Community Health Centers", layout="wide")
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
    st.markdown("### 🏥 **Design Your Clinic**")
    
    # Define route_row_ui function BEFORE the form
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
                                   help="Number of clinic operating days to simulate. More days = more stable results but longer runtime")
        open_hours = st.number_input("Hours open per day", 1, 24, _init_ss("open_hours", 10), 1, "%d",
                                      help="Number of hours the clinic is open each day (e.g., 8am-6pm = 10 hours)")

        cv_speed_label = st.select_slider(
            "Task speed variability",
            options=["Very Low", "Low", "Moderate", "High", "Very High"],
            value=_init_ss("cv_speed_label", "Moderate"),
            help="How much variation is there in how long tasks take? Very Low = everyone works at nearly the same speed. Very High = some tasks finish much faster/slower than average"
        )
        
        # Map Likert scale to CV values
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
                               help="Seed for reproducibility. Same seed = same results")
        num_replications = st.number_input("Number of replications", 1, 1000, _init_ss("num_replications", 30), 1, "%d", 
                                          help="Number of independent simulation runs to average over. More replications = more stable results but longer runtime")

        st.markdown("### 👥 Role Configuration")
        st.caption("Configure staffing, arrivals, and availability for each role")
        
        
        # Front Desk
        with st.expander("🏢 Front Desk", expanded=True):
            cFD1, cFD2, cFD3 = st.columns(3)
            with cFD1:
                fd_cap_form = st.number_input("Staff on duty", 0, 50, _init_ss("fd_cap", 3), 1, "%d", key="fd_cap_input",
                                                           help="Number of front desk staff on duty who handle check-ins, scheduling, and administrative tasks")
            with cFD2:
                arr_fd = st.number_input("Arrivals per hour", 0, 500, _init_ss("arr_fd", 15), 1, "%d", disabled=(fd_cap_form==0), key="arr_fd_input",
                                         help="Average number of tasks arriving at Front Desk per hour (e.g., check-ins, phone calls)")
            with cFD3:
                avail_fd = st.number_input("Availability (min/hour)", 0, 60, _init_ss("avail_fd", 45), 1, "%d", disabled=(fd_cap_form==0), key="avail_fd_input",
                                           help="Minutes per hour that front desk staff are available for work (60 = always available, 45 = 75% available due to breaks, meetings, etc.)")
        
        # Nurse / MAs
        with st.expander("💉 Nurse / MAs", expanded=True):
            cNU1, cNU2, cNU3 = st.columns(3)
            with cNU1:
                nu_cap_form = st.number_input("Staff on duty", 0, 50, _init_ss("nurse_cap", 2), 1, "%d", key="nurse_cap_input",
                                                              help="Number of nurses or medical assistants on duty who handle triage, vitals, and patient prep")
            with cNU2:
                arr_nu = st.number_input("Arrivals per hour", 0, 500, _init_ss("arr_nu", 20), 1, "%d", disabled=(nu_cap_form==0), key="arr_nu_input",
                                         help="Average number of tasks arriving directly at Nurses per hour (e.g., triage requests, prescription refills)")
            with cNU3:
                avail_nu = st.number_input("Availability (min/hour)", 0, 60, _init_ss("avail_nu", 20), 1, "%d", disabled=(nu_cap_form==0), key="avail_nu_input",
                                           help="Minutes per hour that nurses are available for work (lower values simulate interruptions, charting time, etc.)")
        
        # Providers
        with st.expander("👨‍⚕️ Providers", expanded=True):
            cPR1, cPR2, cPR3 = st.columns(3)
            with cPR1:
                pr_cap_form = st.number_input("Staff on duty", 0, 50, _init_ss("provider_cap", 1), 1, "%d", key="provider_cap_input",
                                                                 help="Number of providers (doctors, NPs, PAs) on duty who see patients and make medical decisions")
            with cPR2:
                arr_pr = st.number_input("Arrivals per hour", 0, 500, _init_ss("arr_pr", 10), 1, "%d", disabled=(pr_cap_form==0), key="arr_pr_input",
                                         help="Average number of tasks arriving directly at Providers per hour (e.g., direct consultations)")
            with cPR3:
                avail_pr = st.number_input("Availability (min/hour)", 0, 60, _init_ss("avail_pr", 30), 1, "%d", disabled=(pr_cap_form==0), key="avail_pr_input",
                                           help="Minutes per hour that providers are available for patient-facing work (accounts for charting, calls, etc.)")
        
        # Back Office
        with st.expander("📋 Back Office", expanded=True):
            cBO1, cBO2, cBO3 = st.columns(3)
            with cBO1:
                bo_cap_form = st.number_input("Staff on duty", 0, 50, _init_ss("backoffice_cap", 1), 1, "%d", key="bo_cap_input",
                                                           help="Number of back office staff on duty who handle billing, insurance, follow-up calls, and administrative work")
            with cBO2:
                arr_bo = st.number_input("Arrivals per hour", 0, 500, _init_ss("arr_bo", 5), 1, "%d", disabled=(bo_cap_form==0), key="arr_bo_input",
                                         help="Average number of tasks arriving directly at Back Office per hour (e.g., insurance claims, billing questions)")
            with cBO3:
                avail_bo = st.number_input("Availability (min/hour)", 0, 60, _init_ss("avail_bo", 45), 1, "%d", disabled=(bo_cap_form==0), key="avail_bo_input",
                                           help="Minutes per hour that back office staff are available for work")

        with st.expander("⚙️ Advanced Settings – Service times, loops & routing", expanded=False):
            
            st.markdown("### Burnout Priority Weights")
            st.caption("Rank the burnout dimensions by importance (1 = most important, 3 = least important)")
            cB1, cB2, cB3 = st.columns(3)
            with cB1:
                ee_rank = st.selectbox("Emotional Exhaustion", [1, 2, 3], index=0, key="ee_rank",
                                      help="Rank importance of Emotional Exhaustion (workload intensity, time pressure, availability constraints)")
            with cB2:
                dp_rank = st.selectbox("Depersonalization", [1, 2, 3], index=1, key="dp_rank",
                                      help="Rank importance of Depersonalization (rework, quality issues, queue pressure)")
            with cB3:
                ra_rank = st.selectbox("Reduced Accomplishment", [1, 2, 3], index=2, key="ra_rank",
                                      help="Rank importance of Reduced Accomplishment (delays, incomplete work, inefficiency)")

            # Validate rankings
            ranks = [ee_rank, dp_rank, ra_rank]
            if len(set(ranks)) != 3:
                st.error("⚠️ Each dimension must have a unique rank (1, 2, or 3). Please assign each rank exactly once.")
            else:
                # Show the weights that will be applied
                rank_to_weight = {1: 0.5, 2: 0.3, 3: 0.2}
                st.success(f"✓ Weights will be: EE={rank_to_weight[ee_rank]:.1f}, DP={rank_to_weight[dp_rank]:.1f}, RA={rank_to_weight[ra_rank]:.1f}")
            
            st.markdown("---")
            
            # Front Desk
            with st.expander("🏢 Front Desk", expanded=False):
                st.markdown("**Service Time**")
                svc_frontdesk = st.slider("Mean service time (minutes)", 0.0, 30.0, _init_ss("svc_frontdesk", 3.0), 0.5, disabled=(fd_cap_form==0),
                                          help="Average time (minutes) for front desk to complete a task")
                
                st.markdown("**Rework Loops**")
                cFDL1, cFDL2, cFDL3 = st.columns(3)
                with cFDL1:
                    p_fd_insuff = st.slider("Probability of missing info", 0.0, 1.0, _init_ss("p_fd_insuff", 0.15), 0.01, disabled=(fd_cap_form==0), key="fd_p_insuff",
                                            help="Probability that front desk encounters missing information requiring rework")
                with cFDL2:
                    max_fd_loops = st.number_input("Max loops", 0, 10, _init_ss("max_fd_loops", 2), 1, "%d", disabled=(fd_cap_form==0), key="fd_max_loops",
                                                   help="Maximum number of rework loops before giving up")
                with cFDL3:
                    fd_loop_delay = st.slider("Rework delay (min)", 0.0, 60.0, _init_ss("fd_loop_delay", 5.0), 0.5, disabled=(fd_cap_form==0), key="fd_delay",
                                              help="Time delay (minutes) before starting rework loop")
                
                st.markdown("**Routing: Where tasks go after Front Desk**")
                fd_route = route_row_ui("Front Desk", {"Nurse": 0.50, "Providers": 0.10, "Back Office": 0.10, DONE: 0.30}, 
                                       disabled_source=(fd_cap_form==0), fd_cap_val=fd_cap_form, nu_cap_val=nu_cap_form, 
                                       pr_cap_val=pr_cap_form, bo_cap_val=bo_cap_form)
            
            # Nurse / MAs
            with st.expander("💉 Nurse / MAs", expanded=False):
                st.markdown("**Service Times**")
                cNS1, cNS2 = st.columns(2)
                with cNS1:
                    svc_nurse_protocol = st.slider("Protocol service time (minutes)", 0.0, 30.0, _init_ss("svc_nurse_protocol", 2.0), 0.5, disabled=(nu_cap_form==0),
                                                    help="Average time (minutes) when nurse resolves task via protocol without provider")
                    p_protocol = st.slider("Probability of using protocol", 0.0, 1.0, _init_ss("p_protocol", 0.40), 0.05, disabled=(nu_cap_form==0),
                                          help="Probability that a nurse can resolve a task using standing protocols without provider involvement")
                with cNS2:
                    svc_nurse = st.slider("Non-protocol service time (minutes)", 0.0, 40.0, _init_ss("svc_nurse", 4.0), 0.5, disabled=(nu_cap_form==0),
                                         help="Average time (minutes) for nurse tasks that don't use protocol")
                
                st.markdown("**Rework Loops**")
                cNUL1, cNUL2 = st.columns(2)
                with cNUL1:
                    p_nurse_insuff = st.slider("Probability of insufficient info", 0.0, 1.0, _init_ss("p_nurse_insuff", 0.10), 0.01, disabled=(nu_cap_form==0), key="nu_p_insuff",
                                               help="Probability that nurse needs to send task back to front desk for more info")
                with cNUL2:
                    max_nurse_loops = st.number_input("Max loops", 0, 10, _init_ss("max_nurse_loops", 2), 1, "%d", disabled=(nu_cap_form==0), key="nu_max_loops",
                                                      help="Maximum number of nurse rework loops before giving up")
                
                st.markdown("**Routing: Where tasks go after Nurse**")
                nu_route = route_row_ui("Nurse", {"Providers": 0.40, "Back Office": 0.20, DONE: 0.40}, 
                                       disabled_source=(nu_cap_form==0), fd_cap_val=fd_cap_form, nu_cap_val=nu_cap_form, 
                                       pr_cap_val=pr_cap_form, bo_cap_val=bo_cap_form)
            
            # Providers
            with st.expander("👨‍⚕️ Providers", expanded=False):
                st.markdown("**Service Time**")
                svc_provider = st.slider("Mean service time (minutes)", 0.0, 60.0, _init_ss("svc_provider", 6.0), 0.5, disabled=(pr_cap_form==0),
                                        help="Average time (minutes) for provider to see a patient or complete a task")
                
                st.markdown("**Rework Loops**")
                cPRL1, cPRL2, cPRL3 = st.columns(3)
                with cPRL1:
                    p_provider_insuff = st.slider("Probability of rework needed", 0.0, 1.0, _init_ss("p_provider_insuff", 0.08), 0.01, disabled=(pr_cap_form==0), key="pr_p_insuff",
                                                  help="Probability that provider task needs rework (unclear orders, missing info, etc.)")
                with cPRL2:
                    max_provider_loops = st.number_input("Max loops", 0, 10, _init_ss("max_provider_loops", 2), 1, "%d", disabled=(pr_cap_form==0), key="pr_max_loops",
                                                        help="Maximum number of provider rework loops before giving up")
                with cPRL3:
                    provider_loop_delay = st.slider("Rework delay (min)", 0.0, 60.0, _init_ss("provider_loop_delay", 5.0), 0.5, disabled=(pr_cap_form==0), key="pr_delay",
                                                    help="Time delay (minutes) before provider rework")
                
                st.markdown("**Routing: Where tasks go after Providers**")
                pr_route = route_row_ui("Providers", {"Back Office": 0.30, DONE: 0.70}, 
                                       disabled_source=(pr_cap_form==0), fd_cap_val=fd_cap_form, nu_cap_val=nu_cap_form, 
                                       pr_cap_val=pr_cap_form, bo_cap_val=bo_cap_form)
            
            # Back Office
            with st.expander("📋 Back Office", expanded=False):
                st.markdown("**Service Time**")
                svc_backoffice = st.slider("Mean service time (minutes)", 0.0, 60.0, _init_ss("svc_backoffice", 5.0), 0.5, disabled=(bo_cap_form==0),
                                           help="Average time (minutes) for back office to complete a task (billing, insurance, etc.)")
                
                st.markdown("**Rework Loops**")
                cBOL1, cBOL2, cBOL3 = st.columns(3)
                with cBOL1:
                    p_backoffice_insuff = st.slider("Probability of rework needed", 0.0, 1.0, _init_ss("p_backoffice_insuff", 0.05), 0.01, disabled=(bo_cap_form==0), key="bo_p_insuff",
                                                    help="Probability that back office task needs rework (billing errors, missing documentation)")
                with cBOL2:
                    max_backoffice_loops = st.number_input("Max loops", 0, 10, _init_ss("max_backoffice_loops", 2), 1, "%d", disabled=(bo_cap_form==0), key="bo_max_loops",
                                                          help="Maximum number of back office rework loops before giving up")
                with cBOL3:
                    backoffice_loop_delay = st.slider("Rework delay (min)", 0.0, 60.0, _init_ss("backoffice_loop_delay", 5.0), 0.5, disabled=(bo_cap_form==0), key="bo_delay",
                                                      help="Time delay (minutes) before back office rework")
                
                st.markdown("**Routing: Where tasks go after Back Office**")
                bo_route = route_row_ui("Back Office", {"Front Desk": 0.10, "Nurse": 0.10, "Providers": 0.10, DONE: 0.70}, 
                                       disabled_source=(bo_cap_form==0), fd_cap_val=fd_cap_form, nu_cap_val=nu_cap_form, 
                                       pr_cap_val=pr_cap_form, bo_cap_val=bo_cap_form)
            
            # Build the route dictionary
            route: Dict[str, Dict[str, float]] = {}
            route["Front Desk"] = fd_route
            route["Nurse"] = nu_route
            route["Providers"] = pr_route
            route["Back Office"] = bo_route

        saved = st.form_submit_button("Save", type="primary")

        if saved:
            # Save capacity values to session state
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
            st.success("✅ Configuration saved successfully!")

    # Show configuration preview and run button after save
    if st.session_state.design_saved:
        st.markdown("---")
        st.markdown("### 🎯 Configuration Summary")
        p = st.session_state["design"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Simulation Days", p["sim_minutes"] // DAY_MIN)
            st.metric("Hours/Day", p["open_minutes"] // 60)
        with col2:
            st.metric("Random Seed", p["seed"])
            st.metric("Replications", p["num_replications"])
        with col3:
            active_count = sum(1 for cap in [p["frontdesk_cap"], p["nurse_cap"], p["provider_cap"], p["backoffice_cap"]] if cap > 0)
            st.metric("Active Roles", f"{active_count}/4")
            st.metric("Total Staff", p["frontdesk_cap"] + p["nurse_cap"] + p["provider_cap"] + p["backoffice_cap"])
        
        st.markdown("---")
        
        # Process preview
        dot = build_process_graph(p)
        with st.expander("📋 Process Flow Preview", expanded=False):
            st.caption("Visual representation of your clinic configuration")
            st.graphviz_chart(dot, use_container_width=False)
        
        # Run simulation button
        if st.button("▶️ Run Simulation", type="primary", use_container_width=True):
            st.session_state.wizard_step = 2
            st.rerun()
        
# -------- STEP 2: RUN & RESULTS --------
elif st.session_state.wizard_step == 2:
    st.markdown("## ⚙️ Running Simulation...")
    st.button("← Back to Design", on_click=go_back)

    if not st.session_state["design"]:
        st.info("Use **Save** on Step 1 first.")
        st.session_state.wizard_step = 1
        st.rerun()

    p = st.session_state["design"]
    seed = p.get("seed", 42)
    num_replications = p.get("num_replications", 30)
    
    # Show what's being run
    st.info(f"🎲 Seed: {seed} | 🔄 Replications: {num_replications} | 📅 Days: {p['sim_minutes'] // DAY_MIN}")
    
    # Run simulation automatically
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
    
    status_text.text(f"✅ Completed {num_replications} replications!")
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
    
    # Calculate burnout
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
    
    # ── RENDER ─────────────────────────────────────────────────────────
    st.markdown(f"## 📊 Simulation Results")
    st.caption(f"Averaged over {num_replications} independent replications")
    st.markdown("---")
    
    # ============================================================
    # SECTION 1: SYSTEM PERFORMANCE
    # ============================================================
    st.markdown("## 📈 System Performance")
    st.caption("How well is the clinic handling incoming work?")
    
    # Graphs side by side
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Daily Throughput Trend")
        fig_throughput = plot_daily_throughput(all_metrics, p)
        st.pyplot(fig_throughput, use_container_width=False)
        plt.close(fig_throughput)
    
    with col2:
        st.markdown("### Queue Length Over Time")
        fig_queue = plot_queue_over_time(all_metrics, p, active_roles)
        st.pyplot(fig_queue, use_container_width=False)
        plt.close(fig_queue)
    
    # Help text below graphs
    col1, col2 = st.columns(2)
    with col1:
        help_icon("**Calculation:** Counts tasks completed each day across replications. "
                 "Line shows mean, shaded area is ±1 SD. **Interpretation:** Declining = falling behind; stable/increasing = keeping up.")
    with col2:
        help_icon("**Calculation:** Tracks tasks waiting in each queue every minute (mean ± SD). "
                 "**Interpretation:** Persistent high queues = bottlenecks. Growing queues = can't keep up.")
    
    st.markdown("---")
    
    # ============================================================
    # SECTION 2: BURNOUT & WORKLOAD INDICATORS
    # ============================================================
    st.markdown("## 🔥 Burnout & Workload Indicators")
    st.caption("Which roles are at risk of being overwhelmed?")
    
    # Top row: Burnout Index | Utilization Heatmap
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Burnout Index (MBI-based)")
        fig_burnout = plot_burnout_scores(burnout_data, active_roles)
        st.pyplot(fig_burnout, use_container_width=False)
        plt.close(fig_burnout)
        
        # Rework Impact below Burnout Index
        st.markdown("### Rework Impact")
        fig_rework = plot_rework_impact(all_metrics, p, active_roles)
        st.pyplot(fig_rework, use_container_width=False)
        plt.close(fig_rework)
    
    with col2:
        st.markdown("### Utilization Heatmap")
        fig_heatmap = plot_utilization_heatmap(all_metrics, p, active_roles)
        st.pyplot(fig_heatmap, use_container_width=False)
        plt.close(fig_heatmap)
    
    # Help text below
    col1, col2 = st.columns(2)
    with col1:
        help_icon("**Burnout Calculation (Simplified):**\n"
                "• **Emotional Exhaustion (EE)** = 100 × (0.70×Utilization + 0.30×AvailabilityStress)\n"
                "• **Depersonalization (DP)**    = 100 × (0.60×ReworkPct + 0.40×QueuePressure)\n"
                "• **Reduced Accomplishment (RA)** = 100 × (0.70×WaitInefficiency + 0.30×Incompletion)\n\n"
                "**Overall = Your custom weights × (EE, DP, RA)**\n\n"
                "**Interpretation:** 0–25 Low, 25–50 Moderate, 50–75 High, 75–100 Severe.")
        help_icon("**Rework Calculation:** Original work (blue) vs rework time (red). Rework = loops × 50% of service time. "
                 "**Interpretation:** High rework % = errors, missing info, poor handoffs. Rework drives burnout.")
    with col2:
        help_icon("**Calculation:** % of available time each role spends working per hour (service time ÷ capacity × available minutes). "
                 "**Interpretation:** Red (>80%) = high burnout risk. Yellow (50-80%) = moderate. Green (<50%) = underutilized.")
    
    st.markdown("---")

    st.markdown("## 💾 Download Data")
    with st.spinner("Producing run log for download..."):
        runlog_pkg = _runlog_workbook(events_df, engine=_excel_engine())
    
    st.download_button("Download Run Log (Excel)", data=runlog_pkg["bytes"],
                      file_name=f"RunLog_{num_replications}reps.{runlog_pkg['ext']}",
                      mime=runlog_pkg["mime"], type="primary")
