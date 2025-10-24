import streamlit as st
import simpy
import random
import numpy as np
import math
import pandas as pd
from typing import Dict, Tuple, List

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

def day_index_at(t_min):
    return int(t_min // DAY_MIN)

def effective_open_minutes(sim_minutes, open_minutes):
    full_days = int(sim_minutes // DAY_MIN)
    remainder = sim_minutes % DAY_MIN
    return full_days * open_minutes + min(open_minutes, remainder)

# =============================
# Metrics
# =============================
ROLES = ["Front Desk", "Nurse", "Provider", "Back Office"]
DONE = "Done"

class Metrics:
    def __init__(self):
        self.time_stamps = []
        self.queues = {r: [] for r in ROLES}
        self.waits = {r: [] for r in ROLES}
        self.taps = {r: 0 for r in ROLES}
        self.completed = 0
        self.resolved_admin = 0
        self.resolved_protocol = 0
        self.routed_backoffice = 0

        # arrivals
        self.arrivals_total = 0
        self.arrivals_by_role = {r: 0 for r in ROLES}

        # misrouting
        self.misrouted_at_arrival = 0
        self.misrouted_at_arrival_by_role = {r: 0 for r in ROLES}
        self.misrouted_between = 0
        self.misrouted_between_by_fromrole = {r: 0 for r in ROLES}

        # service time
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
# Step labels (for logs)
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
    "MISROUTE_ARR": "Misrouted at arrival",
    "MISROUTE_BET": "Misrouted between roles",
    "DONE": "Task resolved"
}
def pretty_step(code):
    return STEP_LABELS.get(code, code)

# =============================
# System
# =============================
class CHCSystem:
    def __init__(self, env, params, metrics):
        self.env = env
        self.p = params
        self.m = metrics

        # Store caps
        self.fd_cap = params["frontdesk_cap"]
        self.nu_cap = params["nurse_cap"]
        self.pr_cap = params["provider_cap"]
        self.bo_cap = params["backoffice_cap"]

        # Resources (None when capacity == 0)
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

    def scheduled_service(self, resource, role_account, mean_time, role_for_dist=None):
        """
        If resource is None (capacity 0), skip service entirely.
        Otherwise, perform service in chunks during open hours.
        """
        if resource is None or mean_time <= 1e-12:
            return
        if role_for_dist is None:
            role_for_dist = role_account

        remaining = draw_service_time(role_for_dist, mean_time, self.p["dist_role"], self.p["cv_speed"])
        emr = max(0.0, self.p["emr_overhead"].get(role_account, 0.0))
        remaining += emr

        while remaining > 1e-9:
            if not is_open(self.env.now, self.p["open_minutes"]):
                yield self.env.timeout(minutes_until_open(self.env.now, self.p["open_minutes"]))
            window = minutes_until_close(self.env.now, self.p["open_minutes"])
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
# Processes – arrivals per role & generic workflow
# =============================
def sample_next_role(route_row: Dict[str, float]) -> str:
    # route_row includes keys: ROLES + DONE; values may not sum to 1 → normalize
    keys = list(route_row.keys())
    vals = np.array([max(0.0, float(route_row[k])) for k in keys], dtype=float)
    s = vals.sum()
    if s <= 0:
        # default fail-safe = Done
        return DONE
    probs = vals / s
    return random.choices(keys, weights=probs, k=1)[0]

def pick_random_other_role(exclude: List[str]) -> str:
    choices = [r for r in ROLES if r not in exclude]
    if not choices:
        return DONE
    return random.choice(choices)

def handle_role(env, task_id, s: CHCSystem, role: str):
    if role not in ["Front Desk", "Nurse", "Provider", "Back Office"]:
        return "Done"
    """
    Execute service (and role-specific loops), then return the next role (or DONE).
    """
    res = s.role_to_res[role]

    if role == "Front Desk":
        if res is not None:
            s.m.log(env.now, task_id, "FD_QUEUE", "")
            yield from s.scheduled_service(res, "Front Desk", s.p["svc_frontdesk"])
            s.m.log(env.now, task_id, "FD_DONE", "")
            # FD loops
            fd_loops = 0
            while (fd_loops < s.p["max_fd_loops"]) and (random.random() < s.p["p_fd_insuff"]):
                fd_loops += 1
                s.m.loop_fd_insufficient += 1
                s.m.log(env.now, task_id, "FD_INSUFF", f"Missing info loop #{fd_loops}")
                yield env.timeout(s.p["fd_loop_delay"])
                s.m.log(env.now, task_id, "FD_RETRY_QUEUE", f"Loop #{fd_loops}")
                yield from s.scheduled_service(res, "Front Desk", s.p["svc_frontdesk"])
                s.m.log(env.now, task_id, "FD_RETRY_DONE", f"Loop #{fd_loops}")
        # Admin resolve only if FD exists (same behavior), unless overridden
        if (res is not None) or s.p.get("admin_possible_without_fd", False):
            if random.random() < s.p["p_admin"]:
                s.m.resolved_admin += 1
                return DONE

    elif role == "Nurse":
        if res is not None:
            s.m.log(env.now, task_id, "NU_QUEUE", "")
            if random.random() < s.p["p_protocol"]:
                yield from s.scheduled_service(res, "Nurse", s.p["svc_nurse_protocol"], role_for_dist="NurseProtocol")
                s.m.resolved_protocol += 1
                # optional provider sign-off
                if not s.p.get("require_provider_signoff", False) or (s.role_to_res["Provider"] is None):
                    return DONE
                # else fall through to routing to Provider per matrix
            else:
                yield from s.scheduled_service(res, "Nurse", s.p["svc_nurse"])
                s.m.log(env.now, task_id, "NU_DONE", "")
            # Nurse loops
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
            # Provider loops
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
            s.m.routed_backoffice += 1
            s.m.log(env.now, task_id, "BO_DONE", "")
            # BO loops
            bo_loops = 0
            while (bo_loops < s.p["max_backoffice_loops"]) and (random.random() < s.p["p_backoffice_insuff"]):
                bo_loops += 1
                s.m.loop_backoffice_insufficient += 1
                s.m.log(env.now, task_id, "BO_INSUFF", f"Back Office rework loop #{bo_loops}")
                yield env.timeout(s.p["backoffice_loop_delay"])
                s.m.log(env.now, task_id, "BO_RECHECK_QUEUE", f"Loop #{bo_loops}")
                yield from s.scheduled_service(res, "Back Office", max(0.0, 0.5 * s.p["svc_backoffice"]))
                s.m.log(env.now, task_id, "BO_RECHECK_DONE", f"Loop #{bo_loops}")

    # Decide next step from routing matrix
    row = s.p["route_matrix"].get(role, {DONE: 1.0})
    nxt = sample_next_role(row)

    # Optional: force BO after Provider
    if (role == "Provider") and (nxt != DONE) and s.p.get("force_backoffice_after_provider", False):
        if s.role_to_res["Back Office"] is not None:
            nxt = "Back Office"

    # Misroute between roles (only if going to a role, not DONE)
    if nxt in ROLES and (random.random() < s.p["p_wrong_between"]):
        wrong = pick_random_other_role(exclude=[nxt])
        s.m.misrouted_between += 1
        s.m.misrouted_between_by_fromrole[role] += 1
        s.m.log(env.now, task_id, "MISROUTE_BET", f"{role} intended {nxt} → misrouted to {wrong}")
        nxt = wrong

    return nxt

def task_lifecycle(env, task_id: str, s: CHCSystem, initial_role: str, misrouted_arrival_to: str | None):
    """
    Runs a task starting at initial_role (or misrouted to another role first).
    """
    # arrival log
    s.m.task_arrival_time[task_id] = env.now
    s.m.arrivals_total += 1
    s.m.arrivals_by_role[initial_role] += 1
    s.m.log(env.now, task_id, "ARRIVE", f"Arrived at {initial_role}", arrival_t=env.now)

    role = initial_role

    # If it was misrouted-at-arrival, bounce immediately to another role
    if misrouted_arrival_to is not None:
        s.m.misrouted_at_arrival += 1
        s.m.misrouted_at_arrival_by_role[initial_role] += 1
        s.m.log(env.now, task_id, "MISROUTE_ARR", f"Arrived to {initial_role} but re-routed to {misrouted_arrival_to}")
        role = misrouted_arrival_to

    # Iterate through roles until DONE (guard against infinite loops)
    for _ in range(50):
        nxt = yield from handle_role(env, task_id, s, role)
        if nxt == DONE:
            s.m.completed += 1
            s.m.task_completion_time[task_id] = env.now
            s.m.log(env.now, task_id, "DONE", "Task completed")
            return
        # If next role doesn't exist, we still "visit" it (service skips because resource=None)
        role = nxt

    # Safety fallback
    s.m.completed += 1
    s.m.task_completion_time[task_id] = env.now
    s.m.log(env.now, task_id, "DONE", "Max handoffs reached — forced completion")

def arrival_process_for_role(env, s: CHCSystem, role_name: str, rate_per_hour: float):
    """
    Independent Poisson arrivals to a given role.
    Supports 'wrong at arrival' by re-routing immediately with a log + metric.
    """
    i = 0
    lam = max(0.0, rate_per_hour) / 60.0
    while True:
        inter = random.expovariate(lam) if lam > 0 else 999999999
        yield env.timeout(inter)
        i += 1
        task_id = f"{role_name[:2].upper()}-{i:05d}"

        # Wrong at arrival?
        misroute_to = None
        if random.random() < s.p["p_wrong_at_arrival"]:
            # choose any other role (if none, ignore)
            other = pick_random_other_role(exclude=[role_name])
            misroute_to = other if other != DONE else None

        env.process(task_lifecycle(env, task_id, s, initial_role=role_name, misrouted_arrival_to=misroute_to))

def monitor(env, s: CHCSystem):
    while True:
        s.m.time_stamps.append(env.now)
        for r in ROLES:
            res = s.role_to_res[r]
            s.m.queues[r].append(len(res.queue) if res is not None else 0)
        yield env.timeout(1)

# =============================
# Streamlit UI (2-step wizard, no graphs)
# =============================
st.set_page_config(page_title="CHC Workflow Simulator", layout="wide")
st.title("CHC Workflow Simulator")

if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 1
if "results" not in st.session_state:
    st.session_state["results"] = None
if "design" not in st.session_state:
    st.session_state["design"] = None

def go_next():
    st.session_state.wizard_step = min(2, st.session_state.wizard_step + 1)
def go_back():
    st.session_state.wizard_step = max(1, st.session_state.wizard_step - 1)

# -------- STEP 1: DESIGN --------
if st.session_state.wizard_step == 1:
    st.subheader("Step 1 — Design your clinic")

    with st.form("design_form", clear_on_submit=False):
        c1, c2 = st.columns([1,1])
        with c1:
            sim_days = st.number_input("Days to simulate", min_value=1, max_value=30, value=1, step=1)
            open_hours = st.number_input("Hours open per day", min_value=1, max_value=24, value=8, step=1)
            cv_speed = st.slider("Task speed variability (CV)", 0.0, 0.8, 0.25, 0.05)
        with c2:
            st.markdown("**Staffing (on duty)**")
            fd_cap = st.number_input("Front Desk staff", min_value=0, max_value=50, value=1, step=1)
            nurse_cap = st.number_input("Nurses / MAs", min_value=0, max_value=50, value=1, step=1)
            provider_cap = st.number_input("Providers", min_value=0, max_value=50, value=1, step=1)
            bo_cap = st.number_input("Back Office staff", min_value=0, max_value=50, value=1, step=1)

        st.markdown("### Arrivals (per hour) — can enter anywhere")
        cA1, cA2, cA3, cA4 = st.columns(4)
        with cA1:
            arr_fd = st.number_input("→ Front Desk", min_value=0.0, max_value=500.0, value=12.0, step=1.0)
        with cA2:
            arr_nu = st.number_input("→ Nurse / MAs", min_value=0.0, max_value=500.0, value=0.0, step=1.0)
        with cA3:
            arr_pr = st.number_input("→ Provider", min_value=0.0, max_value=500.0, value=0.0, step=1.0)
        with cA4:
            arr_bo = st.number_input("→ Back Office", min_value=0.0, max_value=500.0, value=0.0, step=1.0)

        st.markdown("### Misrouting settings")
        cM1, cM2, cM3 = st.columns([1,1,1])
        with cM1:
            p_wrong_at_arrival = st.slider("Wrong at arrival", 0.0, 0.5, 0.05, 0.01,
                help="Probability an arriving task was sent to the wrong role and is immediately re-routed.")
        with cM2:
            p_wrong_between = st.slider("Wrong between roles", 0.0, 0.5, 0.05, 0.01,
                help="Probability a handoff goes to a different (wrong) role than intended.")
        with cM3:
            require_provider_signoff = st.checkbox("Require Provider sign-off after Nurse Protocol", value=False)

        # Service times and loops
        st.markdown("### Service times (mean minutes) & loops")
        cS1, cS2 = st.columns([1,1])
        with cS1:
            svc_frontdesk = st.slider("Front Desk", 0.0, 30.0, 3.0, 0.5)
            svc_nurse_protocol = st.slider("Nurse Protocol", 0.0, 30.0, 2.0, 0.5)
            svc_nurse = st.slider("Nurse (non-protocol)", 0.0, 40.0, 4.0, 0.5)
            svc_provider = st.slider("Provider", 0.0, 60.0, 6.0, 0.5)
            svc_backoffice = st.slider("Back Office", 0.0, 60.0, 5.0, 0.5)
        with cS2:
            st.caption("Loop probabilities & delays")
            p_fd_insuff = st.slider("Front Desk loop chance", 0.0, 0.6, 0.05, 0.01)
            max_fd_loops = st.slider("Max FD loops", 0, 10, 3)
            fd_loop_delay = st.slider("FD loop delay (min)", 0.0, 240.0, 30.0, 5.0)
            p_nurse_insuff = st.slider("Nurse loop chance", 0.0, 0.6, 0.05, 0.01)
            max_nurse_loops = st.slider("Max Nurse loops", 0, 10, 2)
            p_provider_insuff = st.slider("Provider loop chance", 0.0, 0.6, 0.00, 0.01)
            max_provider_loops = st.slider("Max Provider loops", 0, 10, 1)
            provider_loop_delay = st.slider("Provider loop delay (min)", 0.0, 240.0, 15.0, 5.0)
            p_backoffice_insuff = st.slider("Back Office loop chance", 0.0, 0.6, 0.00, 0.01)
            max_backoffice_loops = st.slider("Max Back Office loops", 0, 10, 1)
            backoffice_loop_delay = st.slider("Back Office loop delay (min)", 0.0, 240.0, 15.0, 5.0)

        st.markdown("### Routing matrix (probabilities will be normalized)")
        st.caption("For each row, set the probability of sending a task to another role or to Done. Rows will be normalized so they sum to 1.")
        # We collect raw (possibly non-summing) probs and normalize during run.
        route = {}
        # Helper to render one row
        def route_row_ui(from_role: str, defaults: Dict[str, float]):
            st.markdown(f"**{from_role} →**")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                to_fd = st.number_input(f"to FD ({from_role})", 0.0, 1.0, defaults.get("Front Desk", 0.0), 0.05, key=f"r_{from_role}_fd")
            with c2:
                to_nu = st.number_input(f"to Nurse ({from_role})", 0.0, 1.0, defaults.get("Nurse", 0.5 if from_role=="Front Desk" else 0.0), 0.05, key=f"r_{from_role}_nu")
            with c3:
                to_pr = st.number_input(f"to Provider ({from_role})", 0.0, 1.0, defaults.get("Provider", 0.4 if from_role=="Nurse" else 0.0), 0.05, key=f"r_{from_role}_pr")
            with c4:
                to_bo = st.number_input(f"to Back Office ({from_role})", 0.0, 1.0, defaults.get("Back Office", 0.2 if from_role=="Provider" else 0.0), 0.05, key=f"r_{from_role}_bo")
            with c5:
                to_done = st.number_input(f"to Done ({from_role})", 0.0, 1.0, defaults.get(DONE, 1.0 if from_role in ["Provider","Back Office"] else 0.0), 0.05, key=f"r_{from_role}_done")
            route[from_role] = {"Front Desk": to_fd, "Nurse": to_nu, "Provider": to_pr, "Back Office": to_bo, DONE: to_done}

        # Defaults that mirror your previous flow loosely
        route_row_ui("Front Desk", {"Nurse": 0.6, "Provider": 0.0, "Back Office": 0.0, DONE: 0.4})
        route_row_ui("Nurse", {"Provider": 0.5, DONE: 0.5})
        route_row_ui("Provider", {"Back Office": 0.2, DONE: 0.8})
        route_row_ui("Back Office", {DONE: 1.0})

        submitted = st.form_submit_button("Continue →", use_container_width=True)
        if submitted:
            open_minutes = int(open_hours * MIN_PER_HOUR)
            sim_minutes = int(sim_days * DAY_MIN)

            st.session_state["design"] = dict(
                sim_minutes=sim_minutes,
                open_minutes=open_minutes,
                # staffing
                frontdesk_cap=fd_cap, nurse_cap=nurse_cap, provider_cap=provider_cap, backoffice_cap=bo_cap,
                # arrivals by role
                arrivals_per_hour_by_role={
                    "Front Desk": arr_fd,
                    "Nurse": arr_nu,
                    "Provider": arr_pr,
                    "Back Office": arr_bo
                },
                # service
                svc_frontdesk=svc_frontdesk, svc_nurse_protocol=svc_nurse_protocol, svc_nurse=svc_nurse,
                svc_provider=svc_provider, svc_backoffice=svc_backoffice,
                dist_role={"Front Desk":"normal","NurseProtocol":"normal","Nurse":"exponential","Provider":"exponential","Back Office":"exponential"},
                cv_speed=cv_speed,
                emr_overhead={"Front Desk":0.5,"Nurse":0.5,"NurseProtocol":0.5,"Provider":0.5,"Back Office":0.5},
                # loops
                p_fd_insuff=p_fd_insuff, max_fd_loops=max_fd_loops, fd_loop_delay=fd_loop_delay,
                p_nurse_insuff=p_nurse_insuff, max_nurse_loops=max_nurse_loops,
                p_provider_insuff=p_provider_insuff, max_provider_loops=max_provider_loops, provider_loop_delay=provider_loop_delay,
                p_backoffice_insuff=p_backoffice_insuff, max_backoffice_loops=max_backoffice_loops, backoffice_loop_delay=backoffice_loop_delay,
                # misrouting
                p_wrong_at_arrival=p_wrong_at_arrival,
                p_wrong_between=p_wrong_between,
                # routing matrix (raw; normalize during run)
                route_matrix=route,
                # policy toggles
                require_provider_signoff=require_provider_signoff,
                admin_possible_without_fd=False,               # keep your previous default
                force_backoffice_after_provider=False         # keep your previous default
            )
            st.success("Design saved. Continue to run the simulation.")
            go_next()

# --- Safe defaults for older saved designs / partial configs ---
def normalize_params(p: dict) -> dict:
    # Core lists / constants
    ROLES = ["Front Desk", "Nurse", "Provider", "Back Office"]
    DONE  = "Done"

    # Arrivals-by-role (fallback to old single arrival rate → all to Front Desk)
    if "arrivals_per_hour_by_role" not in p:
        rate = float(p.get("arrivals_per_hour", 12.0))  # legacy knob if present
        p["arrivals_per_hour_by_role"] = {
            "Front Desk": rate, "Nurse": 0.0, "Provider": 0.0, "Back Office": 0.0
        }
    else:
        # ensure all roles present
        for r in ROLES:
            p["arrivals_per_hour_by_role"].setdefault(r, 0.0)

    # Routing matrix (normalize later, but ensure presence & DONE column)
    if "route_matrix" not in p:
        p["route_matrix"] = {
            "Front Desk": {"Nurse": 0.6, "Provider": 0.0, "Back Office": 0.0, DONE: 0.4},
            "Nurse": {"Provider": 0.5, DONE: 0.5},
            "Provider": {"Back Office": 0.2, DONE: 0.8},
            "Back Office": {DONE: 1.0},
        }
    for r in ROLES:
        p["route_matrix"].setdefault(r, {})
        p["route_matrix"][r].setdefault(DONE, 1.0 if r in ["Provider", "Back Office"] else 0.0)
        for c in ROLES:
            p["route_matrix"][r].setdefault(c, 0.0)

    # Misrouting params
    p.setdefault("p_wrong_at_arrival", 0.0)
    p.setdefault("p_wrong_between", 0.0)

    # Policies
    p.setdefault("require_provider_signoff", False)
    p.setdefault("admin_possible_without_fd", False)
    p.setdefault("force_backoffice_after_provider", False)

    # Distributions / overheads (used by scheduled_service)
    p.setdefault("dist_role", {
        "Front Desk":"normal","NurseProtocol":"normal","Nurse":"exponential",
        "Provider":"exponential","Back Office":"exponential"
    })
    p.setdefault("cv_speed", 0.25)
    p.setdefault("emr_overhead", {
        "Front Desk":0.5,"Nurse":0.5,"NurseProtocol":0.5,"Provider":0.5,"Back Office":0.5
    })

    # Loop knobs (ensure all present)
    for key, default in [
        ("p_fd_insuff", 0.05), ("max_fd_loops", 3), ("fd_loop_delay", 30.0),
        ("p_nurse_insuff", 0.05), ("max_nurse_loops", 2),
        ("p_provider_insuff", 0.0), ("max_provider_loops", 0), ("provider_loop_delay", 15.0),
        ("p_backoffice_insuff", 0.0), ("max_backoffice_loops", 0), ("backoffice_loop_delay", 15.0),
    ]:
        p.setdefault(key, default)

    # Service means (ensure present)
    for key, default in [
        ("svc_frontdesk", 3.0), ("svc_nurse_protocol", 2.0), ("svc_nurse", 4.0),
        ("svc_provider", 6.0), ("svc_backoffice", 5.0),
    ]:
        p.setdefault(key, default)

    # Staffing / hours (defensive)
    p.setdefault("frontdesk_cap", 1)
    p.setdefault("nurse_cap", 1)
    p.setdefault("provider_cap", 1)
    p.setdefault("backoffice_cap", 1)
    p.setdefault("open_minutes", 8 * 60)
    p.setdefault("sim_minutes", 24 * 60)

    return p


# -------- STEP 2: RUN & RESULTS --------
else:
    st.subheader("Step 2 — Run & Results")
    st.button("← Back to Design", on_click=go_back)

    if not st.session_state["design"]:
        st.info("Use **Continue** on Step 1 first.")
        st.stop()

    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    run = st.button("Run Simulation", type="primary", use_container_width=True)

    if run:
        random.seed(seed)
        np.random.seed(seed)

        p = normalize_params(dict(st.session_state["design"]))  # copy + fill missing keys
        metrics = Metrics()
        env = simpy.Environment()
        system = CHCSystem(env, p, metrics)

        # Start arrival processes (independent Poisson for each role)
        for role in ROLES:
            rate = p["arrivals_per_hour_by_role"].get(role, 0.0)
            env.process(arrival_process_for_role(env, system, role, rate))

        env.process(monitor(env, system))
        env.run(until=p["sim_minutes"])

        # Utilizations (open-time adjusted)
        open_time_available = effective_open_minutes(p["sim_minutes"], p["open_minutes"])
        denom = {
            "Front Desk": max(1, p["frontdesk_cap"]) * open_time_available,
            "Nurse": max(1, p["nurse_cap"]) * open_time_available,
            "Provider": max(1, p["provider_cap"]) * open_time_available,
            "Back Office": max(1, p["backoffice_cap"]) * open_time_available
        }
        util = {r: metrics.service_time_sum[r] / max(1, denom[r]) for r in ROLES}
        util_overall = np.mean(list(util.values()))

        # ---- Metrics tables (no graphs) ----
        st.markdown("### Simulation Metrics")

        # Utilization table
        util_df = pd.DataFrame({
            "Role": ["Provider", "Front Desk", "Nurse", "Back Office", "Overall"],
            "Utilization": [
                pct(min(1.0, util["Provider"])),
                pct(min(1.0, util["Front Desk"])),
                pct(min(1.0, util["Nurse"])),
                pct(min(1.0, util["Back Office"])),
                pct(min(1.0, util_overall))
            ]
        })

        # Loops table
        loop_help = (
            "A 'loop' means the task had to repeat that role’s work due to missing/insufficient information. "
            "Each loop adds delay and another service cycle."
        )
        loop_df = pd.DataFrame({
            "Role": ["Front Desk", "Nurse", "Provider", "Back Office"],
            "Loop Count": [
                metrics.loop_fd_insufficient,
                metrics.loop_nurse_insufficient,
                metrics.loop_provider_insufficient,
                metrics.loop_backoffice_insufficient,
            ]
        })

        # Misrouting table
        mis_help = (
            "Misrouted at arrival: task entered the wrong role and was immediately re-routed.\n"
            "Misrouted between roles: a handoff went to a different role than intended."
        )
        mis_df = pd.DataFrame({
            "Metric": [
                "Misrouted at arrival (count)",
                "Misrouted at arrival (rate)",
                "Misrouted between roles (count)",
                "Misrouted between roles (rate)"
            ],
            "Value": [
                metrics.misrouted_at_arrival,
                pct(metrics.misrouted_at_arrival / max(1, metrics.arrivals_total)),
                metrics.misrouted_between,
                # approximate denom for between-misroutes: total handoffs ~ taps minus first touches (rough)
                pct(metrics.misrouted_between / max(1, metrics.misrouted_between + metrics.completed))
            ]
        })

        c1, c2 = st.columns([1,1])
        with c1:
            st.markdown("#### Utilization (%)")
            st.dataframe(util_df, use_container_width=True)
        with c2:
            st.markdown("#### Loops ", help=loop_help)
            st.dataframe(loop_df, use_container_width=True)

        st.markdown("#### Misrouting ", help=mis_help)
        st.dataframe(mis_df, use_container_width=True)

        # Persist minimal results
        st.session_state["results"] = dict(util_df=util_df, loop_df=loop_df, mis_df=mis_df)
