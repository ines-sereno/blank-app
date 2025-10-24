from typing import Dict, List
import streamlit as st
import simpy
import random
import numpy as np
import math
import pandas as pd

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
class CHCSystem:
    def __init__(self, env, params, metrics):
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

    def scheduled_service(self, resource, role_account, mean_time, role_for_dist=None):
        """
        Respect clinic hours; if resource is None (capacity 0), service is skipped.
        """
        if resource is None or mean_time <= 1e-12:
            return
        if role_for_dist is None:
            role_for_dist = role_account

        remaining = draw_service_time(role_for_dist, mean_time, self.p["dist_role"], self.p["cv_speed"])
        remaining += max(0.0, self.p["emr_overhead"].get(role_account, 0.0))

        # micro-optimization: local reference
        open_minutes = self.p["open_minutes"]

        while remaining > 1e-9:
            if not is_open(self.env.now, open_minutes):
                yield self.env.timeout(minutes_until_open(self.env.now, open_minutes))
            window = minutes_until_close(self.env.now, open_minutes)
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
                # optional: could auto-complete here; we defer to routing matrix for consistency
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
if "design_saved" not in st.session_state:
    st.session_state.design_saved = False

def go_next():
    st.session_state.wizard_step = min(2, st.session_state.wizard_step + 1)
def go_back():
    st.session_state.wizard_step = max(1, st.session_state.wizard_step - 1)

# --- Locale-agnostic probability input (always dot-decimal) ---
def prob_input(
    label: str,
    key: str,
    default: float = 0.0,
    help: str | None = None,
    disabled: bool = False,
) -> float:
    """Probability input for use inside st.form; no Enter needed, dot-decimal enforced on read."""
    if key not in st.session_state:
        st.session_state[key] = f"{float(default):.2f}"

    raw = st.text_input(
        label,
        value=st.session_state[key],
        key=key,
        help=help,
        disabled=disabled,
    )
    try:
        val = float(str(raw).replace(",", "."))
    except ValueError:
        val = float(default)
    val = max(0.0, min(1.0, val))
    st.caption(f"{val:.2f}")
    return val


# -------- STEP 1: DESIGN --------
if st.session_state.wizard_step == 1:

    st.markdown("""
    ### 🏥 **Design Your Clinic**

    This tool simulates how tasks flow through a Community Health Center (CHC) using a **Discrete-Event Simulation (DES)** model.  
    Each task represents a patient request or administrative action that moves through roles like the **Front Desk**, **Nurse/MA**, **Provider**, and **Back Office**.

    The simulation models:
    - **Arrivals** — how frequently tasks enter the system at each role.  
    - **Service times** — how long each role takes to complete its task.  
    - **Rework loops** — when tasks must be repeated due to missing or incorrect information.
    - **Routing probabilities** — how likely tasks are to move between roles or be completed.    
    
    You can customize your clinic’s staffing levels, arrival rates, and workflow logic below.  
    When you click **Save**, your configuration will be used to simulate a typical clinic day and measure utilization and rework rates.
    """)

    # --- Staffing MOVED INTO MAIN SCREEN (inside the form) ---
    with st.form("design_form", clear_on_submit=False):
        st.markdown("### 👥 Staffing (on duty)")
        cStaff1, cStaff2, cStaff3, cStaff4 = st.columns(4)
        with cStaff1:
            fd_cap = st.number_input(
                "Front Desk staff", min_value=0, max_value=50, value=3, step=1, format="%d",
                help="Number of front desk staff simultaneously available."
            )
        with cStaff2:
            nurse_cap = st.number_input(
                "Nurses / MAs", min_value=0, max_value=50, value=2, step=1, format="%d",
                help="Number of nurses/medical assistants on duty."
            )
        with cStaff3:
            provider_cap = st.number_input(
                "Providers", min_value=0, max_value=50, value=1, step=1, format="%d",
                help="Number of providers on duty."
            )
        with cStaff4:
            bo_cap = st.number_input(
                "Back Office staff", min_value=0, max_value=50, value=1, step=1, format="%d",
                help="Number of back-office staff on duty."
            )

        # Flags to grey out dependent controls instantly (computed AFTER staffing)
        fd_off = (fd_cap == 0)
        nu_off = (nurse_cap == 0)
        pr_off = (provider_cap == 0)
        bo_off = (bo_cap == 0)

        c1, c2 = st.columns([1,1])
        with c1:
            sim_days = st.number_input(
                "Days to simulate", min_value=1, max_value=30, value=5, step=1, format="%d",
                help="Number of 24-hour days to include in the simulation."
            )
            open_hours = st.number_input(
                "Hours open per day", min_value=1, max_value=24, value=10, step=1, format="%d",
                help="Clinic operating hours per day during which work can be performed."
            )
            cv_speed = st.slider(
                "Task speed variability (CV)", 0.0, 0.8, 0.25, 0.05,
                help="How variable individual task times are around their mean (coefficient of variation)."
            )

        st.markdown("### Arrivals per hour at each role")
        cA1, cA2, cA3, cA4 = st.columns(4)
        with cA1:
            arr_fd = st.number_input("→ Front Desk", min_value=0, max_value=500, value=15, step=1, format="%d",
                                     help="Average number of tasks arriving to the Front Desk each hour.",
                                     disabled=fd_off)
        with cA2:
            arr_nu = st.number_input("→ Nurse / MAs", min_value=0, max_value=500, value=20, step=1, format="%d",
                                     help="Average number of tasks arriving directly to the Nurse/MA queue per hour.",
                                     disabled=nu_off)
        with cA3:
            arr_pr = st.number_input("→ Provider", min_value=0, max_value=500, value=10, step=1, format="%d",
                                     help="Average number of tasks arriving directly to the Provider per hour.",
                                     disabled=pr_off)
        with cA4:
            arr_bo = st.number_input("→ Back Office", min_value=0, max_value=500, value=5, step=1, format="%d",
                                     help="Average number of tasks arriving directly to the Back Office per hour.",
                                     disabled=bo_off)

        with st.expander("Additional (optional) — service times, loops & interaction matrix", expanded=False):
            st.markdown("#### Service times (mean minutes)")
            cS1, cS2 = st.columns(2)
            with cS1:
                svc_frontdesk = st.slider("Front Desk", 0.0, 30.0, 3.0, 0.5,
                                          help="Average time to process a task at the Front Desk.",
                                          disabled=fd_off)
                svc_nurse_protocol = st.slider("Nurse Protocol", 0.0, 30.0, 2.0, 0.5,
                                               help="Average time when a task is handled entirely by standing nurse protocol.",
                                               disabled=nu_off)
                svc_nurse = st.slider("Nurse (non-protocol)", 0.0, 40.0, 4.0, 0.5,
                                      help="Average time for a standard nurse/MA task (non-protocol).",
                                      disabled=nu_off)
            with cS2:
                svc_provider = st.slider("Provider", 0.0, 60.0, 6.0, 0.5,
                                         help="Average time for a provider to complete their part of a task.",
                                         disabled=pr_off)
                svc_backoffice = st.slider("Back Office", 0.0, 60.0, 5.0, 0.5,
                                           help="Average time for back-office processing.",
                                           disabled=bo_off)
                p_protocol = st.slider("Probability Nurse resolves via protocol", 0.0, 1.0, 0.40, 0.05,
                                       help="Chance that the nurse protocol resolves the task without needing the provider.",
                                       disabled=nu_off)

            st.markdown("#### Rework Loop Probabilities")
            cL1, cL2 = st.columns(2)
            with cL1:
                p_fd_insuff = st.slider("Front Desk loop chance", 0.0, 0.6, 0.05, 0.01,
                                        help="Chance a Front Desk task needs to loop for missing information.",
                                        disabled=fd_off)
                max_fd_loops = st.slider("Max Front Desk loops", 0, 10, 3, 1,
                                         help="Maximum number of Front Desk rework loops per task.",
                                         disabled=fd_off)
                fd_loop_delay = st.slider("Front Desk loop delay (min)", 0.0, 240.0, 30.0, 5.0,
                                          help="Delay spent collecting info before reworking at Front Desk.",
                                          disabled=fd_off)
                p_nurse_insuff = st.slider("Nurse loop chance", 0.0, 0.6, 0.05, 0.01,
                                           help="Chance the nurse sends the task back for clarification.",
                                           disabled=nu_off)
                max_nurse_loops = st.slider("Max Nurse loops", 0, 10, 2, 1,
                                            help="Maximum number of Nurse rework loops per task.",
                                            disabled=nu_off)
            with cL2:
                p_provider_insuff = st.slider("Provider loop chance", 0.0, 0.6, 0.00, 0.01,
                                              help="Chance a Provider needs to rework the same task.",
                                              disabled=pr_off)
                max_provider_loops = st.slider("Max Provider loops", 0, 10, 1, 1,
                                               help="Maximum number of Provider rework loops per task.",
                                               disabled=pr_off)
                provider_loop_delay = st.slider("Provider loop delay (min)", 0.0, 240.0, 15.0, 5.0,
                                                help="Delay before provider rework can occur.",
                                                disabled=pr_off)
                p_backoffice_insuff = st.slider("Back Office loop chance", 0.0, 0.6, 0.00, 0.01,
                                                help="Chance Back Office needs to rework the same task.",
                                                disabled=bo_off)
                max_backoffice_loops = st.slider("Max Back Office loops", 0, 10, 1, 1,
                                                 help="Maximum number of Back Office rework loops per task.",
                                                 disabled=bo_off)
                backoffice_loop_delay = st.slider("Back Office loop delay (min)", 0.0, 240.0, 15.0, 5.0,
                                                  help="Delay before back-office rework can occur.",
                                                  disabled=bo_off)

            # ----- Interaction matrix (routing) -----
            st.markdown("#### Interaction matrix - Routing Probabilities")
            st.caption("For each role, set probabilities of sending the task to another role or Done.")

            route: Dict[str, Dict[str, float]] = {}

            def route_row_ui(from_role: str, defaults: Dict[str, float], disabled: bool = False) -> Dict[str, float]:
                st.markdown(f"**{from_role} →**")
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    to_fd = prob_input(
                        f"to FD ({from_role})",
                        key=f"r_{from_role}_fd",
                        default=float(defaults.get("Front Desk", 0.2)),
                        help="Probability to route next to Front Desk.",
                        disabled=disabled,
                    )
                with c2:
                    to_nu = prob_input(
                        f"to Nurse ({from_role})",
                        key=f"r_{from_role}_nu",
                        default=float(defaults.get("Nurse", 0.4)),
                        help="Probability to route next to Nurse/MA.",
                        disabled=disabled,
                    )
                with c3:
                    to_pr = prob_input(
                        f"to Provider ({from_role})",
                        key=f"r_{from_role}_pr",
                        default=float(defaults.get("Provider", 0.2)),
                        help="Probability to route next to Provider.",
                        disabled=disabled,
                    )
                with c4:
                    to_bo = prob_input(
                        f"to Back Office ({from_role})",
                        key=f"r_{from_role}_bo",
                        default=float(defaults.get("Back Office", 0.5)),
                        help="Probability to route next to Back Office.",
                        disabled=disabled,
                    )
                with c5:
                    to_done = prob_input(
                        f"to Done ({from_role})",
                        key=f"r_{from_role}_done",
                        default=float(defaults.get(DONE, 0.2)),
                        help="Probability the task finishes after this role.",
                        disabled=disabled,
                    )
                return {
                    "Front Desk": to_fd,
                    "Nurse": to_nu,
                    "Provider": to_pr,
                    "Back Office": to_bo,
                    DONE: to_done,
                }

            # 👇 Calls kept INSIDE the expander so routing stays under the dropdown
            route["Front Desk"]  = route_row_ui("Front Desk",  {"Nurse": 0.6, DONE: 0.4}, disabled=fd_off)
            route["Nurse"]       = route_row_ui("Nurse",       {"Provider": 0.5, DONE: 0.5}, disabled=nu_off)
            route["Provider"]    = route_row_ui("Provider",    {"Back Office": 0.2, DONE: 0.8}, disabled=pr_off)
            route["Back Office"] = route_row_ui("Back Office", {DONE: 1.0},                      disabled=bo_off)

        # --- Save button INSIDE the form ---
        saved = st.form_submit_button("Save", use_container_width=True)

        if saved:
            open_minutes = int(open_hours * MIN_PER_HOUR)
            sim_minutes = int(sim_days * DAY_MIN)

            # build and store the design dict
            st.session_state["design"] = dict(
                sim_minutes=sim_minutes,
                open_minutes=open_minutes,
                # staffing (now from main form)
                frontdesk_cap=fd_cap, nurse_cap=nurse_cap, provider_cap=provider_cap, backoffice_cap=bo_cap,
                # arrivals by role (integers)
                arrivals_per_hour_by_role={
                    "Front Desk": int(arr_fd),
                    "Nurse":      int(arr_nu),
                    "Provider":   int(arr_pr),
                    "Back Office":int(arr_bo),
                },
                # service + dist + overheads
                svc_frontdesk=svc_frontdesk, svc_nurse_protocol=svc_nurse_protocol, svc_nurse=svc_nurse,
                svc_provider=svc_provider,   svc_backoffice=svc_backoffice,
                dist_role={"Front Desk":"normal","NurseProtocol":"normal","Nurse":"exponential","Provider":"exponential","Back Office":"exponential"},
                cv_speed=cv_speed,
                emr_overhead={"Front Desk":0.5,"Nurse":0.5,"NurseProtocol":0.5,"Provider":0.5,"Back Office":0.5},
                # loops
                p_fd_insuff=p_fd_insuff, max_fd_loops=max_fd_loops, fd_loop_delay=fd_loop_delay,
                p_nurse_insuff=p_nurse_insuff, max_nurse_loops=max_nurse_loops,
                p_provider_insuff=p_provider_insuff, max_provider_loops=max_provider_loops, provider_loop_delay=provider_loop_delay,
                p_backoffice_insuff=p_backoffice_insuff, max_backoffice_loops=max_backoffice_loops, backoffice_loop_delay=backoffice_loop_delay,
                # nurse protocol + routing matrix
                p_protocol=p_protocol,
                route_matrix=route
            )

            st.session_state.design_saved = True
            st.success("Configuration saved.")

    # --- Continue button OUTSIDE the form ---
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

    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1, format="%d",
                           help="Seed for the random number generator to reproduce results.")
    run = st.button("Run Simulation", type="primary", use_container_width=True)

    if run:
        random.seed(seed)
        np.random.seed(seed)

        p = st.session_state["design"]
        metrics = Metrics()
        env = simpy.Environment()
        system = CHCSystem(env, p, metrics)

        # Start independent arrival processes for each role
        for role in ROLES:
            rate = int(p["arrivals_per_hour_by_role"].get(role, 0))
            env.process(arrival_process_for_role(env, system, role, rate))

        env.process(monitor(env, system))
        env.run(until=p["sim_minutes"])

        # ---- Utilizations (open-time adjusted) ----
        open_time_available = effective_open_minutes(p["sim_minutes"], p["open_minutes"])
        denom = {
            "Front Desk": max(1, p["frontdesk_cap"]) * open_time_available,
            "Nurse":      max(1, p["nurse_cap"])      * open_time_available,
            "Provider":   max(1, p["provider_cap"])   * open_time_available,
            "Back Office":max(1, p["backoffice_cap"]) * open_time_available,
        }
        util = {r: metrics.service_time_sum[r] / max(1, denom[r]) for r in ["Front Desk","Nurse","Provider","Back Office"]}
        util_overall = np.mean(list(util.values()))

        # Only show roles with capacity > 0
        active_roles_caps = [
            ("Provider",    p["provider_cap"]),
            ("Front Desk",  p["frontdesk_cap"]),
            ("Nurse",       p["nurse_cap"]),
            ("Back Office", p["backoffice_cap"]),
        ]
        active_roles = [r for r, cap in active_roles_caps if cap > 0]

        # Utilization table (filtered)
        util_rows = [{"Role": r, "Utilization": pct(min(1.0, util[r]))} for r in active_roles]
        util_rows.append({"Role": "Overall", "Utilization": pct(min(1.0, util_overall))})
        util_df = pd.DataFrame(util_rows)

        # Loops table (filtered)
        loop_help = "A 'loop' is a rework cycle at that role caused by missing or insufficient information."
        loop_counts = {
            "Front Desk":  metrics.loop_fd_insufficient,
            "Nurse":       metrics.loop_nurse_insufficient,
            "Provider":    metrics.loop_provider_insufficient,
            "Back Office": metrics.loop_backoffice_insufficient,
        }
        loop_df = pd.DataFrame(
            [{"Role": r, "Loop Count": loop_counts[r]} for r in active_roles if r in loop_counts]
        )

        # ---- Render tables ----
        c1, c2 = st.columns([1,1])
        with c1:
            st.markdown("#### Utilization (%)")
            st.dataframe(util_df, use_container_width=True)
        with c2:
            st.markdown("#### Loops ", help=loop_help)
            st.dataframe(loop_df, use_container_width=True)

        # Persist minimal results
        st.session_state["results"] = dict(util_df=util_df, loop_df=loop_df)
