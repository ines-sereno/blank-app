import streamlit as st
import simpy
import random
import numpy as np
import math
import pandas as pd
from typing import Dict

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
class Metrics:
    def __init__(self):
        self.time_stamps = []
        self.queues = {"Front Desk": [], "Nurse": [], "Provider": [], "Back Office": []}
        self.waits = {"Front Desk": [], "Nurse": [], "Provider": [], "Back Office": []}
        self.taps = {"Front Desk": 0, "Nurse": 0, "Provider": 0, "Back Office": 0}
        self.completed = 0
        self.resolved_admin = 0
        self.resolved_protocol = 0
        self.routed_backoffice = 0
        self.arrivals = 0
        self.service_time_sum = {"Front Desk": 0.0, "Nurse": 0.0, "Provider": 0.0, "Back Office": 0.0}
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
        self.frontdesk = simpy.Resource(env, capacity=params["frontdesk_cap"])
        self.nurse = simpy.Resource(env, capacity=params["nurse_cap"])
        self.provider = simpy.Resource(env, capacity=params["provider_cap"])
        self.backoffice = simpy.Resource(env, capacity=params["backoffice_cap"])

    def scheduled_service(self, resource, role_account, mean_time, role_for_dist=None):
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
# Processes
# =============================
def arrival_process(env, system):
    i = 0
    while True:
        lam = system.p["arrivals_per_hour"] / 60.0
        inter = random.expovariate(lam) if lam > 0 else 999999
        yield env.timeout(inter)
        i += 1
        task_id = f"T-{i:05d}"
        system.m.arrivals += 1
        system.m.task_arrival_time[task_id] = env.now
        system.m.log(env.now, task_id, "ARRIVE", "New task (email/phone/message)", arrival_t=env.now)
        env.process(task_workflow(env, task_id, system))

def task_workflow(env, task_id, s: CHCSystem):
    fd_loops = 0
    nurse_loops = 0
    provider_loops = 0
    bo_loops = 0

    # Front Desk
    s.m.log(env.now, task_id, "FD_QUEUE", "")
    yield from s.scheduled_service(s.frontdesk, "Front Desk", s.p["svc_frontdesk"])
    s.m.log(env.now, task_id, "FD_DONE", "")

    # Front Desk loops
    while (fd_loops < s.p["max_fd_loops"]) and (random.random() < s.p["p_fd_insuff"]):
        fd_loops += 1
        s.m.loop_fd_insufficient += 1
        s.m.log(env.now, task_id, "FD_INSUFF", f"Missing info loop #{fd_loops}")
        yield env.timeout(s.p["fd_loop_delay"])
        s.m.log(env.now, task_id, "FD_RETRY_QUEUE", f"Loop #{fd_loops}")
        yield from s.scheduled_service(s.frontdesk, "Front Desk", s.p["svc_frontdesk"])
        s.m.log(env.now, task_id, "FD_RETRY_DONE", f"Loop #{fd_loops}")

    # Resolved administratively?
    if random.random() < s.p["p_admin"]:
        s.m.resolved_admin += 1
        s.m.completed += 1
        s.m.task_completion_time[task_id] = env.now
        s.m.log(env.now, task_id, "DONE", "Resolved administratively")
        return

    # Nurse
    s.m.log(env.now, task_id, "NU_QUEUE", "")
    if random.random() < s.p["p_protocol"]:
        yield from s.scheduled_service(s.nurse, "Nurse", s.p["svc_nurse_protocol"], role_for_dist="NurseProtocol")
        s.m.resolved_protocol += 1
        s.m.completed += 1
        s.m.task_completion_time[task_id] = env.now
        s.m.log(env.now, task_id, "DONE", "Resolved by nurse protocol")
        return
    else:
        yield from s.scheduled_service(s.nurse, "Nurse", s.p["svc_nurse"])
        s.m.log(env.now, task_id, "NU_DONE", "")

    # Nurse loops (back to FD then re-check Nurse)
    while (nurse_loops < s.p["max_nurse_loops"]) and (random.random() < s.p["p_nurse_insuff"]):
        nurse_loops += 1
        s.m.loop_nurse_insufficient += 1
        s.m.log(env.now, task_id, "NU_INSUFF", f"Back to FD loop #{nurse_loops}")
        s.m.log(env.now, task_id, "FD_QUEUE", f"After nurse loop #{nurse_loops}")
        yield from s.scheduled_service(s.frontdesk, "Front Desk", s.p["svc_frontdesk"])
        s.m.log(env.now, task_id, "FD_DONE", f"After nurse loop #{nurse_loops}")
        s.m.log(env.now, task_id, "NU_RECHECK_QUEUE", f"Loop #{nurse_loops}")
        yield from s.scheduled_service(s.nurse, "Nurse", max(0.0, 0.5 * s.p["svc_nurse"]))
        s.m.log(env.now, task_id, "NU_RECHECK_DONE", f"Loop #{nurse_loops}")

    # Provider
    s.m.log(env.now, task_id, "PR_QUEUE", "")
    yield from s.scheduled_service(s.provider, "Provider", s.p["svc_provider"])
    s.m.log(env.now, task_id, "PR_DONE", "")

    # Provider loops (rework at Provider)
    while (provider_loops < s.p["max_provider_loops"]) and (random.random() < s.p["p_provider_insuff"]):
        provider_loops += 1
        s.m.loop_provider_insufficient += 1
        s.m.log(env.now, task_id, "PR_INSUFF", f"Provider rework loop #{provider_loops}")
        yield env.timeout(s.p["provider_loop_delay"])
        s.m.log(env.now, task_id, "PR_RECHECK_QUEUE", f"Loop #{provider_loops}")
        yield from s.scheduled_service(s.provider, "Provider", max(0.0, 0.5 * s.p["svc_provider"]))
        s.m.log(env.now, task_id, "PR_RECHECK_DONE", f"Loop #{provider_loops}")

    # Back Office (optional)
    if random.random() < s.p["p_backoffice"] or s.p.get("force_backoffice", False):
        s.m.log(env.now, task_id, "BO_QUEUE", "")
        yield from s.scheduled_service(s.backoffice, "Back Office", s.p["svc_backoffice"])
        s.m.routed_backoffice += 1
        s.m.log(env.now, task_id, "BO_DONE", "")

        # Back Office loops (rework at Back Office)
        while (bo_loops < s.p["max_backoffice_loops"]) and (random.random() < s.p["p_backoffice_insuff"]):
            bo_loops += 1
            s.m.loop_backoffice_insufficient += 1
            s.m.log(env.now, task_id, "BO_INSUFF", f"Back Office rework loop #{bo_loops}")
            yield env.timeout(s.p["backoffice_loop_delay"])
            s.m.log(env.now, task_id, "BO_RECHECK_QUEUE", f"Loop #{bo_loops}")
            yield from s.scheduled_service(s.backoffice, "Back Office", max(0.0, 0.5 * s.p["svc_backoffice"]))
            s.m.log(env.now, task_id, "BO_RECHECK_DONE", f"Loop #{bo_loops}")

    s.m.completed += 1
    s.m.task_completion_time[task_id] = env.now
    s.m.log(env.now, task_id, "DONE", "Task completed")

def monitor(env, s: CHCSystem):
    while True:
        s.m.time_stamps.append(env.now)
        s.m.queues["Front Desk"].append(len(s.frontdesk.queue))
        s.m.queues["Nurse"].append(len(s.nurse.queue))
        s.m.queues["Provider"].append(len(s.provider.queue))
        s.m.queues["Back Office"].append(len(s.backoffice.queue))
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
            arrivals_per_hour = st.number_input("Average task arrivals per hour", min_value=1, max_value=120, value=12, step=1)
            open_hours = st.number_input("Hours open per day", min_value=1, max_value=24, value=8, step=1)
        with c2:
            st.markdown("**Staffing (on duty)**")
            fd_cap = st.number_input("Front Desk staff", min_value=0, max_value=50, value=1, step=1)
            nurse_cap = st.number_input("Nurses / MAs", min_value=0, max_value=50, value=1, step=1)
            provider_cap = st.number_input("Providers", min_value=0, max_value=50, value=1, step=1)
            bo_cap = st.number_input("Back Office staff", min_value=0, max_value=50, value=1, step=1)

        with st.expander("Advanced (optional) — Routing, times, variability & loops", expanded=False):
            c3, c4, c5 = st.columns(3)
            with c3:
                p_admin = st.slider("Resolved at Front Desk", 0.0, 1.0, 0.30, 0.05)
                p_protocol = st.slider("Resolved by Nurse protocol", 0.0, 1.0, 0.40, 0.05)
                p_backoffice = st.slider("Requires Back Office after Provider", 0.0, 1.0, 0.20, 0.05)
            with c4:
                svc_frontdesk = st.slider("Mean service time — Front Desk (min)", 0.0, 30.0, 3.0, 0.5)
                svc_nurse_protocol = st.slider("Mean service time — Nurse Protocol (min)", 0.0, 30.0, 2.0, 0.5)
                svc_nurse = st.slider("Mean service time — Nurse (min)", 0.0, 40.0, 4.0, 0.5)
            with c5:
                svc_provider = st.slider("Mean service time — Provider (min)", 0.0, 60.0, 6.0, 0.5)
                svc_backoffice = st.slider("Mean service time — Back Office (min)", 0.0, 60.0, 5.0, 0.5)
                cv_speed = st.slider("Task speed variability (CV)", 0.0, 0.8, 0.25, 0.05)

            st.markdown("**Loop settings**")
            c6, c7 = st.columns(2)
            with c6:
                p_fd_insuff = st.slider("Chance Front Desk bounce", 0.0, 0.6, 0.05, 0.01)
                max_fd_loops = st.slider("Max Front Desk loops", 0, 10, 3)
                fd_loop_delay = st.slider("Front Desk loop delay (min)", 0.0, 240.0, 30.0, 5.0)
                p_nurse_insuff = st.slider("Chance Nurse needs more info", 0.0, 0.6, 0.05, 0.01)
                max_nurse_loops = st.slider("Max Nurse loops", 0, 10, 2)
            with c7:
                p_provider_insuff = st.slider("Chance Provider rework", 0.0, 0.6, 0.00, 0.01)  # default 0
                max_provider_loops = st.slider("Max Provider loops", 0, 10, 1)
                provider_loop_delay = st.slider("Provider loop delay (min)", 0.0, 240.0, 15.0, 5.0)
                p_backoffice_insuff = st.slider("Chance Back Office rework", 0.0, 0.6, 0.00, 0.01)  # default 0
                max_backoffice_loops = st.slider("Max Back Office loops", 0, 10, 1)
                backoffice_loop_delay = st.slider("Back Office loop delay (min)", 0.0, 240.0, 15.0, 5.0)

        submitted = st.form_submit_button("Continue →", use_container_width=True)
        if submitted:
            open_minutes = int(open_hours * MIN_PER_HOUR)
            sim_minutes = int(sim_days * DAY_MIN)

            # Defaults if Advanced not touched (just in case)
            defaults = dict(
                p_admin=0.30, p_protocol=0.40, p_backoffice=0.20,
                svc_frontdesk=3.0, svc_nurse_protocol=2.0, svc_nurse=4.0, svc_provider=6.0, svc_backoffice=5.0,
                cv_speed=0.25,
                p_fd_insuff=0.05, max_fd_loops=3, fd_loop_delay=30.0,
                p_nurse_insuff=0.05, max_nurse_loops=2,
                p_provider_insuff=0.0, max_provider_loops=0, provider_loop_delay=15.0,
                p_backoffice_insuff=0.0, max_backoffice_loops=0, backoffice_loop_delay=15.0
            )
            for k, v in defaults.items():
                if k not in locals():
                    locals()[k] = v

            st.session_state["design"] = dict(
                sim_minutes=sim_minutes,
                arrivals_per_hour=arrivals_per_hour,
                open_minutes=open_minutes,
                frontdesk_cap=fd_cap, nurse_cap=nurse_cap, provider_cap=provider_cap, backoffice_cap=bo_cap,
                p_admin=p_admin, p_protocol=p_protocol, p_backoffice=p_backoffice,
                svc_frontdesk=svc_frontdesk, svc_nurse_protocol=svc_nurse_protocol, svc_nurse=svc_nurse,
                svc_provider=svc_provider, svc_backoffice=svc_backoffice,
                dist_role={"Front Desk":"normal","NurseProtocol":"normal","Nurse":"exponential","Provider":"exponential","Back Office":"exponential"},
                cv_speed=cv_speed,
                emr_overhead={"Front Desk":0.5,"Nurse":0.5,"NurseProtocol":0.5,"Provider":0.5,"Back Office":0.5},
                # loops
                p_fd_insuff=p_fd_insuff, max_fd_loops=max_fd_loops, fd_loop_delay=fd_loop_delay,
                p_nurse_insuff=p_nurse_insuff, max_nurse_loops=max_nurse_loops,
                p_provider_insuff=p_provider_insuff, max_provider_loops=max_provider_loops, provider_loop_delay=provider_loop_delay,
                p_backoffice_insuff=p_backoffice_insuff, max_backoffice_loops=max_backoffice_loops, backoffice_loop_delay=backoffice_loop_delay
            )
            go_next()

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

        p = st.session_state["design"]
        metrics = Metrics()
        env = simpy.Environment()
        system = CHCSystem(env, p, metrics)
        env.process(arrival_process(env, system))
        env.process(monitor(env, system))
        env.run(until=p["sim_minutes"])

        # Utilizations (open-time adjusted)
        open_time_available = effective_open_minutes(p["sim_minutes"], p["open_minutes"])
        denom_fd = max(1, p["frontdesk_cap"]) * open_time_available
        denom_nu = max(1, p["nurse_cap"]) * open_time_available
        denom_pr = max(1, p["provider_cap"]) * open_time_available
        denom_bo = max(1, p["backoffice_cap"]) * open_time_available

        util_fd = metrics.service_time_sum["Front Desk"] / max(1, denom_fd)
        util_nu = metrics.service_time_sum["Nurse"] / max(1, denom_nu)
        util_pr = metrics.service_time_sum["Provider"] / max(1, denom_pr)
        util_bo = metrics.service_time_sum["Back Office"] / max(1, denom_bo)
        util_overall = np.mean([util_fd, util_nu, util_pr, util_bo])

        # Build the neat KPI table ONLY with what you asked for
        df_kpis = pd.DataFrame({
            "Metric": [
                "Provider Utilization",
                "Front Desk Utilization",
                "Nurse Utilization",
                "Back Office Utilization",
                "Overall Utilization",
                "Front Desk Loops",
                "Nurse Loops",
                "Provider Loops",
                "Back Office Loops",
            ],
            "Value": [
                pct(min(1.0, util_pr)),
                pct(min(1.0, util_fd)),
                pct(min(1.0, util_nu)),
                pct(min(1.0, util_bo)),
                pct(min(1.0, util_overall)),
                metrics.loop_fd_insufficient,
                metrics.loop_nurse_insufficient,
                metrics.loop_provider_insufficient,
                metrics.loop_backoffice_insufficient,
            ]
        })

    st.markdown("### Simulation Metrics")

    # Add an info tooltip
    st.markdown("""
        #### Loops ℹ️
        Each **loop** means a task had to be reworked at that stage because of missing information or insufficient detail.  
        They’re counted whenever the workflow repeats that role’s service (e.g., Front Desk → Nurse → back to Front Desk).  
        - Front Desk loops: Missing or incomplete info before routing onward  
        - Nurse loops: Returned for clarification and rechecked  
        - Provider / Back Office loops: Rework before final resolution
    """)

    st.dataframe(df_kpis, use_container_width=True)

    st.caption("Note: Loop counts are total occurrences across all simulated tasks.")

        # Persist the last results (if you want to export later)
        st.session_state["results"] = dict(df_kpis=df_kpis)
