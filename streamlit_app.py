import streamlit as st
import simpy
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict
import graphviz
import plotly.graph_objects as go

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
        # loops
        self.loop_fd_insufficient = 0
        self.loop_nurse_insufficient = 0
        self.looped_tasks_fd = set()
        self.looped_tasks_nurse = set()
        # events: (time_min, task_id, step_code, note, arrival_time_min)
        self.events = []
        # lifecycle
        self.task_arrival_time: Dict[str, float] = {}
        self.task_completion_time: Dict[str, float] = {}

    def log(self, t, name, step, note="", arrival_t=None):
        self.events.append((t, name, step, note, arrival_t if arrival_t is not None else self.task_arrival_time.get(name)))

# =============================
# Step labels
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
    "NU_INSUFF": "Nurse: missing info (back to Front Desk)",
    "NU_RECHECK_QUEUE": "Nurse: re-check queued",
    "NU_RECHECK_DONE": "Nurse: re-check completed",
    "PR_QUEUE": "Provider: queued",
    "PR_DONE": "Provider: completed",
    "BO_QUEUE": "Back Office: queued",
    "BO_DONE": "Back Office: completed",
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

    # Front Desk
    s.m.log(env.now, task_id, "FD_QUEUE", "")
    yield from s.scheduled_service(s.frontdesk, "Front Desk", s.p["svc_frontdesk"])
    s.m.log(env.now, task_id, "FD_DONE", "")

    # Front Desk loops
    while (fd_loops < s.p["max_fd_loops"]) and (random.random() < s.p["p_fd_insuff"]):
        fd_loops += 1
        s.m.loop_fd_insufficient += 1
        s.m.looped_tasks_fd.add(task_id)
        s.m.log(env.now, task_id, "FD_INSUFF", f"Missing info loop #{fd_loops}")
        yield env.timeout(s.p["fd_loop_delay"])
        s.m.log(env.now, task_id, "FD_RETRY_QUEUE", f"Loop #{fd_loops}")
        yield from s.scheduled_service(s.frontdesk, "Front Desk", s.p["svc_frontdesk"])
        s.m.log(env.now, task_id, "FD_RETRY_DONE", f"Loop #{fd_loops}")

    # Resolved at Front Desk?
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

    # Nurse loops back to FD + re-check
    while (nurse_loops < s.p["max_nurse_loops"]) and (random.random() < s.p["p_nurse_insuff"]):
        nurse_loops += 1
        s.m.loop_nurse_insufficient += 1
        s.m.looped_tasks_nurse.add(task_id)
        s.m.log(env.now, task_id, "NU_INSUFF", f"Back to Front Desk loop #{nurse_loops}")
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

    # Optional Back Office
    if random.random() < s.p["p_backoffice"]:
        s.m.log(env.now, task_id, "BO_QUEUE", "")
        yield from s.scheduled_service(s.backoffice, "Back Office", s.p["svc_backoffice"])
        s.m.routed_backoffice += 1
        s.m.log(env.now, task_id, "BO_DONE", "")

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
# FLOW DIAGRAMS
# =============================
def render_static_flow():
    dot = graphviz.Digraph()
    dot.attr(rankdir="LR", fontsize="12")

    dot.node("A", "Arrivals")
    dot.node("FD", "Front Desk")
    dot.node("NR", "Nurse")
    dot.node("NP", "Nurse Protocol\n(Resolved)")
    dot.node("PR", "Provider")
    dot.node("BO", "Back Office")
    dot.node("DONE", "Resolved")

    dot.edges(["AFD"])  # A -> FD
    dot.edge("FD", "DONE", label="Admin resolve")
    dot.edge("FD", "NR", label="Else → Nurse")
    dot.edge("NR", "NP", label="Protocol")
    dot.edge("NR", "PR", label="Non-protocol")
    dot.edge("PR", "BO", label="Some cases")
    dot.edge("PR", "DONE", label="Most resolve")
    dot.edge("BO", "DONE", label="Finalize")

    st.graphviz_chart(dot, use_container_width=True)

def render_sankey_from_events(df_events, metrics):
    # Derive link counts from events + tallies
    arrivals = metrics.arrivals
    fd_resolve = metrics.resolved_admin
    nurse_protocol = metrics.resolved_protocol
    pr_done = int((df_events["Step Code"] == "PR_DONE").sum())
    bo_done = int((df_events["Step Code"] == "BO_DONE").sum())
    prov_done_no_bo = max(0, pr_done - bo_done)

    # Nodes
    labels = ["Arrivals", "Front Desk", "Resolved (Admin)",
              "Nurse", "Resolved (Protocol)", "Provider",
              "Back Office", "Resolved (Final)"]
    idx = {name:i for i,name in enumerate(labels)}

    # Links (source, target, value)
    links = [
        (idx["Arrivals"], idx["Front Desk"], arrivals),
        (idx["Front Desk"], idx["Resolved (Admin)"], fd_resolve),
        (idx["Front Desk"], idx["Nurse"], max(0, arrivals - fd_resolve)),
        (idx["Nurse"], idx["Resolved (Protocol)"], nurse_protocol),
        (idx["Nurse"], idx["Provider"], pr_done),
        (idx["Provider"], idx["Back Office"], bo_done),
        (idx["Provider"], idx["Resolved (Final)"], prov_done_no_bo),
        (idx["Back Office"], idx["Resolved (Final)"], bo_done),
    ]

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=12, thickness=14),
        link=dict(
            source=[s for s,t,v in links],
            target=[t for s,t,v in links],
            value=[v for s,t,v in links],
        )
    )])
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# =============================
# Streamlit UI (Wizard)
# =============================
st.set_page_config(page_title="CHC Workflow Simulator", layout="wide")
st.title("CHC Workflow Simulator")

if "step" not in st.session_state:
    st.session_state.step = "Design"

step = st.radio(
    "Navigate",
    ["Design", "Run & Results"],
    index=0 if st.session_state.step == "Design" else 1,
    horizontal=True,
    help="Start by designing your clinic, then run the simulation."
)
st.session_state.step = step

# Keep results across steps
if "results" not in st.session_state:
    st.session_state["results"] = None

# -------- STEP 1: DESIGN --------
if step == "Design":
    st.subheader("Step 1 — Design your clinic")

    c1, c2 = st.columns([1,1])
    with c1:
        sim_hours = st.number_input("Simulation time (hours)", min_value=1, max_value=24*7, value=8, step=1)
        arrivals_per_hour = st.number_input("Average task arrivals per hour", min_value=1, max_value=120, value=12, step=1)
        open_hours = st.number_input("Hours open per day", min_value=1, max_value=24, value=8, step=1)

    with c2:
        st.markdown("**Staffing (on duty)**")
        fd_cap = st.number_input("Front Desk staff", min_value=0, max_value=50, value=1, step=1)
        nurse_cap = st.number_input("Nurses / MAs", min_value=0, max_value=50, value=1, step=1)
        provider_cap = st.number_input("Providers", min_value=0, max_value=50, value=1, step=1)
        bo_cap = st.number_input("Back Office staff", min_value=0, max_value=50, value=1, step=1)

    st.markdown("---")
    st.markdown("#### How tasks travel through the system")
    render_static_flow()

    with st.expander("Advanced (optional) — Routing, times & variability", expanded=False):
        # Routing
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

        # Distributions + overheads + loops
        c6, c7, c8 = st.columns(3)
        with c6:
            dist_fd = st.selectbox("Front Desk distribution", ["normal", "exponential"], index=0)
            dist_nurse_protocol = st.selectbox("Nurse Protocol distribution", ["normal", "exponential"], index=0)
            emr_fd = st.slider("Front Desk EMR overhead (min)", 0.0, 5.0, 0.5, 0.1)
            p_fd_insuff = st.slider("Chance Front Desk bounce", 0.0, 0.6, 0.05, 0.01)
        with c7:
            dist_nurse = st.selectbox("Nurse distribution", ["exponential", "normal"], index=0)
            dist_provider = st.selectbox("Provider distribution", ["exponential", "normal"], index=0)
            emr_nu = st.slider("Nurse EMR overhead (min)", 0.0, 5.0, 0.5, 0.1)
            p_nurse_insuff = st.slider("Chance Nurse needs more info", 0.0, 0.6, 0.05, 0.01)
        with c8:
            dist_backoffice = st.selectbox("Back Office distribution", ["exponential", "normal"], index=0)
            emr_pr = st.slider("Provider EMR overhead (min)", 0.0, 5.0, 0.5, 0.1)
            emr_bo = st.slider("Back Office EMR overhead (min)", 0.0, 5.0, 0.1, 0.1)
            max_fd_loops = st.slider("Max Front Desk info loops", 0, 10, 3)
            max_nurse_loops = st.slider("Max Nurse re-check loops", 0, 10, 2)
            fd_loop_delay = st.slider("Time to collect missing info (min)", 0.0, 240.0, 30.0, 5.0)

        st.caption("Tip: You can leave these as defaults and jump to **Run & Results**.")

    # pack simple params to session so Step 2 can use them directly
    open_minutes = open_hours * MIN_PER_HOUR
    sim_minutes = sim_hours * MIN_PER_HOUR

    # Defaults if user didn't open the Advanced expander
    local_vars = locals()
    for k, v in dict(
        p_admin=0.30, p_protocol=0.40, p_backoffice=0.20,
        svc_frontdesk=3.0, svc_nurse_protocol=2.0, svc_nurse=4.0, svc_provider=6.0, svc_backoffice=5.0,
        cv_speed=0.25,
        dist_fd="normal", dist_nurse_protocol="normal", dist_nurse="exponential",
        dist_provider="exponential", dist_backoffice="exponential",
        emr_fd=0.5, emr_nu=0.5, emr_pr=0.5, emr_bo=0.5,
        p_fd_insuff=0.05, p_nurse_insuff=0.05, max_fd_loops=3, max_nurse_loops=2, fd_loop_delay=30.0
    ).items():
        if k not in local_vars:
            locals()[k] = v

    st.session_state["design"] = dict(
        sim_minutes=sim_minutes,
        arrivals_per_hour=arrivals_per_hour,
        open_minutes=open_minutes,
        frontdesk_cap=fd_cap, nurse_cap=nurse_cap, provider_cap=provider_cap, backoffice_cap=bo_cap,
        p_admin=p_admin, p_protocol=p_protocol, p_backoffice=p_backoffice,
        svc_frontdesk=svc_frontdesk, svc_nurse_protocol=svc_nurse_protocol, svc_nurse=svc_nurse,
        svc_provider=svc_provider, svc_backoffice=svc_backoffice,
        dist_role={
            "Front Desk": dist_fd,
            "NurseProtocol": dist_nurse_protocol,
            "Nurse": dist_nurse,
            "Provider": dist_provider,
            "Back Office": dist_backoffice
        },
        cv_speed=cv_speed,
        emr_overhead={
            "Front Desk": emr_fd, "Nurse": emr_nu, "NurseProtocol": emr_nu, "Provider": emr_pr, "Back Office": emr_bo
        },
        p_fd_insuff=p_fd_insuff, max_fd_loops=max_fd_loops, fd_loop_delay=fd_loop_delay,
        p_nurse_insuff=p_nurse_insuff, max_nurse_loops=max_nurse_loops
    )
    st.success("Looks good! Switch to **Run & Results** above to execute the simulation.")

# -------- STEP 2: RUN & RESULTS --------
else:
    st.subheader("Step 2 — Run & Results")

    if "design" not in st.session_state:
        st.info("Go to **Design** first to set up your clinic.")
        st.stop()

    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    run = st.button("Run Simulation")

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

        # Build event DataFrame
        df_events = pd.DataFrame(
            metrics.events,
            columns=["Time (min)","Task ID","Step Code","Note","Arrival Time (min)"]
        ).sort_values("Time (min)")
        df_events["Step"] = df_events["Step Code"].map(pretty_step)
        df_events["Day"] = (df_events["Time (min)"] // DAY_MIN).astype(int)

        # Daily backlog & carryover
        total_days = int(math.ceil(p["sim_minutes"] / DAY_MIN))
        backlog_by_day, carryover_by_day = [], []
        for d in range(total_days):
            arrivals_to_date = sum(1 for _,t in metrics.task_arrival_time.items() if day_index_at(t) <= d)
            completed_to_date = sum(1 for _,t in metrics.task_completion_time.items() if t is not None and day_index_at(t) <= d)
            backlog = max(0, arrivals_to_date - completed_to_date)
            backlog_by_day.append((d, backlog))
            carry = 0
            for tid, t_a in metrics.task_arrival_time.items():
                if day_index_at(t_a) == d:
                    t_c = metrics.task_completion_time.get(tid, None)
                    if (t_c is None) or (day_index_at(t_c) > d):
                        carry += 1
            carryover_by_day.append((d, carry))

        # Cycle time
        ct_rows = []
        for tid, t_c in metrics.task_completion_time.items():
            if t_c is not None:
                t_a = metrics.task_arrival_time.get(tid)
                if t_a is not None:
                    ct_rows.append(dict(task_id=tid, cycle_time_min=t_c - t_a))
        df_cycle = pd.DataFrame(ct_rows)

        def safe_avg(xs): return float(np.mean(xs)) if len(xs)>0 else 0.0
        def safe_p90(xs): return float(np.percentile(xs,90)) if len(xs)>0 else 0.0

        # Same-day resolution
        same_day_resolved = 0
        for tid, t_c in metrics.task_completion_time.items():
            if t_c is not None and day_index_at(t_c) == day_index_at(metrics.task_arrival_time.get(tid, -999999)):
                same_day_resolved += 1
        same_day_rate = (same_day_resolved / max(1, metrics.completed))

        # Utilizations (open-time adjusted)
        open_time_available = effective_open_minutes(p["sim_minutes"], p["open_minutes"])
        denom_fd = max(1, p["frontdesk_cap"]) * open_time_available
        denom_nu = max(1, p["nurse_cap"]) * open_time_available
        denom_pr = max(1, p["provider_cap"]) * open_time_available
        denom_bo = max(1, p["backoffice_cap"]) * open_time_available

        df_kpis = pd.DataFrame({
            "Metric": [
                "Tasks arrived", "Tasks completed", "Completion rate",
                "Resolved at Front Desk", "Resolved by Nurse Protocol", "Sent to Back Office",
                "Utilization — Front Desk (open-time adj.)", "Utilization — Nurse (open-time adj.)",
                "Utilization — Provider (open-time adj.)", "Utilization — Back Office (open-time adj.)",
                "Average wait (min) — Front Desk", "90th percentile wait (min) — Front Desk",
                "Average wait (min) — Nurse", "90th percentile wait (min) — Nurse",
                "Average wait (min) — Provider", "90th percentile wait (min) — Provider",
                "Average wait (min) — Back Office", "90th percentile wait (min) — Back Office",
                "Average cycle time (min, completed)", "90th percentile cycle time (min, completed)",
                "Same-day resolution rate",
                "FD missing-info loops (total)", "% tasks with any FD loop",
                "Nurse missing-info loops (total)", "% tasks with any Nurse loop"
            ],
            "Value": [
                metrics.arrivals,
                metrics.completed,
                pct(metrics.completed / max(1, metrics.arrivals)),
                metrics.resolved_admin,
                metrics.resolved_protocol,
                metrics.routed_backoffice,
                pct(min(1.0, metrics.service_time_sum["Front Desk"] / denom_fd)),
                pct(min(1.0, metrics.service_time_sum["Nurse"] / denom_nu)),
                pct(min(1.0, metrics.service_time_sum["Provider"] / denom_pr)),
                pct(min(1.0, metrics.service_time_sum["Back Office"] / denom_bo)),
                round(safe_avg(metrics.waits["Front Desk"]),2), round(safe_p90(metrics.waits["Front Desk"]),2),
                round(safe_avg(metrics.waits["Nurse"]),2), round(safe_p90(metrics.waits["Nurse"]),2),
                round(safe_avg(metrics.waits["Provider"]),2), round(safe_p90(metrics.waits["Provider"]),2),
                round(safe_avg(metrics.waits["Back Office"]),2), round(safe_p90(metrics.waits["Back Office"]),2),
                (round(df_cycle["cycle_time_min"].mean(),2) if not df_cycle.empty else 0.0),
                (round(np.percentile(df_cycle["cycle_time_min"],90),2) if not df_cycle.empty else 0.0),
                pct(same_day_rate),
                metrics.loop_fd_insufficient,
                pct(len(metrics.looped_tasks_fd) / max(1, metrics.arrivals)),
                metrics.loop_nurse_insufficient,
                pct(len(metrics.looped_tasks_nurse) / max(1, metrics.arrivals)),
            ]
        })

        # Persist
        st.session_state["results"] = dict(
            df_events=df_events,
            df_kpis=df_kpis,
            queues=metrics.queues,
            times=metrics.time_stamps,
            waits=metrics.waits,
            backlog_by_day=backlog_by_day,
            carryover_by_day=carryover_by_day,
            cycle=df_cycle,
            metrics=metrics,
        )

    res = st.session_state["results"]
    if not res:
        st.info("Configure **Design** and click **Run Simulation** above.")
    else:
        # KPIs
        st.subheader("Key Performance Indicators")
        st.dataframe(res["df_kpis"], use_container_width=True)

        # Flow realized (Sankey)
        st.subheader("How tasks actually flowed (this run)")
        render_sankey_from_events(res["df_events"], res["metrics"])

        # Event Log (quick filters)
        st.subheader("Event Log")
        df = res["df_events"]
        c1, c2, c3 = st.columns([1.2,1.2,0.8])
        with c1:
            stage_opts = ["(All Steps)"] + sorted(df["Step"].unique().tolist())
            pick_stage = st.selectbox("Filter by step", stage_opts, index=0)
        with c2:
            id_sub = st.text_input("Filter by Task ID", "")
        with c3:
            latest_only = st.checkbox("Show last 500", value=True)

        df_view = df
        if pick_stage != "(All Steps)":
            df_view = df_view[df_view["Step"] == pick_stage]
        if id_sub.strip():
            df_view = df_view[df_view["Task ID"].str.contains(id_sub.strip(), case=False, regex=False)]
        if latest_only:
            df_view = df_view.tail(500)
        st.dataframe(df_view[["Time (min)","Day","Task ID","Step","Note"]], use_container_width=True, height=280)

        # Optional small Pareto
        st.subheader("Top steps (Pareto)")
        pareto = df["Step"].value_counts().reset_index()
        pareto.columns = ["Step","Count"]
        st.dataframe(pareto.head(10), use_container_width=True)

        fig_p = plt.figure(figsize=(3.2, 1.4))
        plt.bar(pareto["Step"].head(8), pareto["Count"].head(8))
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.ylabel("Count")
        plt.title("Top steps")
        plt.tight_layout()
        st.pyplot(fig_p, clear_figure=True)
