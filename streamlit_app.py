import streamlit as st
import simpy
import random
import numpy as np
import math
import matplotlib.pyplot as plt
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

def day_index_at(t_min):
    return int(t_min // DAY_MIN)

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
        self.loop_fd_insufficient = 0
        self.loop_nurse_insufficient = 0
        # event: (time_min, task_id, step_code, note, arrival_time_min)
        self.events = []
        # lifecycle
        self.task_arrival_time = {}     # id -> time_min
        self.task_completion_time = {}  # id -> time_min

    def log(self, t, name, step, note="", arrival_t=None):
        self.events.append((t, name, step, note, arrival_t if arrival_t is not None else self.task_arrival_time.get(name)))

# =============================
# Step labels (user-facing)
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
        # Resources
        self.frontdesk = simpy.Resource(env, capacity=params["frontdesk_cap"])
        self.nurse = simpy.Resource(env, capacity=params["nurse_cap"])
        self.provider = simpy.Resource(env, capacity=params["provider_cap"])
        self.backoffice = simpy.Resource(env, capacity=params["backoffice_cap"])

    def scheduled_service(self, resource, role_account, mean_time, role_for_dist=None):
        """
        Service chunks that respect clinic hours; releases resource at close and resumes when open.
        """
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

    # Missing info loops at Front Desk
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

    # Missing info discovered by Nurse (back to FD, then re-check)
    while (nurse_loops < s.p["max_nurse_loops"]) and (random.random() < s.p["p_nurse_insuff"]):
        nurse_loops += 1
        s.m.loop_nurse_insufficient += 1
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

    # Back Office (optional)
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
# Streamlit UI
# =============================
st.set_page_config(page_title="CHC Burnout Model — HSyE at Northeastern University", layout="wide")
st.title("CHC Burnout Model — HSyE at Northeastern University")

# ---- Landing intro (with a percentile note) ----
st.markdown("""
### Welcome to the CHC Task Workflow Simulator
This model simulates **asynchronous tasks** a clinic handles every day — **emails, phone calls, and portal messages** — as they move through **Front Desk → Nurse → Provider → (optional) Back Office**.

**What you can do here**
- Stress-test your clinic by changing **arrivals**, **staffing**, **open hours**, **EMR overhead**, and **routing** (where tasks get resolved).
- Explore operational pain points like **backlogs**, **waits**, **cycle time**, and **carryover to the next day**.
- Visualize bottlenecks with compact charts and dive deeper in the **Analysis** tab.

**How to get started**
1) Open the **Simulation Controls** on the left (sections are collapsible).
2) Choose **Simulation time (hours)**, **arrivals/hour**, **open hours**, and **capacities**.
3) Click **Run Simulation**.  
Results show under the **Run** tab; deeper diagnostics live in the **Analysis** tab.
""")

# --- Sidebar controls (expanders) ---
with st.sidebar:
    st.header("Simulation Controls")

    with st.expander("Simulation Duration & Arrivals", expanded=True):
        sim_hours = st.slider("Simulation time (hours)", 1, 24*7, 8, step=1)
        sim_minutes = sim_hours * MIN_PER_HOUR
        arrivals_per_hour = st.slider("Average task arrivals per hour", 1, 60, 12)

    with st.expander("Clinic Hours"):
        open_hours = st.slider("Hours open per day", 1, 24, 8)
        open_minutes = open_hours * MIN_PER_HOUR

    with st.expander("Capacities (Staff on duty)"):
        fd_cap = st.slider("Front Desk", 0, 10, 1)
        nurse_cap = st.slider("Nurse / MA", 0, 10, 1)
        provider_cap = st.slider("Provider", 0, 10, 1)
        bo_cap = st.slider("Back Office", 0, 10, 1)

    with st.expander("Routing (Where tasks resolve)"):
        p_admin = st.slider("Resolved at Front Desk", 0.0, 1.0, 0.30, 0.05)
        p_protocol = st.slider("Resolved by Nurse protocol", 0.0, 1.0, 0.40, 0.05)
        p_backoffice = st.slider("Requires Back Office after Provider", 0.0, 1.0, 0.20, 0.05)

    with st.expander("Service Times (mean minutes)"):
        svc_frontdesk = st.slider("Front Desk", 0.0, 30.0, 3.0, 0.5)
        svc_nurse_protocol = st.slider("Nurse Protocol", 0.0, 30.0, 2.0, 0.5)
        svc_nurse = st.slider("Nurse (non-protocol)", 0.0, 40.0, 4.0, 0.5)
        svc_provider = st.slider("Provider", 0.0, 60.0, 6.0, 0.5)
        svc_backoffice = st.slider("Back Office", 0.0, 60.0, 5.0, 0.5)

    with st.expander("Variability & EMR Overheads"):
        cv_speed = st.slider("Task speed variability (CV)", 0.0, 0.8, 0.25, 0.05)
        emr_fd = st.slider("Front Desk EMR overhead", 0.0, 5.0, 0.5, 0.1)
        emr_nu = st.slider("Nurse EMR overhead", 0.0, 5.0, 0.5, 0.1)
        emr_pr = st.slider("Provider EMR overhead", 0.0, 5.0, 0.5, 0.1)
        emr_bo = st.slider("Back Office EMR overhead", 0.0, 5.0, 0.5, 0.1)

    with st.expander("Bounces (Missing Information)"):
        p_fd_insuff = st.slider("Chance Front Desk bounce", 0.0, 0.6, 0.05, 0.01)
        max_fd_loops = st.slider("Max Front Desk info loops", 0, 10, 3)
        fd_loop_delay = st.slider("Time to collect missing info (min)", 0.0, 240.0, 30.0, 5.0)
        p_nurse_insuff = st.slider("Chance Nurse needs more info", 0.0, 0.6, 0.05, 0.01)
        max_nurse_loops = st.slider("Max Nurse re-check loops", 0, 10, 2)

    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    run = st.button("Run Simulation")

# --- Cache last results unless user clicks Run ---
if "results" not in st.session_state:
    st.session_state["results"] = None

if run:
    random.seed(seed)
    np.random.seed(seed)

    params = dict(
        arrivals_per_hour=arrivals_per_hour,
        open_minutes=open_minutes,
        frontdesk_cap=fd_cap, nurse_cap=nurse_cap, provider_cap=provider_cap, backoffice_cap=bo_cap,
        p_admin=p_admin, p_protocol=p_protocol, p_backoffice=p_backoffice,
        svc_frontdesk=svc_frontdesk, svc_nurse_protocol=svc_nurse_protocol, svc_nurse=svc_nurse,
        svc_provider=svc_provider, svc_backoffice=svc_backoffice,
        dist_role={"Front Desk":"normal","NurseProtocol":"normal","Nurse":"exponential",
                   "Provider":"exponential","Back Office":"exponential"},
        cv_speed=cv_speed,
        emr_overhead={"Front Desk":emr_fd,"Nurse":emr_nu,"NurseProtocol":emr_nu,"Provider":emr_pr,"Back Office":emr_bo},
        p_fd_insuff=p_fd_insuff, max_fd_loops=max_fd_loops, fd_loop_delay=fd_loop_delay,
        p_nurse_insuff=p_nurse_insuff, max_nurse_loops=max_nurse_loops
    )

    metrics = Metrics()
    env = simpy.Environment()
    system = CHCSystem(env, params, metrics)
    env.process(arrival_process(env, system))
    env.process(monitor(env, system))
    env.run(until=sim_minutes)

    # Build event DataFrame once; persist in session
    df_events = pd.DataFrame(
        metrics.events,
        columns=["Time (min)","Task ID","Step Code","Note","Arrival Time (min)"]
    ).sort_values("Time (min)")
    df_events["Step"] = df_events["Step Code"].map(pretty_step)
    df_events["Day"] = (df_events["Time (min)"] // DAY_MIN).astype(int)

    # Daily backlog + carryover
    total_days = int(math.ceil(sim_minutes / DAY_MIN))
    backlog_by_day = []
    carryover_by_day = []
    for d in range(total_days):
        arrivals_to_date = sum(1 for tid,t in metrics.task_arrival_time.items() if day_index_at(t) <= d)
        completed_to_date = sum(1 for tid,t in metrics.task_completion_time.items() if t is not None and day_index_at(t) <= d)
        backlog = max(0, arrivals_to_date - completed_to_date)
        backlog_by_day.append((d, backlog))

        carry = 0
        for tid, t_a in metrics.task_arrival_time.items():
            a_day = day_index_at(t_a)
            if a_day == d:
                t_c = metrics.task_completion_time.get(tid, None)
                if (t_c is None) or (day_index_at(t_c) > d):
                    carry += 1
        carryover_by_day.append((d, carry))

    # Helper stats
    def safe_avg(xs): return float(np.mean(xs)) if len(xs)>0 else 0.0
    def safe_p90(xs): return float(np.percentile(xs,90)) if len(xs)>0 else 0.0

    # Cycle time per completed task (minutes)
    ct_rows = []
    for tid, t_c in metrics.task_completion_time.items():
        if t_c is not None:
            t_a = metrics.task_arrival_time.get(tid)
            if t_a is not None:
                ct_rows.append(dict(task_id=tid, cycle_time_min=t_c - t_a))
    df_cycle = pd.DataFrame(ct_rows)

    # KPI table (labels improved)
    df_kpis = pd.DataFrame({
        "Metric": [
            "Tasks arrived", "Tasks completed", "Completion rate",
            "Resolved at Front Desk", "Resolved by Nurse Protocol", "Sent to Back Office",
            "Utilization — Front Desk", "Utilization — Nurse", "Utilization — Provider", "Utilization — Back Office",
            "Average wait (min) — Front Desk", "90th percentile wait (min) — Front Desk",
            "Average wait (min) — Nurse", "90th percentile wait (min) — Nurse",
            "Average wait (min) — Provider", "90th percentile wait (min) — Provider",
            "Average wait (min) — Back Office", "90th percentile wait (min) — Back Office",
            "Average cycle time (min, completed)", "90th percentile cycle time (min, completed)"
        ],
        "Value": [
            metrics.arrivals,
            metrics.completed,
            pct(metrics.completed / max(1, metrics.arrivals)),
            metrics.resolved_admin,
            metrics.resolved_protocol,
            metrics.routed_backoffice,
            pct(min(1.0, metrics.service_time_sum["Front Desk"] / (max(1,fd_cap) * sim_minutes))),
            pct(min(1.0, metrics.service_time_sum["Nurse"] / (max(1,nurse_cap) * sim_minutes))),
            pct(min(1.0, metrics.service_time_sum["Provider"] / (max(1,provider_cap) * sim_minutes))),
            pct(min(1.0, metrics.service_time_sum["Back Office"] / (max(1,bo_cap) * sim_minutes))),
            round(safe_avg(metrics.waits["Front Desk"]),2), round(safe_p90(metrics.waits["Front Desk"]),2),
            round(safe_avg(metrics.waits["Nurse"]),2), round(safe_p90(metrics.waits["Nurse"]),2),
            round(safe_avg(metrics.waits["Provider"]),2), round(safe_p90(metrics.waits["Provider"]),2),
            round(safe_avg(metrics.waits["Back Office"]),2), round(safe_p90(metrics.waits["Back Office"]),2),
            (round(df_cycle["cycle_time_min"].mean(),2) if not df_cycle.empty else 0.0),
            (round(np.percentile(df_cycle["cycle_time_min"],90),2) if not df_cycle.empty else 0.0)
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
        cycle=df_cycle
    )

# =============================
# Main content (Tabs)
# =============================
res = st.session_state["results"]
tab_run, tab_analysis = st.tabs(["Run", "Analysis"])

# ----------------- RUN TAB -----------------
with tab_run:
    if not res:
        st.info("Set parameters and press **Run Simulation**.")
    else:
        st.subheader("Key Performance Indicators")
        st.dataframe(res["df_kpis"], use_container_width=True)

        # Event Log (filterable)
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

# ----------------- ANALYSIS TAB -----------------
with tab_analysis:
    if not res:
        st.info("Run a simulation first.")
    else:
        # Keep tables if you want (no charts except Pareto). Example: daily backlog table.
        st.subheader("Daily Backlog & Carryover (Table)")
        daily_df = pd.DataFrame({
            "Day": [d for d,_ in res["backlog_by_day"]],
            "Incomplete tasks at end of day": [v for _,v in res["backlog_by_day"]],
            "Tasks carried to next day": [v for _,v in res["carryover_by_day"]]
        })
        st.dataframe(daily_df, use_container_width=True)

        # Pareto only (smaller figure)
        st.subheader("Where Do Steps Accumulate? (Pareto)")
        df_ev = res["df_events"]
        pareto = df_ev["Step"].value_counts().reset_index()
        pareto.columns = ["Step","Count"]
        st.dataframe(pareto.head(10), use_container_width=True)

        # Smaller Pareto chart
        fig_p = plt.figure(figsize=(3.2, 1.4))  # smaller than before
        plt.bar(pareto["Step"].head(8), pareto["Count"].head(8))
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.ylabel("Count")
        plt.title("Top steps")
        plt.tight_layout()
        st.pyplot(fig_p, clear_figure=True)
