import streamlit as st
import simpy
import random
import numpy as np
import math
import io
import matplotlib.pyplot as plt

# -----------------------------
# Utility helpers
# -----------------------------
# draws an exponential service time with the specified mean in minutes
def exp_time(mean):
    # Exponential with mean "mean"; handle 0 safely
    if mean <= 0:
        return 0.0
    return random.expovariate(1.0/mean)

# draws a normal service time using mean and a coefficient of variation
def normal_time(mean, cv=0.4):
    if mean <= 0:
        return 0.0
    sd = max(1e-6, mean * cv)
    t = random.gauss(mean, sd)
    return max(0.0, t)

# generated a lognormal multiplier with mean 1 and user-chosen CV
# for per-task heterogeneity; each task is randomly faster/slower
def speed_multiplier_from_cv(cv):
    """
    Per-task heterogeneity: draw a lognormal multiplier with mean 1 and user CV.
    """
    if cv <= 0:
        return 1.0
    sigma = math.sqrt(math.log(1 + cv**2))
    mu = -0.5 * sigma**2
    return np.random.lognormal(mean=mu, sigma=sigma)

# picks which base distribution to use for a given role
def draw_service_time(role, mean, dist_map, cv_task):
    """
    Role-based service-time draw with heterogeneity multiplier.
    """
    dist = dist_map.get(role, "exponential")
    if dist == "normal":
        base = normal_time(mean, cv=0.4)
    else:
        base = exp_time(mean)
    mult = speed_multiplier_from_cv(cv_task)
    return base * mult

# formats a fraction as a percentage string
def pct(x):
    return f"{100*x:.1f}%"

# -----------------------------
# Metrics container
# -----------------------------
class Metrics:
    def __init__(self):
        # times when we sampled queues
        self.time_stamps = []

        # queue length time series for each role
        self.queues = {"FrontDesk": [], "Nurse": [], "Provider": [], "BackOffice": []}

        # list of nidividual wait time each entity experienced before getting service at that role
        self.waits = {"FrontDesk": [], "Nurse": [], "Provider": [], "BackOffice": []}
        
        # how many service taps occurres at each role
        self.taps = {"FrontDesk": 0, "Nurse": 0, "Provider": 0, "BackOffice": 0}

        # counters for flow outcomes
        self.resolved_admin = 0
        self.resolved_protocol = 0
        self.completed = 0
        self.routed_backoffice = 0

        # how many requests entered the system
        self.arrivals = 0

        # utilization approximation - total busy minutes per role
        self.service_time_sum = {"FrontDesk": 0.0, "Nurse": 0.0, "Provider": 0.0, "BackOffice": 0.0}

        # loop-backs & misroutes
        self.loop_fd_insufficient = 0
        self.loop_nurse_insufficient = 0
        self.misroutes = 0

        # light event log (time, name, stage, note)
        self.events = []

    # event log that can be exported
    def log(self, t, name, stage, note=""):
        self.events.append((t, name, stage, note))

# -----------------------------
# System with SimPy resources
# -----------------------------
class CHCSystem:
    def __init__(self, env, params, metrics):
        self.env = env
        self.p = params
        self.m = metrics

        # Resources - and capacity 
        self.frontdesk = simpy.Resource(env, capacity=params["frontdesk_cap"])
        self.nurse = simpy.Resource(env, capacity=params["nurse_cap"])
        self.provider = simpy.Resource(env, capacity=params["provider_cap"])
        self.backoffice = simpy.Resource(env, capacity=params["backoffice_cap"])

    def service(self, role, mean_time):
        # Role-based dist + heterogeneity + EMR overhead
        dur_core = draw_service_time(role, mean_time, self.p["dist_role"], self.p["cv_speed"])
        dur_emr = max(0.0, self.p["emr_overhead"].get(role, 0.0))
        dur = dur_core + dur_emr
        self.m.service_time_sum[role] += dur
        yield self.env.timeout(dur)

# -----------------------------
# Arrival generator
# -----------------------------
def arrival_process(env, system):
    i = 0
    while True:
        # inter-arrival based on Poisson process
        lam = system.p["arrivals_per_hour"] / 60.0

        # exponential inter-arrival time and waits that long
        inter = random.expovariate(lam) if lam > 0 else 999999
        yield env.timeout(inter)
        i += 1
        system.m.arrivals += 1
        name = f"Req-{i:04d}"
        system.m.log(env.now, name, "ARRIVE", "")
        env.process(workflow(env, name, system))

# -----------------------------
# Workflow logic
# -----------------------------
# end to end path for a single request
def workflow(env, name, s: CHCSystem):
    start_time = env.now

    # ---- Front Desk ----
    with s.frontdesk.request() as req:
        t_req = env.now # request capacity
        yield req
        s.m.waits["FrontDesk"].append(env.now - t_req) # record wait time
        s.m.taps["FrontDesk"] += 1 # record a tap
        s.m.log(env.now, name, "FRONT_DESK_IN", "") 
        yield env.process(s.service("FrontDesk", s.p["svc_frontdesk"])) # perform front desk service
        s.m.log(env.now, name, "FRONT_DESK_OUT", "")

    # --- Possible insufficient info at Front Desk -> loop back after delay ---
    if random.random() < s.p.get("p_fd_insuff", 0.0):
        s.m.loop_fd_insufficient += 1
        s.m.log(env.now, name, "FD_INSUFF", "Request more info")
        yield env.timeout(s.p.get("fd_loop_delay", 0.0))
        with s.frontdesk.request() as req2:
            t_req2 = env.now
            yield req2
            s.m.waits["FrontDesk"].append(env.now - t_req2)
            s.m.taps["FrontDesk"] += 1
            s.m.log(env.now, name, "FRONT_DESK_IN2", "Loop-back")
            yield env.process(s.service("FrontDesk", s.p["svc_frontdesk"]))
            s.m.log(env.now, name, "FRONT_DESK_OUT2", "Loop-back done")

    # admin vs clinical
    if random.random() < s.p["p_admin"]:
        s.m.resolved_admin += 1
        s.m.completed += 1
        s.m.log(env.now, name, "DONE", "Resolved at Front Desk")
        return

    # ---- Nurse/MA ----
    with s.nurse.request() as req:
        t_req = env.now
        yield req
        s.m.waits["Nurse"].append(env.now - t_req)
        s.m.taps["Nurse"] += 1
        s.m.log(env.now, name, "NURSE_IN", "")
        # protocol?
        if random.random() < s.p["p_protocol"]:
            yield env.process(s.service("Nurse", s.p["svc_nurse_protocol"]))
            s.m.resolved_protocol += 1
            s.m.completed += 1
            s.m.log(env.now, name, "DONE", "Resolved by protocol")
            return
        else:
            yield env.process(s.service("Nurse", s.p["svc_nurse"]))
            s.m.log(env.now, name, "NURSE_OUT", "")

    # --- Possible insufficient info discovered by Nurse ---
    if random.random() < s.p.get("p_nurse_insuff", 0.0):
        s.m.loop_nurse_insufficient += 1
        with s.frontdesk.request() as req3:
            t_req3 = env.now
            yield req3
            s.m.waits["FrontDesk"].append(env.now - t_req3)
            s.m.taps["FrontDesk"] += 1
            s.m.log(env.now, name, "FRONT_DESK_IN3", "Collect missing info")
            yield env.process(s.service("FrontDesk", s.p["svc_frontdesk"]))
            s.m.log(env.now, name, "FRONT_DESK_OUT3", "Info collected")
        with s.nurse.request() as req4:
            t_req4 = env.now
            yield req4
            s.m.waits["Nurse"].append(env.now - t_req4)
            s.m.taps["Nurse"] += 1
            s.m.log(env.now, name, "NURSE_RECHECK_IN", "Re-check after info")
            yield env.process(s.service("Nurse", max(0.0, 0.5 * s.p["svc_nurse"])))
            s.m.log(env.now, name, "NURSE_RECHECK_OUT", "")

    # ---- Provider ----
    with s.provider.request() as req:
        t_req = env.now
        yield req
        s.m.waits["Provider"].append(env.now - t_req)
        s.m.taps["Provider"] += 1
        s.m.log(env.now, name, "PROVIDER_IN", "")
        # Misrouting: wrong provider first
        if random.random() < s.p.get("p_misroute", 0.0):
            s.m.misroutes += 1
            yield env.process(s.service("Provider", s.p["svc_provider"]))
            s.m.log(env.now, name, "PROVIDER_WRONG_OUT", "Misroute handled")
        # Correct provider touch
        yield env.process(s.service("Provider", s.p["svc_provider"]))
        s.m.log(env.now, name, "PROVIDER_OUT", "")

    # ---- Back Office (sometimes) ----
    if random.random() < s.p["p_backoffice"]:
        with s.backoffice.request() as req:
            t_req = env.now
            yield req
            s.m.waits["BackOffice"].append(env.now - t_req)
            s.m.taps["BackOffice"] += 1
            s.m.log(env.now, name, "BACKOFFICE_IN", "")
            yield env.process(s.service("BackOffice", s.p["svc_backoffice"]))
            s.m.routed_backoffice += 1
            s.m.log(env.now, name, "BACKOFFICE_OUT", "")

    s.m.completed += 1
    s.m.log(env.now, name, "DONE", "")

# -----------------------------
# Monitor queues each minute
# -----------------------------
def monitor(env, s: CHCSystem):
    while True:
        s.m.time_stamps.append(env.now)
        s.m.queues["FrontDesk"].append(len(s.frontdesk.queue))
        s.m.queues["Nurse"].append(len(s.nurse.queue))
        s.m.queues["Provider"].append(len(s.provider.queue))
        s.m.queues["BackOffice"].append(len(s.backoffice.queue))
        yield env.timeout(1)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="CHC Burnout DES", layout="wide")
st.title("Community Health Center Workflow — DES Simulator (SimPy)")

with st.sidebar:
    st.header("Simulation Controls")
    sim_minutes = st.slider("Simulation time (minutes)", 60, 24*60, 8*60, step=30)
    arrivals_per_hour = st.slider("Arrivals per hour", 1, 40, 12)
    st.subheader("Capacities (# staff)")
    fd_cap = st.slider("Front Desk capacity", 0, 5, 1)
    nurse_cap = st.slider("Nurse/MA capacity", 0, 5, 1)
    provider_cap = st.slider("Provider capacity", 0, 5, 1)
    bo_cap = st.slider("Back Office capacity", 0, 5, 1)

    st.subheader("Routing Probabilities")
    p_admin = st.slider("Admin at Front Desk", 0.0, 1.0, 0.30, 0.05)
    p_protocol = st.slider("Resolved by Protocol (Nurse)", 0.0, 1.0, 0.40, 0.05)
    p_backoffice = st.slider("Requires Back Office", 0.0, 1.0, 0.20, 0.05)

    st.subheader("Service Times (mean minutes)")
    svc_frontdesk = st.slider("Front Desk mean", 0.0, 15.0, 3.0, 0.5)
    svc_nurse_protocol = st.slider("Nurse Protocol mean", 0.0, 15.0, 2.0, 0.5)
    svc_nurse = st.slider("Nurse (non-protocol) mean", 0.0, 20.0, 4.0, 0.5)
    svc_provider = st.slider("Provider mean", 0.0, 30.0, 6.0, 0.5)
    svc_backoffice = st.slider("Back Office mean", 0.0, 30.0, 5.0, 0.5)

    st.subheader("Heterogeneity")
    cv_speed = st.slider("Per-task speed CV (all roles)", 0.0, 0.6, 0.25, 0.05)

    st.subheader("Tech/EMR Overhead (minutes per touch)")
    emr_fd = st.slider("Front Desk EMR overhead", 0.0, 5.0, 0.5, 0.1)
    emr_nu = st.slider("Nurse EMR overhead", 0.0, 5.0, 0.5, 0.1)
    emr_pr = st.slider("Provider EMR overhead", 0.0, 5.0, 0.5, 0.1)
    emr_bo = st.slider("Back Office EMR overhead", 0.0, 5.0, 0.5, 0.1)

    st.subheader("Loop-backs / Misrouting")
    p_fd_insuff = st.slider("Insufficient info at Front Desk (loop back)", 0.0, 0.4, 0.05, 0.01)
    p_nurse_insuff = st.slider("Insufficient info at Nurse (back to FD, then Nurse)", 0.0, 0.4, 0.05, 0.01)
    p_misroute = st.slider("Misrouted after Nurse (wrong provider first)", 0.0, 0.3, 0.05, 0.01)
    fd_loop_delay = st.slider("Patient info loop delay (min)", 0.0, 120.0, 30.0, 5.0)

    seed = st.number_input("Random seed (for reproducibility)", min_value=0, max_value=999999, value=42, step=1)

    run = st.button("Run Simulation")

if run:
    random.seed(seed)
    np.random.seed(seed)

    params = dict(
        arrivals_per_hour=arrivals_per_hour,
        frontdesk_cap=fd_cap,
        nurse_cap=nurse_cap,
        provider_cap=provider_cap,
        backoffice_cap=bo_cap,
        p_admin=p_admin,
        p_protocol=p_protocol,
        p_backoffice=p_backoffice,
        svc_frontdesk=svc_frontdesk,
        svc_nurse_protocol=svc_nurse_protocol,
        svc_nurse=svc_nurse,
        svc_provider=svc_provider,
        svc_backoffice=svc_backoffice,
        # role-specific distributions
        dist_role={"FrontDesk":"normal","NurseProtocol":"normal","Nurse":"exponential","Provider":"exponential","BackOffice":"exponential"},
        # heterogeneity
        cv_speed=cv_speed,
        # EMR overhead
        emr_overhead={"FrontDesk": emr_fd, "Nurse": emr_nu, "NurseProtocol": emr_nu, "Provider": emr_pr, "BackOffice": emr_bo},
        # loop-backs / misrouting
        p_fd_insuff=p_fd_insuff,
        p_nurse_insuff=p_nurse_insuff,
        p_misroute=p_misroute,
        fd_loop_delay=fd_loop_delay,
    )

    metrics = Metrics()
    env = simpy.Environment()
    system = CHCSystem(env, params, metrics)
    env.process(arrival_process(env, system))
    env.process(monitor(env, system))
    env.run(until=sim_minutes)

    # ----------- KPIs -----------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Arrivals", f"{metrics.arrivals}")
        st.metric("Completed", f"{metrics.completed}")
    with col2:
        st.metric("Resolved at Front Desk", f"{metrics.resolved_admin} ({pct(metrics.resolved_admin/max(1,metrics.arrivals))})")
        st.metric("Resolved by Protocol", f"{metrics.resolved_protocol} ({pct(metrics.resolved_protocol/max(1,metrics.arrivals))})")
    with col3:
        routed_pct = metrics.routed_backoffice/max(1,metrics.arrivals)
        st.metric("Routed to Back Office", f"{metrics.routed_backoffice} ({pct(routed_pct)})")
    with col4:
        # crude utilizations (sum service time / (capacity * sim time))
        util_fd = metrics.service_time_sum["FrontDesk"] / (max(1,fd_cap) * sim_minutes)
        util_nu = metrics.service_time_sum["Nurse"] / (max(1,nurse_cap) * sim_minutes)
        util_pr = metrics.service_time_sum["Provider"] / (max(1,provider_cap) * sim_minutes)
        util_bo = metrics.service_time_sum["BackOffice"] / (max(1,bo_cap) * sim_minutes)
        st.metric("Provider Utilization", pct(min(1.0,util_pr)))
        st.metric("Nurse Utilization", pct(min(1.0,util_nu)))
        st.metric("Front Desk Utilization", pct(min(1.0,util_fd)))

    # Loop-back / misroute metrics
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("FD Insufficient-Info Loops", f"{metrics.loop_fd_insufficient}")
    with colB:
        st.metric("Nurse Insufficient-Info Loops", f"{metrics.loop_nurse_insufficient}")
    with colC:
        st.metric("Misroutes (Wrong Provider First)", f"{metrics.misroutes}")

    st.markdown("---")

    # ----------- Plots -----------
    # 1) Queue length time series
    fig1 = plt.figure(figsize=(8,3))
    for role, q in metrics.queues.items():
        plt.plot(metrics.time_stamps, q, label=role)
    plt.xlabel("Time (min)")
    plt.ylabel("Queue length")
    plt.title("Queue lengths over time")
    plt.legend()
    st.pyplot(fig1, clear_figure=True)

    # 2) Wait time histograms (per station) - one chart per instructions
    for role in ["FrontDesk", "Nurse", "Provider", "BackOffice"]:
        waits = metrics.waits[role]
        fig = plt.figure(figsize=(6,3))
        if len(waits) > 0:
            plt.hist(waits, bins=min(30, max(5, int(math.sqrt(len(waits))))))
        plt.xlabel(f"Wait time in {role} queue (min)")
        plt.ylabel("Count")
        plt.title(f"Wait time distribution — {role}")
        st.pyplot(fig, clear_figure=True)

    # ----------- Event log table -----------
    st.subheader("Event log (last 500 rows)")
    import pandas as pd
    df = pd.DataFrame(metrics.events, columns=["time", "id", "stage", "note"])
    df = df.sort_values("time").tail(500)
    st.dataframe(df, use_container_width=True)

    # ----------- Downloadable CSV -----------
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download event log CSV", data=csv, file_name="event_log.csv", mime="text/csv")

    st.markdown("""
    **Notes**
    - Queue lengths are sampled every simulated minute.
    - Utilization is approximated as total service time / (capacity × sim time).
    - Increase arrivals or reduce capacity to see backlogs form (burnout proxy).
    - Increase Protocol % to offload Provider workload.
    """)
else:
    st.info("Set your parameters in the sidebar and press **Run Simulation** to see results.")

