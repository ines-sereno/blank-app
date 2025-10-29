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
# Diagram builder (Graphviz DOT)
# =============================
def build_process_graph(p: Dict) -> str:
    def _fmt_pct(x: float) -> str:
        try:
            return f"{100*float(x):.0f}%"
        except:
            return "0%"

    # 🎨 tweak this to any hex you like
    legend_fill = "#E9F7FF"  # soft blue; try "#FFF3E0" for soft orange, "#F3E8FF" for lilac, etc.

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

    # Role nodes (dim if cap==0)
    for r in roles:
        fill = "#EFEFEF" if cap.get(r, 0) <= 0 else "#F7F7F7"
        lines.append(f'  "{r}" [label="{r}\\ncap={cap.get(r,0)}\\nsvc≈{svc.get(r,0):.1f} min", fillcolor="{fill}"];')

    # Nurse protocol annotation
    if p_protocol is not None and svc_proto is not None and cap.get("Nurse", 0) >= 0:
        proto_label = f'Nurse Protocol\\np={_fmt_pct(p_protocol)}\\nsvc≈{svc_proto:.1f} min'
        lines += [
            f'  "NurseProto" [shape=note, fillcolor="#FFFBE6", color="#B59F3B", label="{proto_label}", fontsize=9];',
            '  "Nurse" -> "NurseProto" [style=dotted, label=" info "];'
        ]

    # Done sink
    lines.append('  "Done" [shape=doublecircle, fillcolor="#E8F5E9", color="#5E8D5B"];')

    # Routing edges
    for src, row in route.items():
        for tgt, prob in row.items():
            try:
                prob_f = float(prob)
            except:
                prob_f = 0.0
            if prob_f > 0:
                lines.append(f'  "{src}" -> "{tgt}" [label="{_fmt_pct(prob_f)}"];')

    # Loop self-edges
    for r, (p_loop, max_loops) in loops.items():
        try:
            p_f = float(p_loop)
        except:
            p_f = 0.0
        if p_f > 0:
            lines.append(f'  "{r}" -> "{r}" [style=dashed, color="#999", label="loop {_fmt_pct(p_f)} / max {max_loops}"];')

    # ── Minimal textbox legend (top-left) ────────────────────────────────────────
    lines += [
        f'  legend [shape=box, style="rounded,filled", fillcolor="{legend_fill}", color="#AFC8D8", fontsize=8, '
        '          label="cap = capacity\\nsvc = mean svc time\\n→ routing prob\\n↺ loop prob / max"];',
        '  { rank=min; legend; }',
        '  legend -> "Front Desk" [style=invis, weight=9999];'
    ]
    # ────────────────────────────────────────────────────────────────────────────

    lines.append('}')
    return "\n".join(lines)


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

def go_next():
    st.session_state.wizard_step = min(2, st.session_state.wizard_step + 1)
def go_back():
    st.session_state.wizard_step = max(1, st.session_state.wizard_step - 1)

# Helper to ensure default values in session_state
def _init_ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

# --- Locale-agnostic probability input (text, dot-decimal)
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

# -------- STEP 1: DESIGN --------
if st.session_state.wizard_step == 1:

    st.markdown("""
    ### 🏥 **Design Your Clinic**
    Customize staffing, arrivals, times, loops, and routing. **Staffing below updates the page live**;
    other inputs are saved when you click **Save**.
    """)

    # --- Staffing on MAIN screen (LIVE, not in a form) ---
    st.markdown("### 👥 Staffing (on duty)")
    cStaff1, cStaff2, cStaff3, cStaff4 = st.columns(4)
    with cStaff1:
        st.session_state.fd_cap = st.number_input(
            "Front Desk staff", min_value=0, max_value=50,
            value=_init_ss("fd_cap", 3), step=1, format="%d",
            help="Number of front desk staff simultaneously available."
        )
    with cStaff2:
        st.session_state.nurse_cap = st.number_input(
            "Nurses / MAs", min_value=0, max_value=50,
            value=_init_ss("nurse_cap", 2), step=1, format="%d",
            help="Number of nurses/medical assistants on duty."
        )
    with cStaff3:
        st.session_state.provider_cap = st.number_input(
            "Providers", min_value=0, max_value=50,
            value=_init_ss("provider_cap", 1), step=1, format="%d",
            help="Number of providers on duty."
        )
    with cStaff4:
        st.session_state.bo_cap = st.number_input(
            "Back Office staff", min_value=0, max_value=50,
            value=_init_ss("backoffice_cap", 1), step=1, format="%d",
            help="Number of back-office staff on duty."
        )

    # Live flags based on staffing
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
        # Full-width controls for horizon & variability
        st.markdown("### Simulation horizon & variability")
        sim_days = st.number_input(
            "Days to simulate", min_value=1, max_value=30, value=_init_ss("sim_days", 5),
            step=1, format="%d",
            help="Number of 24-hour days to include in the simulation."
        )
        open_hours = st.number_input(
            "Hours open per day", min_value=1, max_value=24, value=_init_ss("open_hours", 10),
            step=1, format="%d",
            help="Clinic operating hours per day during which work can be performed."
        )
        cv_speed = st.slider(
            "Task speed variability (CV)", 0.0, 0.8, _init_ss("cv_speed", 0.25), 0.05,
            help="How variable individual task times are around their mean (coefficient of variation)."
        )

        st.markdown("### Arrivals per hour at each role")
        cA1, cA2, cA3, cA4 = st.columns(4)
        with cA1:
            arr_fd = st.number_input("→ Front Desk", min_value=0, max_value=500, value=_init_ss("arr_fd", 15),
                                     step=1, format="%d",
                                     help="Average number of tasks arriving to the Front Desk each hour.",
                                     disabled=fd_off)
        with cA2:
            arr_nu = st.number_input("→ Nurse / MAs", min_value=0, max_value=500, value=_init_ss("arr_nu", 20),
                                     step=1, format="%d",
                                     help="Average number of tasks arriving directly to the Nurse/MA queue per hour.",
                                     disabled=nu_off)
        with cA3:
            arr_pr = st.number_input("→ Provider", min_value=0, max_value=500, value=_init_ss("arr_pr", 10),
                                     step=1, format="%d",
                                     help="Average number of tasks arriving directly to the Provider per hour.",
                                     disabled=pr_off)
        with cA4:
            arr_bo = st.number_input("→ Back Office", min_value=0, max_value=500, value=_init_ss("arr_bo", 5),
                                     step=1, format="%d",
                                     help="Average number of tasks arriving directly to the Back Office per hour.",
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

            # Loops (rework) inputs
            st.markdown("#### Loops (rework probabilities, caps, and delays)")
            cL1, cL2 = st.columns(2)
            with cL1:
                p_fd_insuff = st.slider("Front Desk: probability of missing info", 0.0, 1.0,
                                        _init_ss("p_fd_insuff", 0.15), 0.01, disabled=fd_off)
                max_fd_loops = st.number_input("Front Desk: max loops", min_value=0, max_value=10,
                                               value=_init_ss("max_fd_loops", 2), step=1, format="%d", disabled=fd_off)
                fd_loop_delay = st.slider("Front Desk: rework delay (min)", 0.0, 60.0,
                                          _init_ss("fd_loop_delay", 5.0), 0.5, disabled=fd_off)

                p_nurse_insuff = st.slider("Nurse: probability of insufficient info", 0.0, 1.0,
                                           _init_ss("p_nurse_insuff", 0.10), 0.01, disabled=nu_off)
                max_nurse_loops = st.number_input("Nurse: max loops", min_value=0, max_value=10,
                                                  value=_init_ss("max_nurse_loops", 2), step=1, format="%d", disabled=nu_off)
            with cL2:
                p_provider_insuff = st.slider("Provider: probability of rework needed", 0.0, 1.0,
                                              _init_ss("p_provider_insuff", 0.08), 0.01, disabled=pr_off)
                max_provider_loops = st.number_input("Provider: max loops", min_value=0, max_value=10,
                                                     value=_init_ss("max_provider_loops", 2), step=1, format="%d", disabled=pr_off)
                provider_loop_delay = st.slider("Provider: rework delay (min)", 0.0, 60.0,
                                                _init_ss("provider_loop_delay", 5.0), 0.5, disabled=pr_off)

                p_backoffice_insuff = st.slider("Back Office: probability of rework needed", 0.0, 1.0,
                                                _init_ss("p_backoffice_insuff", 0.05), 0.01, disabled=bo_off)
                max_backoffice_loops = st.number_input("Back Office: max loops", min_value=0, max_value=10,
                                                       value=_init_ss("max_backoffice_loops", 2), step=1, format="%d", disabled=bo_off)
                backoffice_loop_delay = st.slider("Back Office: rework delay (min)", 0.0, 60.0,
                                                  _init_ss("backoffice_loop_delay", 5.0), 0.5, disabled=bo_off)

            st.markdown("#### Interaction matrix — Routing Probabilities")
            st.caption("Self-routing is disabled. You cannot route to roles with 0 capacity.")

            route: Dict[str, Dict[str, float]] = {}

            # Build a row UI that omits self-target and disables targets with zero capacity
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
                            help=("Disabled: role has 0 staff" if (tgt in ROLES and cap_map[tgt]==0) else None),
                            disabled=tgt_disabled
                        )
                        if tgt_disabled:
                            val = 0.0
                    row[tgt] = val
                return row

            route["Front Desk"]  = route_row_ui("Front Desk",  {"Nurse": 0.50, "Provider": 0.10, "Back Office": 0.10, DONE: 0.30}, disabled_source=fd_off)
            route["Nurse"]       = route_row_ui("Nurse",       {"Provider": 0.40, "Back Office": 0.20, DONE: 0.40}, disabled_source=nu_off)
            route["Provider"]    = route_row_ui("Provider",    {"Nurse": 0.50, "Back Office": 0.10, DONE: 0.40}, disabled_source=pr_off)
            route["Back Office"] = route_row_ui("Back Office", {"Front Desk": 0.10, "Nurse": 0.10, "Provider": 0.10, DONE: 0.70}, disabled_source=bo_off)

        saved = st.form_submit_button("Save", use_container_width=True)

        if saved:
            open_minutes = int(open_hours * MIN_PER_HOUR)
            sim_minutes = int(sim_days * DAY_MIN)

            # ---- Sanitize routing matrix on Save ----
            for r in ROLES:
                if r in route:
                    route[r].pop(r, None)  # remove self routes if present
            for r in ROLES:
                if r in route:
                    for tgt in list(route[r].keys()):
                        if tgt in ROLES and cap_map[tgt] == 0:
                            route[r][tgt] = 0.0

            # build and store the design dict using LIVE staffing values
            st.session_state["design"] = dict(
                sim_minutes=sim_minutes,
                open_minutes=open_minutes,
                # staffing (live, from session_state)
                frontdesk_cap=st.session_state.fd_cap,
                nurse_cap=st.session_state.nurse_cap,
                provider_cap=st.session_state.provider_cap,
                backoffice_cap=st.session_state.bo_cap,
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

    # --- Process preview diagram ---
    st.markdown("### Process preview")
    st.caption("Live view of staffing, routing, nurse protocol, and loop settings based on your saved design.")
    dot = build_process_graph(st.session_state["design"])
    # keep a consistent size; do not stretch to container width
    st.graphviz_chart(dot, use_container_width=False)

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
