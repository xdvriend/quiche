"""Figures and tables for the Fe(SH)4- QPE demo.

Each function shows a Plotly figure:

    plot_energy_analysis                -> energy error and spin ladder vs phase qubits, + spin-ordering table
    plot_runtime_analysis               -> GPU and CPU walltimes vs phase qubits
    plot_runtime_for_chemical_accuracy  -> walltime to chemical accuracy per spin state

All three read the saved results, so they work without rerunning any calculation:

    results_file         the QPE results JSON (e.g. orca_output/qpe_trotter_reps_1_results.json)
    orca_results_file    the ORCA summary JSON (orca_output/orca_results.json), for the CASSCF energies
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import HTML, display
from plotly.subplots import make_subplots

PLOTLY_JS = True
CHEM_ACC = 1.6e-3  # Ha

# Backend labels, used as subplot titles and legend names.
GPU_LABEL = "GPU (1x NVIDIA T4)"
CPU_LABEL = "CPU (4 cores)"

# Where the notebook loads Plotly's JavaScript from for the centered figures:
# "cdn" keeps the notebook small (needs internet in the browser), True embeds it in every figure (~4 MB each).
PLOTLY_JS = "cdn"

# Extra space between subplot titles and their plots, in pixels.
TITLE_SHIFT = 20

# One colour per spin multiplicity, as RGB.
COLORS = {2: (31, 119, 180), 4: (255, 127, 14), 6: (44, 160, 44)}

# Shared axis style: light dotted grid, solid black axis lines.
AXIS_STYLE = dict(
    showgrid=True, gridcolor="rgba(0, 0, 0, 0.25)", griddash="dot", gridwidth=0.5,
    showline=True, linecolor="black", linewidth=2,
    ticks="outside", tickcolor="black",
)


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------

def load_results(results_file):
    """QPE results as a DataFrame, sorted by phase qubits (the runs were saved out of order)."""
    return pd.read_json(results_file).sort_values("phase_qubits")


def load_fci_energies(orca_results_file):
    """CASSCF (= FCI in the active space) energy per multiplicity, from the ORCA summary."""
    return {r["multiplicity"]: r["energy"] for r in pd.read_json(orca_results_file).to_dict("records")}


def line_and_marker(multiplicity):
    """See-through line and solid marker style for one multiplicity."""
    r, g, b = COLORS[multiplicity]
    return dict(color=f"rgba({r}, {g}, {b}, 0.35)", width=2), dict(color=f"rgb({r}, {g}, {b})", size=9)


def human_time(seconds):
    """Walltime as a readable label, e.g. 1.2 h or 7.3 years."""
    for unit, size in [("years", 365.25 * 86400), ("days", 86400), ("h", 3600), ("min", 60)]:
        if seconds >= size:
            return f"{seconds / size:.1f} {unit}"
    return f"{seconds:.1f} s"


def extrapolate_cpu_walltime(last_cpu_run, target_run):
    """Extrapolate the CPU walltime from the largest measured CPU run to a run only the GPU did.

    The circuit has (2^m - 1) * r Trotter steps (m phase qubits, r Trotter steps per U), and on the
    CPU each step costs in proportion to the size of the quantum state (2^m for the phase register).
    So the walltime scales as (2^m - 1) * r * 2^m. The target's r is taken from its GPU run.
    """
    def cost(run):
        m = run["phase_qubits"]
        return (2**m - 1) * run["trotter_reps"] * 2**m

    return last_cpu_run["walltime_s"] * cost(target_run) / cost(last_cpu_run)


def show_centered_figure(fig):
    """Show a Plotly figure centered in the notebook output (fig.show() always puts it on the left)."""
    html = fig.to_html(full_html=False, include_plotlyjs=PLOTLY_JS)
    display(HTML(f'<div style="display: flex; justify-content: center;">{html}</div>'))


def show_centered_table(table):
    """Show a DataFrame as a centered table in the notebook output.

    Markdown has no way to center a table, so the table is shown as HTML, which looks the same.
    """
    html = table.to_html(index=False).replace("<table", '<table style="margin-left: auto; margin-right: auto;"', 1)
    display(HTML(html))


def runs(results, backend, multiplicity):
    """The runs of one backend and multiplicity, sorted by phase qubits."""
    return results[(results["backend"] == backend) & (results["multiplicity"] == multiplicity)]


# ----------------------------------------------------------------------------------------------
# Figure 1: energy error and spin ladder
# ----------------------------------------------------------------------------------------------

def plot_energy_analysis(results_file, orca_results_file):
    """Energy error (left) and spin ladder relative to the sextet (right) vs phase qubits, GPU runs.

    Below the figure: a table showing, per number of phase qubits, whether QPE gives the correct
    spin ordering, and the gaps between neighbouring spin states, compared with CASSCF.
    """
    results = load_results(results_file)
    fci_energy = load_fci_energies(orca_results_file)
    gpu = results[results["backend"] == "GPU"]
    reference = fci_energy[6]  # the sextet: the lowest spin state
    x_range = [gpu["phase_qubits"].min() - 0.5, gpu["phase_qubits"].max() + 0.5]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Energy error", "Spin ladder"),
                        horizontal_spacing=0.1)

    for multiplicity in COLORS:
        data = runs(results, "GPU", multiplicity)
        line, marker = line_and_marker(multiplicity)

        # Left: energy error.
        fig.add_trace(go.Scatter(
            x=data["phase_qubits"], y=data["abs_error"], mode="lines+markers",
            name=f"2S+1 = {multiplicity}", legendgroup=str(multiplicity), line=line, marker=marker,
        ), row=1, col=1)

        # Right: QPE energy relative to the sextet's CASSCF energy, in mHa.
        fig.add_trace(go.Scatter(
            x=data["phase_qubits"], y=(data["energy"] - reference) * 1000, mode="lines+markers",
            name=f"2S+1 = {multiplicity}", legendgroup=str(multiplicity), showlegend=False, line=line, marker=marker,
        ), row=1, col=2)

        # Right: the CASSCF energy of this spin state, as a dotted line in the same colour.
        fig.add_trace(go.Scatter(
            x=x_range, y=[(fci_energy[multiplicity] - reference) * 1000] * 2, mode="lines",
            name=f"CASSCF 2S+1 = {multiplicity}", legendgroup=str(multiplicity), showlegend=False,
            line=dict(color=marker["color"], dash="dot", width=1.5),
        ), row=1, col=2)

    # Legend entries for the two reference lines.
    fig.add_trace(go.Scatter(x=x_range, y=[CHEM_ACC] * 2, mode="lines", name="1.6 mHa",
                             line=dict(color="black", dash="dash", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="CASSCF",
                             line=dict(color="black", dash="dot", width=1.5)), row=1, col=2)

    fig.update_layout(template="plotly_white", width=1000, height=450)
    fig.update_annotations(yshift=TITLE_SHIFT)  # the subplot titles are annotations: move them up
    fig.update_xaxes(title="Phase qubits", dtick=1, range=x_range, **AXIS_STYLE)
    fig.update_yaxes(title="|E<sub>QPE</sub> − E<sub>FCI</sub>| (Ha)", type="log", dtick=1, exponentformat="power",
                     row=1, col=1, **AXIS_STYLE)
    fig.update_yaxes(title="E − E<sub>CASSCF</sub>(2S+1 = 6) (mHa)", row=1, col=2, **AXIS_STYLE)
    show_centered_figure(fig)

    # Table: CASSCF reference first, then the spin ordering and the gaps between neighbouring
    # spin states (quartet above sextet, doublet above quartet) per number of phase qubits.
    def gaps(energy):
        return {
            "ΔE quartet − sextet (mHa)": f"{(energy[4] - energy[6]) * 1000:.1f}",
            "ΔE doublet − quartet (mHa)": f"{(energy[2] - energy[4]) * 1000:.1f}",
        }

    rows = [{"Phase qubits": "CASSCF", "Spin ordering": "", **gaps(fci_energy)}]
    correct_order = sorted(COLORS, key=lambda mult: fci_energy[mult])
    energies = gpu.pivot_table(index="phase_qubits", columns="multiplicity", values="energy").dropna()
    for m, row in energies.iterrows():
        order = sorted(COLORS, key=lambda mult: row[mult])
        rows.append({
            "Phase qubits": m,
            "Spin ordering": "correct" if order == correct_order else "wrong",
            **gaps(row),
        })
    show_centered_table(pd.DataFrame(rows))


# ----------------------------------------------------------------------------------------------
# Figure 2: walltimes on GPU and CPU
# ----------------------------------------------------------------------------------------------

def plot_runtime_analysis(results_file):
    """Walltime vs phase qubits: GPU (left) and CPU (right), with the CPU extrapolated where only the GPU ran."""
    results = load_results(results_file)

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=(GPU_LABEL, CPU_LABEL), horizontal_spacing=0.06)

    for multiplicity in COLORS:
        line, marker = line_and_marker(multiplicity)
        gpu = runs(results, "GPU", multiplicity)
        cpu = runs(results, "CPU", multiplicity)

        # Measured walltimes: GPU left, CPU right (one legend entry per multiplicity).
        for col, data in [(1, gpu), (2, cpu)]:
            fig.add_trace(go.Scatter(
                x=data["phase_qubits"], y=data["walltime_s"], mode="lines+markers",
                name=f"2S+1 = {multiplicity}", legendgroup=str(multiplicity), showlegend=(col == 1),
                line=line, marker=marker,
            ), row=1, col=col)


        # CPU extrapolation to the phase qubits that only the GPU ran.
        missing = gpu[~gpu["phase_qubits"].isin(cpu["phase_qubits"])]
        if len(cpu) >= 1 and len(missing) >= 1:
            last = cpu.iloc[-1]  # largest measured CPU run
            extrapolated = [extrapolate_cpu_walltime(last, run) for _, run in missing.iterrows()]
            fig.add_trace(go.Scatter(
                x=[last["phase_qubits"]] + list(missing["phase_qubits"]),  # start the dashed line at the last measured point
                y=[last["walltime_s"]] + extrapolated,
                mode="lines+markers",
                name=f"2S+1 = {multiplicity}, extrapolated", legendgroup=str(multiplicity), showlegend=False,
                line=dict(color=line["color"], width=2, dash="dash"),
                marker=dict(color=marker["color"], symbol="x", size=[0] + [9] * len(missing)),  # no marker on the measured point
            ), row=1, col=2)

    # One legend entry explaining the extrapolated style.
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines+markers", name="CPU, extrapolated",
        line=dict(color="rgba(0, 0, 0, 0.35)", width=2, dash="dash"),
        marker=dict(color="black", symbol="x", size=9),
    ), row=1, col=2)

    x_range = [results["phase_qubits"].min() - 0.5, results["phase_qubits"].max() + 0.5]
    fig.update_layout(template="plotly_white", width=1000, height=450)
    fig.update_annotations(yshift=TITLE_SHIFT)  # the subplot titles are annotations: move them up
    fig.update_xaxes(title="Phase qubits", dtick=1, range=x_range, **AXIS_STYLE)
    fig.update_yaxes(type="log", dtick=1, exponentformat="power", **AXIS_STYLE)  # one grid line per power of ten
    fig.update_yaxes(title="Walltime (s)", row=1, col=1)
    show_centered_figure(fig)


# ----------------------------------------------------------------------------------------------
# Figure 3: walltime to chemical accuracy
# ----------------------------------------------------------------------------------------------

def plot_runtime_for_chemical_accuracy(results_file):
    """Walltime of the first run within chemical accuracy, per spin state, on GPU and CPU (bar chart).

    The first run within chemical accuracy is the GPU run with the fewest phase qubits whose error
    is below 1.6 mHa. The CPU time for the same run is measured if the CPU ran it, else extrapolated
    (hatched bar).
    """
    results = load_results(results_file)

    rows = []
    for multiplicity in COLORS:
        gpu = runs(results, "GPU", multiplicity)
        cpu = runs(results, "CPU", multiplicity)
        within = gpu[gpu["abs_error"] <= CHEM_ACC]
        if within.empty:
            print(f"2S+1 = {multiplicity}: no GPU run within chemical accuracy yet")
            continue
        first = within.iloc[0]  # fewest phase qubits that reach chemical accuracy

        measured = cpu[cpu["phase_qubits"] == first["phase_qubits"]]
        if not measured.empty:
            cpu_time, extrapolated = measured["walltime_s"].iloc[0], False
        else:
            cpu_time, extrapolated = extrapolate_cpu_walltime(cpu.iloc[-1], first), True

        rows.append({"multiplicity": multiplicity, "phase_qubits": first["phase_qubits"],
                     "gpu_s": first["walltime_s"], "cpu_s": cpu_time, "cpu_extrapolated": extrapolated})
    chem = pd.DataFrame(rows)

    labels = [f"2S+1 = {m}" for m in chem["multiplicity"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=chem["gpu_s"], name=GPU_LABEL,
        marker=dict(color="rgb(118, 185, 0)", line=dict(color="black", width=1)),
        text=chem["gpu_s"].map(human_time), textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=labels, y=chem["cpu_s"], name=CPU_LABEL,
        marker=dict(
            color="rgba(0, 0, 0, 0.35)", line=dict(color="black", width=1),
            pattern=dict(shape=["x" if e else "" for e in chem["cpu_extrapolated"]], fgcolor="black"),
        ),
        text=chem["cpu_s"].map(human_time), textposition="outside",
    ))

    # Same style as the other figures; the log axis leaves room above the bars for the labels.
    y_min, y_max = chem[["gpu_s", "cpu_s"]].min().min(), chem[["gpu_s", "cpu_s"]].max().max()
    fig.update_layout(template="plotly_white", barmode="group", width=600, height=450)
    fig.update_xaxes(showline=True, linecolor="black", linewidth=2, ticks="outside", tickcolor="black")
    fig.update_yaxes(title="Walltime to chemical accuracy (s)", type="log", dtick=1, exponentformat="power",
                     range=[np.log10(y_min / 3), np.log10(y_max * 30)], **AXIS_STYLE)
    show_centered_figure(fig)
