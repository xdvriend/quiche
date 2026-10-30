"""One QPE job: all three spin states of Fe(SH)4- for one (backend, error budget) pair.

Usage:  python qpe_job.py ID

    ID  0 ... 10  ->  GPU, error budget 2.0, 1.0, ..., 2/2^10  (4 ... 14 phase qubits)
    ID 11 ... 21  ->  CPU, same error budgets                 (4 ... 14 phase qubits)

Run from demo/. Results are appended to orca_output/qpe_trotter_reps_1_results.json, the file the notebook uses.
"""
import argparse
import contextlib
import fcntl
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from cudaq_algorithms import chemistry
import cudaq

import quiche
from quiche.core import ElectronicHamiltonian, Errors, PhaseEstimation, Simulation
from quiche.cudaq import measured_circuit_kernel, sample_post_processor, textbook_qpe_circuit
from quiche.dispatch import QPESpec
from quiche.io import fcidump

from statePrep.cudaq_circuit_loading import load_circuit_from_json, write_and_import_kernel

# Settings, as in the notebook.
multiplicities = [2, 4, 6]
num_electrons = 5
shots = 1000
trotter_reps = 1  # fixed Trotter steps per application of U; None = QUICHE's budget
output_dir = Path.cwd() / "orca_output"
results_file = output_dir / f"qpe_trotter_reps_{trotter_reps}_results.json"

# The job list: one job per (backend, error budget).
errors = [2.0 / 2**k for k in range(11)]
jobs = [("GPU", error) for error in errors] + [("CPU", error) for error in errors]

parser = argparse.ArgumentParser(description="Run QPE for all spin states of Fe(SH)4- for one (backend, error budget).")
parser.add_argument("id", type=int, help=f"job ID, 0 ... {len(jobs) - 1} (0-10: GPU, 11-21: CPU)")
backend, error = jobs[parser.parse_args().id]
print(f"Backend: {backend} | error budget: {error}", flush=True)

# Select the backend: the GPU in single precision, the CPU in double precision.
if backend == "GPU":
    cudaq.set_target("nvidia", option="fp32")
else:
    cudaq.set_target("qpp-cpu")


# Reference energies (CASSCF = FCI in the active space) from the ORCA step.
fci_energy = {r["multiplicity"]: r["energy"] for r in json.loads((output_dir / "orca_results.json").read_text())}

# QPE Hamiltonians: H - c0 for QPE, c0 is added back to the measured energies.
h_qpe, c0 = {}, {}
for multiplicity in multiplicities:
    sq_ham = fcidump.read(str(output_dir / f"FeSH4_mult{multiplicity}.fcidump"))
    h_cq = chemistry.qubit_hamiltonian(sq_ham.one_body, sq_ham.two_body, scalar_offset=sq_ham.core_energy)
    c0[multiplicity], h_qpe[multiplicity] = quiche.PauliSum.from_cudaq(h_cq).split_identity()

# DMRG guide states. Each job writes its generated kernel files to its own temporary folder,
# so jobs running at the same time don't overwrite each other's files.
dmrg = {}
with contextlib.chdir(tempfile.mkdtemp()):
    for multiplicity in multiplicities:
        circuit_path = output_dir / f"FeSH4_mult{multiplicity}_dmrg_circuit.json"
        dmrg[multiplicity] = write_and_import_kernel(
            load_circuit_from_json(circuit_path), kernel_name=f"dmrg_mult{multiplicity}", module_name=f"dmrg_mult{multiplicity}"
        )


def get_qpe(multiplicity, error, reps=None):
    # Split the budget between the Trotter error and the QPE error; QPESpec picks the
    # QPE ancillas and the Trotter time/order/steps from it.
    qpe = QPESpec(
        hamiltonian=ElectronicHamiltonian(
            num_electrons=num_electrons, mapping=quiche.Mapping.JordanWigner, paulis=h_qpe[multiplicity]
        ),
        state_prep=None,  # the DMRG kernel is injected below instead
        algorithm=PhaseEstimation.Textbook,
        simulation=Simulation.Trotter,
        error_budget=Errors(
            estimation=error / 2,
            simulation=error / 2,
            rotations=error / 2,
            state_prep=error / 2,
            overlap=1.0,
        ),
    )
    if reps is None:
        reps = qpe.reps  # QUICHE's Trotter steps from the error budget
    kernel, work_qubits, decode = textbook_qpe_circuit(
        h_qpe[multiplicity], Simulation.Trotter, qpe.num_qpe_ancillas, qpe.time, reps, qpe.order,
        n_qubits=qpe.num_data,
    )
    circuit = measured_circuit_kernel(
        kernel, work_qubits, qpe.num_qpe_ancillas, state_prep=dmrg[multiplicity], num_data=qpe.num_data
    )
    post = sample_post_processor(decode, 1)
    return qpe, circuit, post, reps


def save_result(result):
    # Append one result to the shared results file. The lock makes jobs that finish at the
    # same time wait for each other, so that no job overwrites another job's result.
    with open(results_file.with_suffix(".lock"), "w") as lock:
        fcntl.lockf(lock, fcntl.LOCK_EX)
        results = json.loads(results_file.read_text()) if results_file.exists() else []
        results.append(result)
        results_file.write_text(json.dumps(results, indent=2))
        fcntl.lockf(lock, fcntl.LOCK_UN)


@cudaq.kernel
def warmup():
    q = cudaq.qubit()
    h(q)


# Start the backend up outside the timing.
cudaq.sample(warmup, shots_count=1)

# Runs already in the results file are skipped.
done = set()
if results_file.exists():
    done = {(r["backend"], r["multiplicity"], r["error_budget"]) for r in json.loads(results_file.read_text())}

for multiplicity in multiplicities:
    if (backend, multiplicity, error) in done:
        print(f"{backend} | mult {multiplicity} | error budget {error}: already done, skipping", flush=True)
        continue

    # Build the circuit, then simulate and time it.
    qpe, circuit, post, reps = get_qpe(multiplicity, error, reps=trotter_reps)
    start = time.perf_counter()
    samples = cudaq.sample(circuit, shots_count=shots)
    walltime = time.perf_counter() - start

    energy = float(np.median(post(samples))) + c0[multiplicity]
    save_result({
        "backend": backend,
        "multiplicity": multiplicity,
        "error_budget": error,
        "phase_qubits": qpe.num_qpe_ancillas,
        "total_qubits": qpe.num_qpe_ancillas + qpe.num_data,
        "trotter_reps": reps,
        "energy": energy,
        "abs_error": abs(energy - fci_energy[multiplicity]),
        "walltime_s": walltime,
    })
    print(f"{backend} | mult {multiplicity} | {qpe.num_qpe_ancillas} phase qubits | "
          f"|E - E_FCI| = {abs(energy - fci_energy[multiplicity]):.2e} Ha | {walltime:.1f} s", flush=True)
