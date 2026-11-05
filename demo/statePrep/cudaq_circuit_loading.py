import json
import cudaq
import importlib.util
import os

def load_circuit_from_json(json_path):
    with open(json_path, 'r') as f:
        circuit = json.load(f)
    return circuit


def write_and_import_kernel(circuit, kernel_name="generated_kernel", module_name="generated_kernel_module"):
    # Support both {"operations": [...]} and plain list
    if isinstance(circuit, dict) and 'operations' in circuit:
        ops = circuit['operations']
        n_qubits = circuit.get('num_qubits')
        if n_qubits is None:
            raise ValueError("num_qubits must be present in the circuit JSON.")
    elif isinstance(circuit, list):
        ops = circuit
        raise ValueError("num_qubits must be provided in the circuit dictionary, not just a list.")
    else:
        raise ValueError("Unrecognized circuit format.")

    # Build the kernel code as a string
    code_lines = [
        'import cudaq',
        f'@cudaq.kernel',
        f'def {kernel_name}(q: cudaq.qvector):'
    ]
    for gate in ops:
        gtype = gate.get('op', gate.get('gate'))
        if gtype == 'Rx':
            angle = gate['angle']
            target = gate['target'] - 1  # convert to 0-based
            code_lines.append(f'    rx({angle}, q[{target}])')
        elif gtype == 'Ry':
            angle = gate['angle']
            target = gate['target'] - 1
            code_lines.append(f'    ry({angle}, q[{target}])')
        elif gtype == 'Rz':
            angle = gate['angle']
            target = gate['target'] - 1
            code_lines.append(f'    rz({angle}, q[{target}])')
        elif gtype == 'CNOT':
            control = gate['control'] - 1
            target = gate['target'] - 1
            code_lines.append(f'    cx(q[{control}], q[{target}])')
        else:
            raise ValueError(f"Unsupported gate: {gtype}")
    code = '\n'.join(code_lines)

    # Write to a .py file
    py_path = f"{module_name}.py"
    with open(py_path, "w") as f:
        f.write(code + "\n")

    # Import the module
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Return the kernel function
    return getattr(mod, kernel_name)


def julia_ops_to_cudaq_kernel(circuit_ops_array, num_qubits, kernel_name="generated_kernel", module_name="generated_kernel_module"):
    """
    Convert a Julia-originating list of gate dicts and num_qubits directly to a CUDA-Q kernel.
    This expects circuit_ops_array to be a list of Python dicts (not Julia objects).
    If the input is a Julia object, convert it to a Python list of dicts first (e.g., using list() and dict()).
    """
    # If using juliacall, ensure conversion to native Python types
    # (juliacall usually auto-converts, but double-check)
    ops = [dict(gate) for gate in circuit_ops_array]
    circuit_dict = {"num_qubits": num_qubits, "operations": ops}
    return write_and_import_kernel(circuit_dict, kernel_name=kernel_name, module_name=module_name)
