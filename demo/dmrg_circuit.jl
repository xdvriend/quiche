# dmrg_circuit.jl
#
# FCIDUMP  ->  DMRG ground state (MPS)  ->  state-preparation circuit (JSON)
#
# Usage (from the folder that contains dmrg_env/):
#
#   julia +1.12 --project=dmrg_env dmrg_circuit.jl INPUT.fcidump OUTPUT.json [options]
#
# Options (all optional, written as --name=value):
#   --maxdim=100        maximum DMRG bond dimension
#   --nsweeps=10        number of DMRG sweeps
#   --layers=15         maximum number of disentangling layers in the circuit compiler
#   --tol=1e-4          compiler truncation tolerance
#   --refine=20         environment-tensor refinement sweeps (0 = skip refinement)
#   --precision=fp64    fp64 | fp32 (compiler working precision)
#   --seed=2026         random seed for the initial MPS
#
# Output protocol: everything chatty (DMRG / compiler progress) goes to STDERR.
# STDOUT only contains `key=value` lines, one per line, with exactly one "=" each,
# so that they can be parsed with  dict(line.split("=") for line in out.splitlines()).
#
# Based on the tutorials in https://github.com/Quantum-Motion/chem-state-prep
# and the API of https://github.com/Quantum-Motion/MPSCircuits.jl.

using ITensors
using ITensorMPS
using MPSCircuits
using Random

# ------------------------------------------------------------------------------
# Command line
# ------------------------------------------------------------------------------

const USAGE = "usage: julia --project=dmrg_env dmrg_circuit.jl INPUT.fcidump OUTPUT.json " *
              "[--maxdim=N] [--nsweeps=N] [--layers=N] [--tol=X] [--refine=N] " *
              "[--precision=fp64|fp32] [--seed=N]"

function parse_cli(args)
    positional = String[]
    options = Dict{String,String}()
    for arg in args
        if startswith(arg, "--")
            body = arg[3:end]
            if occursin('=', body)
                key, val = split(body, '='; limit=2)
                options[String(key)] = String(val)
            else
                options[body] = "true"
            end
        else
            push!(positional, arg)
        end
    end
    return positional, options
end

function option(options, key, default::T) where {T}
    haskey(options, key) || return default
    return T === String ? options[key] : parse(T, options[key])
end

# ------------------------------------------------------------------------------
# Robust FCIDUMP normalisation
# ------------------------------------------------------------------------------
#
# MPSCircuits.Chemistry.fcidump_to_mpo expects a fixed 4-line header and a
# particular number format. To work with any (restricted, real) FCIDUMP we read
# the file ourselves, de-duplicate the integrals, and re-write a normalised
# temporary FCIDUMP that the package parser handles.

function header_int(header, key; default=nothing)
    m = match(Regex(key * raw"\s*=\s*(-?\d+)", "i"), header)
    if m === nothing
        default === nothing && error("FCIDUMP header has no $key entry.")
        return default
    end
    return parse(Int, m.captures[1])
end

function normalise_fcidump(path::AbstractString)
    text = read(path, String)

    term = findfirst(r"&END|/"i, text)
    term === nothing && error("Could not find the end of the FCIDUMP header (&END or /) in $path.")
    header = String(SubString(text, 1, prevind(text, first(term))))
    body = String(SubString(text, nextind(text, last(term))))

    occursin(r"\bI?UHF\s*=\s*(1|\.?T)"i, header) &&
        error("Spin-unrestricted (UHF) FCIDUMP files are not supported.")

    norb = header_int(header, "NORB")
    nelec = header_int(header, "NELEC")
    ms2 = header_int(header, "MS2"; default=0)
    norb >= 1 || error("NORB must be positive, got $norb.")
    0 <= nelec <= 2 * norb || error("NELEC=$nelec is incompatible with NORB=$norb.")

    core = 0.0
    one_body = Dict{Tuple{Int,Int},Float64}()
    two_body = Dict{NTuple{4,Int},Float64}()

    for raw in split(body, '\n')
        line = strip(replace(raw, ',' => ' '))
        isempty(line) && continue
        tok = split(line)
        length(tok) == 5 || error("Malformed FCIDUMP integral line: $(repr(raw))")

        v = parse(Float64, replace(tok[1], r"[dD]" => "e"))  # Fortran exponents
        i, j, k, l = parse.(Int, tok[2:5])
        max(i, j, k, l) <= norb || error("Orbital index larger than NORB in line: $(repr(raw))")

        if i > 0 && j > 0 && k > 0 && l > 0
            # canonical representative of the 8-fold symmetry class
            p, q = max(i, j), min(i, j)
            r, s = max(k, l), min(k, l)
            if (p, q) < (r, s)
                p, q, r, s = r, s, p, q
            end
            two_body[(p, q, r, s)] = v
        elseif i > 0 && j > 0 && k == 0 && l == 0
            one_body[(max(i, j), min(i, j))] = v
        elseif i == 0 && j == 0 && k == 0 && l == 0
            core = v
        end
        # anything else (e.g. orbital-energy records "e i 0 0 0") is ignored
    end

    tmp = tempname() * ".fcidump"
    open(tmp, "w") do io
        println(io, "&FCI NORB= $norb,NELEC= $nelec,MS2= $ms2,")
        println(io, " ORBSYM=", join(fill(1, norb), ","), ",")
        println(io, " ISYM= 1,")
        println(io, "&END")
        println(io, core, " 0 0 0 0")
        for ((p, q), v) in one_body
            println(io, v, " ", p, " ", q, " 0 0")
        end
        for ((p, q, r, s), v) in two_body
            println(io, v, " ", p, " ", q, " ", r, " ", s)
        end
    end
    return tmp, norb, nelec, ms2
end

# ------------------------------------------------------------------------------
# DMRG + circuit compilation
# ------------------------------------------------------------------------------

# Schedules follow the chem-state-prep tutorial; shorter/longer runs reuse the last entry.
fit_schedule(v, n) = [v[min(i, length(v))] for i in 1:n]

function dmrg_schedule(nsweeps::Int, maxdim::Int)
    ramp = [2, 4, 10, 30]
    cutoff = [1e-8, 1e-8, 5e-9, 5e-9, 1e-9, 1e-9, 5e-10, 5e-10, 5e-10, 5e-10]
    noise = [1e-7, 1e-7, 5e-8, 5e-8, 1e-8, 1e-8, 5e-9, 1e-9, 0.0, 0.0]

    maxdims = [min(maxdim, i <= length(ramp) ? ramp[i] : maxdim) for i in 1:nsweeps]
    cutoffs = fit_schedule(cutoff, nsweeps)
    noises = fit_schedule(noise, nsweeps)
    noises[end] = 0.0  # always finish noise-free

    sweeps = ITensorMPS.Sweeps(nsweeps)
    ITensorMPS.maxdim!(sweeps, maxdims...)
    ITensorMPS.cutoff!(sweeps, cutoffs...)
    ITensorMPS.noise!(sweeps, noises...)
    return sweeps
end

function run_pipeline(input::String, output::String, opts)
    maxdim = option(opts, "maxdim", 100)
    nsweeps = option(opts, "nsweeps", 10)
    layers = option(opts, "layers", 15)
    tol = option(opts, "tol", 1e-4)
    refine_iters = option(opts, "refine", 20)
    precision = Symbol(option(opts, "precision", "fp64"))
    seed = option(opts, "seed", 2026)

    Random.seed!(seed)

    # 1. FCIDUMP -> MPO Hamiltonian (Jordan-Wigner, 2 spin orbitals per spatial orbital)
    tmp, norb, nelec, ms2 = normalise_fcidump(input)
    H, sites, init_state = try
        MPSCircuits.Chemistry.fcidump_to_mpo(tmp)
    finally
        rm(tmp; force=true)
    end

    # 2. DMRG ground state
    psi0 = ITensorMPS.random_mps(sites, init_state)
    dmrg_energy, psi = ITensorMPS.dmrg(H, psi0, dmrg_schedule(nsweeps, maxdim); outputlevel=0)
    chi = ITensorMPS.maxlinkdim(psi)

    # 3. MPS -> circuit (dense MPS: QN structure is not needed for compilation)
    target = ITensorMPS.dense(psi)
    circuit = compile_mps_circuit(
        target,
        MPSCircuits.DecomposeAllAnalytical();
        n_layers_max=layers,
        tolerance=tol,
        backend=:cpu,
        precision=precision,
    )
    fidelity = evaluate_circuit_fidelity(circuit, target)

    final_circuit, final_fidelity = circuit, fidelity
    if refine_iters > 0
        final_circuit = refine_mps_circuit(
            target,
            circuit;
            n_iterations=refine_iters,
            backend=:cpu,
            precision=precision,
        )
        final_fidelity = evaluate_circuit_fidelity(final_circuit, target)
    end

    # 4. Export (KAK-transpiled to CNOT / Ry / Rz)
    write_circuit_json(final_circuit, output)

    return [
        "num_orbitals" => norb,
        "num_electrons" => nelec,
        "ms2" => ms2,
        "num_qubits" => 2 * norb,
        "dmrg_energy" => dmrg_energy,
        "max_bond_dim" => chi,
        "circuit_fidelity" => fidelity,
        "refined_circuit_fidelity" => final_fidelity,
        "circuit_file" => output,
    ]
end

# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

function main(args)
    positional, opts = parse_cli(args)
    if length(positional) != 2
        println(stderr, USAGE)
        exit(2)
    end
    input, output = positional
    isfile(input) || error("FCIDUMP file not found: $input")

    real_stdout = stdout
    # Anything the libraries print goes to stderr; only the results go to stdout.
    results = redirect_stdout(stderr) do
        run_pipeline(input, output, opts)
    end
    for (key, value) in results
        println(real_stdout, key, "=", value)
    end
end

main(ARGS)