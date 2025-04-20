# =============================================================================
# Quantum Russian Peasant Multiplier (State Preparation Version) for IBM Runtime
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.visualization import plot_histogram # Still imported, but call will be commented out
import traceback
import datetime # For timestamp

# Qiskit Aer for local simulation (optional fallback)
try:
    from qiskit_aer import AerSimulator
except ImportError:
    print("Warning: qiskit-aer not found. Local simulation fallback will not be available.")
    AerSimulator = None

# Qiskit IBM Runtime
# Ensure qiskit-ibm-runtime is installed (pip install qiskit-ibm-runtime)
try:
    from qiskit_ibm_runtime import (
        QiskitRuntimeService,
        Sampler,              # Use Sampler primitive (likely V2)
        Session
        # Options removed as we need the specific SamplerOptions
    )
    # Import the specific options class for Sampler
    from qiskit_ibm_runtime.options import SamplerOptions
    print("qiskit-ibm-runtime imported successfully.")
except ImportError as e:
      print(f"Error importing qiskit_ibm_runtime components: {e}")
      print("IBM Runtime execution will not be available.")
      QiskitRuntimeService = None
      Sampler = None
      Session = None
      SamplerOptions = None # Set placeholder if import fails

# --- IBM Account Setup (Run once if needed) ---
# Uncomment and run the following lines ONCE to save your API token if you haven't already.
# Replace 'YOUR_IBM_QUANTUM_API_TOKEN' with your actual token.
# try:
#     if QiskitRuntimeService:
#         QiskitRuntimeService.save_account(
#             token='YOUR_IBM_QUANTUM_API_TOKEN', # Replace with your actual token
#             channel='ibm_quantum',          # Typically 'ibm_quantum'
#             instance='ibm-q/open/main',     # Common free access instance
#             overwrite=True                  # Set to True to overwrite existing saved credentials
#         )
#         print("IBM Quantum account credentials saved/updated.")
#     else:
#         print("QiskitRuntimeService not available, skipping account save.")
# except NameError:
#     print("QiskitRuntimeService class not found, skipping account save.")
# except Exception as e:
#     print(f"An error occurred during IBM Quantum account setup: {e}")
# --- End IBM Account Setup ---


# =============================================================================
# Helper Functions
# =============================================================================

def compute_russian_peasant_steps(a, b):
    """
    Calculate the steps needed for Russian Peasant multiplication classically.
    This is for informational purposes and understanding the algorithm.
    The quantum circuit generated below uses the final result directly.

    Returns:
        tuple: (doubling_steps, addition_values)
    """
    addition_values = []
    print(f"\nClassical Russian Peasant steps for {a} × {b}:")
    print(f"{'Step':<4} | {'a (Doubling)':<15} | {'b (Halving)':<15} | {'Add to result?'}")
    print("-" * 4 + "-+-" + "-" * 15 + "-+-" + "-" * 15 + "-+-" + "-" * 15)

    step = 1
    current_a = a
    current_b = b

    while current_b > 0:
        add_marker = ""
        if current_b % 2 == 1:
            addition_values.append(current_a)
            add_marker = f"Yes, add {current_a}"
        else:
            add_marker = "No"
        print(f"{step:<4} | {current_a:<15} | {current_b:<15} | {add_marker}")

        current_a *= 2 # Double a
        current_b //= 2 # Halve b (integer division)
        step += 1

    doubling_steps = step - 1
    print("-" * 55)
    return doubling_steps, addition_values

def create_quantum_rpm_state_prep(wave_peak="1", a=2, b=3, verbose=True):
    """
    Creates a quantum circuit that prepares the state corresponding to the
    result of a*b, controlled by a wave_peak qubit.

    *** This circuit performs state preparation based on a classical calculation, ***
    *** it does NOT implement quantum multiplication via arithmetic gates. ***

    Parameters:
        wave_peak (str): "1" if a peak is detected, "0" if not
        a (int): First number in multiplication (a × b)
        b (int): Second number in multiplication (a × b)
        verbose (bool): Whether to print detailed steps

    Returns:
        QuantumCircuit: The constructed circuit
    """
    if wave_peak not in ["0", "1"]:
        raise ValueError("wave_peak must be '0' or '1'")
    if a < 0 or b < 0: # Allow 0 for calculation, but maybe not for steps?
        raise ValueError("Inputs a and b must be non-negative integers for this implementation.")

    # Calculate the expected final result (a × b) CLASSICALLY
    expected_result = a * b
    # Determine bits needed for the result. Handle result=0 case.
    result_bits_needed = expected_result.bit_length() if expected_result > 0 else 1

    if verbose:
        # Compute classical steps just for printing explanation
        compute_russian_peasant_steps(a, b)
        print(f"\nTarget state preparation for {a} × {b}")
        print(f"Classical result: {expected_result} (binary: {bin(expected_result)[2:]})")
        print(f"Result bits needed: {result_bits_needed}")

    # Use result_bits_needed for the result register size
    n_result_qubits = result_bits_needed

    # Create quantum and classical registers
    # 1 qubit for wave_peak + n_result_qubits for result
    total_qubits = 1 + n_result_qubits
    qreg = QuantumRegister(total_qubits, 'q')
    creg = ClassicalRegister(n_result_qubits, 'c') # Measure only the result bits

    # Create the circuit
    qc = QuantumCircuit(qreg, creg, name=f"RPM_StatePrep_{a}x{b}")

    # Define qubits by role
    wave_peak_qubit = qreg[0]
    result_register = qreg[1 : 1 + n_result_qubits] # Qubits q[1] onwards

    # Step 1: Initialize wave peak detector
    if wave_peak == "1":
        qc.x(wave_peak_qubit)
    qc.barrier(label="Wave Peak Init")

    # Step 2: Prepare the result state |expected_result> if wave_peak is |1>
    # If wave_peak is |0>, result register remains |0...0>
    result_bin = f'{expected_result:0{n_result_qubits}b}' # Format to required width

    if verbose:
        print(f"Target result binary string: {result_bin} (length: {len(result_bin)})")

    # Apply CX gates controlled by wave_peak to set the result bits
    for i in range(n_result_qubits):
        # Qubit index i (LSB) corresponds to bit index (n_result_qubits - 1 - i)
        bit_index_in_string = n_result_qubits - 1 - i
        if result_bin[bit_index_in_string] == '1':
            qc.cx(wave_peak_qubit, result_register[i])

    qc.barrier(label=f"Prep State |{expected_result}>")

    # Step 3: Measure the result register
    # Measure result_register qubits into classical bits 0 to n_result_qubits-1
    qc.measure(result_register, creg)

    return qc

# =============================================================================
# Simulation and Execution Functions
# =============================================================================

def simulate_local(test_configs):
    """Simulates the circuits locally using qiskit-aer."""
    print("\n=== Testing Locally using qiskit-aer ===")
    if AerSimulator is None:
        print("AerSimulator not found. Cannot simulate locally.")
        return {}

    results = {}
    # all_plots = [] # Removed plotting
    try:
        simulator = AerSimulator()
    except Exception as e:
        print(f"Error initializing AerSimulator: {e}")
        return {}

    for config in test_configs:
        wave_peak = config["wave_peak"]
        a = config["a"]
        b = config["b"]
        expected = config["expected"]
        test_case = f"LocalSim: Peak={wave_peak}, {a}×{b}"
        print(f"\n--- Simulating: {test_case} ---")

        try:
            qc = create_quantum_rpm_state_prep(wave_peak, a, b, verbose=False) # Less verbose for sim loop
            n_result_qubits = expected.bit_length() if expected > 0 else 1
            expected_bin_string = f"{expected:0{n_result_qubits}b}"
            shots = 1024

            print(f"  Running simulation with {shots} shots...")
            tqc = transpile(qc, simulator) # Transpile for simulator
            job = simulator.run(tqc, shots=shots)
            result = job.result()
            counts = result.get_counts(tqc)
            print("  Simulation complete.")
            results[test_case] = counts

            # Sort counts by value (shots) descending for better readability
            sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)

            measured_keys = list(counts.keys())
            print(f"  Expected binary (ideal): {expected_bin_string}")
            # FIX: Print sorted counts
            #print(f"  Measured outcomes (sorted by count): {sorted_counts}")
            if measured_keys:
                # Get the top measured outcome (first element after sorting)
                top_outcome_lsb = sorted_counts[0][0]
                print(f"  Top measured (LSB first): {top_outcome_lsb}")

            # Plotting removed as requested
            # fig = plot_histogram(counts, title=f"{test_case}\nExpected(LSB first)={expected_bin_string[::-1]}")
            # all_plots.append(fig)

        except Exception as e:
            print(f"  !!! Error during simulation for {test_case}: {e}")
            traceback.print_exc()
            results[test_case] = {"ERROR": str(e)}

    # Plotting removed as requested
    # print("\nDisplaying local simulation plots...")
    # if all_plots: plt.show()
    # else: print("No plots generated.")
    print("\nLocal simulation finished. Plotting was disabled.")
    return results

def run_on_ibm_hardware(test_configs, backend_name=None, use_simulator=False):
    """Runs the circuits on IBM Quantum hardware or simulator via Runtime."""
    print(f"\n=== Testing on IBM Backend ({'Simulator' if use_simulator else 'Hardware'}) ===")
    if not QiskitRuntimeService or not SamplerOptions: # Check if specific options class was imported
        print("QiskitRuntimeService or SamplerOptions not available. Cannot run on IBM backend.")
        return {}

    try:
        # Initialize service (assumes account is saved)
        service = QiskitRuntimeService()
        print(f"Initialized QiskitRuntimeService (Account: {service.active_account()})")
    except Exception as e:
        print(f"Failed to initialize IBM Runtime Service: {e}")
        traceback.print_exc()
        return {}

    # --- Backend Selection ---
    backend = None
    if backend_name:
        print(f"Attempting to use specified backend: {backend_name}")
        try:
            backend = service.backend(backend_name)
        except Exception as e:
            print(f"Error getting specified backend '{backend_name}': {e}")
            return {}
    elif use_simulator:
        # Find a suitable simulator backend
        sim_backends = service.backends(simulator=True, operational=True)
        if not sim_backends:
            print("No operational IBM simulators found.")
            return {}
        # Prefer newer simulators if possible, e.g., simulator_stabilizer
        preferred_sims = [s for s in sim_backends if 'stabilizer' in s.name]
        if preferred_sims: backend = preferred_sims[0]
        else: backend = sim_backends[0] # Fallback to any simulator
        print(f"Using IBM simulator backend: {backend.name}")
    else:
        # Find the least busy suitable hardware backend
        # Determine max qubits needed from test_configs
        max_qubits = 0
        for cfg in test_configs:
             res_bits = (cfg['a'] * cfg['b']).bit_length() if (cfg['a'] * cfg['b']) > 0 else 1
             max_qubits = max(max_qubits, 1 + res_bits)
        print(f"Searching for least busy hardware with >= {max_qubits} qubits...")
        try:
            # Use service.least_busy() for convenience
            hardware_backends = service.least_busy(min_num_qubits=max_qubits, simulator=False, operational=True)
            backend = hardware_backends
            print(f"Using least busy hardware backend: {backend.name}")
        except Exception as e:
             print(f"Could not find suitable IBM hardware: {e}")
             return {}


    if not backend:
        print("Backend selection failed.")
        return {}
    print(f"Using backend: {backend.name} (Max qubits: {backend.num_qubits})")

    # --- Prepare Circuits ---
    circuits_to_run_map = {} # Map config index to circuit
    print("\nPreparing and transpiling circuits...")
    transpiled_circuits = []
    circuit_metadata = [] # Store config for result processing

    for i, config in enumerate(test_configs):
        wave_peak = config["wave_peak"]
        a = config["a"]
        b = config["b"]
        expected = config["expected"]
        test_case = f"IBM({backend.name}): Peak={wave_peak}, {a}×{b}"
        print(f"  Preparing circuit for: {test_case}")
        try:
            qc = create_quantum_rpm_state_prep(wave_peak, a, b, verbose=False)
            # Check qubit count against backend limit
            if qc.num_qubits > backend.num_qubits:
                 print(f"    SKIPPING: Circuit requires {qc.num_qubits} qubits, backend '{backend.name}' has {backend.num_qubits}.")
                 continue

            print(f"    Transpiling circuit with {qc.num_qubits} qubits...")
            # Use optimization level 1 for mapping, 3 for more reduction
            # Optimization level is set here during transpile
            tqc = transpile(qc, backend=backend, optimization_level=1)
            transpiled_circuits.append(tqc)
            circuit_metadata.append({**config, "test_case": test_case}) # Store config
            print(f"    Transpilation done (Depth: {tqc.depth()}, Ops: {len(tqc.count_ops())}).")

        except Exception as e:
            print(f"  !!! Error preparing/transpiling circuit for {test_case}: {e}")
            traceback.print_exc()

    if not transpiled_circuits:
        print("No circuits were successfully prepared for execution.")
        return {}

    # --- Execute Job ---
    results = {}
    # all_plots = [] # Removed plotting
    # Define shots directly for the run method
    num_shots = 4096

    # Use the specific SamplerOptions class
    options = SamplerOptions()
    # Set specific options if needed, e.g.:
    # options.resilience_level = 1 # Enable basic error mitigation if desired/supported
    # options.max_execution_time = 300 # Example: set max execution time in seconds

    print(f"\nSubmitting job with {len(transpiled_circuits)} circuits to {backend.name} (Shots={num_shots})...")
    try:
        # Use Session for managing jobs on hardware/simulators
        with Session(backend=backend) as session:
            print(f"Session active (ID: {session.session_id})")
            # Initialize Sampler within the session, passing the correct options type
            sampler = Sampler(options=options) # No session= needed here
            # Pass circuits as a positional argument, shots as keyword
            job = sampler.run(transpiled_circuits, shots=num_shots)
            print(f"Job submitted (ID: {job.job_id()}). Waiting for results...")
            primitive_result = job.result() # Wait for job completion
            print("Job completed.")

            # --- Process Results ---
            print("\nProcessing results...")
            if len(primitive_result) != len(circuit_metadata):
                 print(f"Warning: Number of results ({len(primitive_result)}) != number of circuits submitted ({len(circuit_metadata)}).")

            for i, pub_result in enumerate(primitive_result):
                 if i >= len(circuit_metadata): continue # Avoid index error if results mismatch
                 metadata = circuit_metadata[i]
                 test_case = metadata["test_case"]
                 expected = metadata["expected"]
                 print(f"\n--- Result for: {test_case} ---")

                 try:
                     # Sampler V2 returns data in pub_result.data.<creg_name>
                     # Assuming creg name is 'c' as defined in the circuit
                     if not hasattr(pub_result, 'data') or not hasattr(pub_result.data, 'c'):
                          print("  !!! Error: Result structure missing 'data.c'.")
                          print(f"  Result content: {pub_result}")
                          results[test_case] = {"ERROR": "Result structure missing data.c"}
                          continue

                     counts = pub_result.data.c.get_counts() # Get counts from DataBin
                     results[test_case] = counts

                     # Sort counts by value (shots) descending for better readability
                     sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)

                     n_result_qubits = expected.bit_length() if expected > 0 else 1
                     expected_bin_string = f"{expected:0{n_result_qubits}b}"

                     measured_keys = list(counts.keys())
                     print(f"  Expected binary (ideal): {expected_bin_string}")
                     # FIX: Print sorted counts
                     print(f"  Measured outcomes (sorted by count): {sorted_counts}")
                     # Qiskit counts keys are LSB-first strings
                     if measured_keys:
                         # Get the top measured outcome (first element after sorting)
                         top_outcome_lsb = sorted_counts[0][0]
                         print(f"  Top measured (LSB first): {top_outcome_lsb}")


                     # Plotting removed as requested
                     # fig = plot_histogram(counts, title=f"{test_case}\nExpected(LSB first)={expected_bin_string[::-1]}")
                     # all_plots.append(fig)

                 except Exception as proc_err:
                      print(f"  !!! Error processing result for {test_case}: {proc_err}")
                      traceback.print_exc()
                      results[test_case] = {"ERROR": str(proc_err)}

    except Exception as e:
        print(f"\n!!! Error during IBM Runtime execution: {e}")
        traceback.print_exc()
        # Store partial results if any
        return results

    # Plotting removed as requested
    # print(f"\nDisplaying plots from {backend.name} run...")
    # if all_plots: plt.show()
    # else: print("No plots generated.")
    print(f"\nIBM run on {backend.name} finished. Plotting was disabled.")
    return results


# =============================================================================
# Main Execution Block
# =============================================================================
if __name__ == "__main__":
    print("Quantum Russian Peasant Multiplication (State Prep Version)")
    print("=" * 70)
    print("*** IMPORTANT: This circuit prepares the state |result> based on ***")
    print("*** classical pre-calculation. It does NOT perform quantum multiplication. ***")
    print("-" * 70)

    # --- Example Circuit ---
    wave_peak_example = "1"
    a_example = 7
    b_example = 3
    try:
        print(f"\n--- Example Quantum Circuit for State Prep of {a_example} × {b_example} ---")
        qc_example = create_quantum_rpm_state_prep(
            wave_peak=wave_peak_example, a=a_example, b=b_example, verbose=True
        )
        print(f"\nCircuit diagram:")
        print(qc_example.draw(output='text', fold=80))
    except Exception as e:
        print(f"\nAn error occurred during example circuit creation or drawing: {e}")
        traceback.print_exc()

    # --- Test Configurations ---
    test_configs = [
        {"wave_peak": "1", "a": 3, "b": 50, "expected": 150},
        {"wave_peak": "1", "a": 7, "b": 3, "expected": 21},
        {"wave_peak": "1", "a": 4, "b": 4, "expected": 16},
        {"wave_peak": "1", "a": 3, "b": 6, "expected": 18},
        {"wave_peak": "0", "a": 5, "b": 5, "expected": 0}, # Expect 0 if peak is 0
    ]

    # --- Choose Execution Mode ---
    print("\nChoose Execution Mode:")
    print("  1. Local Simulator (qiskit-aer)")
    print("  2. IBM Quantum Simulator (via Runtime)")
    print("  3. IBM Quantum Hardware (Least Busy, via Runtime)")
    print("  4. IBM Quantum Hardware (Specify Name, via Runtime)")
    run_option = input("Enter option number (1-4): ").strip()

    if run_option == '1':
        local_results = simulate_local(test_configs)
        # print("\nLocal Simulation Results Summary:")
        # print(local_results)
    elif run_option == '2':
        if QiskitRuntimeService:
            ibm_results = run_on_ibm_hardware(test_configs, use_simulator=True)
            # print("\nIBM Simulator Results Summary:")
            # print(ibm_results)
        else: print("IBM Runtime not available.")
    elif run_option == '3':
         if QiskitRuntimeService:
            ibm_results = run_on_ibm_hardware(test_configs, use_simulator=False, backend_name=None)
            # print("\nIBM Hardware Results Summary:")
            # print(ibm_results)
         else: print("IBM Runtime not available.")
    elif run_option == '4':
         if QiskitRuntimeService:
            backend_input = input("Enter the specific IBM Quantum backend name: ").strip()
            if backend_input:
                ibm_results = run_on_ibm_hardware(test_configs, use_simulator=False, backend_name=backend_input)
                # print("\nIBM Hardware Results Summary:")
                # print(ibm_results)
            else: print("No backend name entered.")
         else: print("IBM Runtime not available.")
    else:
        print("Invalid option selected.")

    # Final summary
    print("-" * 70)
    print("\n--- Execution Summary ---")
    print("This script demonstrated preparing quantum states corresponding to")
    print("the result of a*b, controlled by a 'wave peak' qubit.")
    print("It does not perform the multiplication steps using quantum arithmetic.")
    print("\nExecution finished.")
    print("-" * 70)

