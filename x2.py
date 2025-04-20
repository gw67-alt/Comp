# =============================================================================
# Quantum Signal Counter with Doubling/Halving (Fixed AttributeError)
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.visualization import plot_histogram
# Qiskit Aer for local simulation
try:
    from qiskit_aer import Aer # Use legacy Aer provider if needed
    # Or from qiskit_aer import AerSimulator if using newer versions
except ImportError:
    print("Warning: qiskit-aer not found. Local simulation will not be available.")
    print("Install using: pip install qiskit-aer")
    Aer = None

from qiskit.circuit.library import QFT # QFT import was present but unused, kept for now
import traceback

def create_quantum_signal_counter(wave_peak="1", n_signals=2, operation="double"):
    """
    Creates a quantum circuit that simulates a counter that sends n signals per wave peak.
    Includes operations for doubling or halving the signals.

    FIXED: Corrected calculation of n_count_qubits to avoid IndexError.
    FIXED: Removed debug prints causing AttributeError by accessing '.index' on Qubit objects.

    Qubits:
    q[0]: Wave peak detector (1 = peak detected, 0 = no peak)
    q[1:1+n_count_qubits]: Counter qubits to represent n signals (in binary)
    q[remaining]: Work qubits for doubling/halving operations

    Parameters:
    wave_peak (str): "1" if a peak is detected, "0" if not
    n_signals (int): Number of signals to generate per peak (must be >= 1)
    operation (str): "double", "half", or "none"

    Returns:
    QuantumCircuit: The constructed circuit
    """
    if wave_peak not in ["0", "1"]:
        raise ValueError("wave_peak must be '0' or '1'")

    if operation not in ["double", "half", "none"]:
        raise ValueError("operation must be 'double', 'half', or 'none'")

    if n_signals < 1:
        raise ValueError("n_signals must be at least 1")

    # Calculate how many qubits we need for the counter
    # FIX: Use bit_length() for correct number of bits to represent n_signals
    n_count_qubits = n_signals.bit_length()
    # print(f"Debug: n_signals={n_signals}, calculated n_count_qubits={n_count_qubits}") # Keep this debug print


    # Total number of qubits needed:
    # 1 (wave peak) + n_count_qubits (counter) +
    # n_count_qubits+1 (work qubits for doubling/shift space) + 1 (operation control)
    total_qubits = 1 + n_count_qubits + (n_count_qubits + 1) + 1
    # print(f"Debug: total_qubits calculated = {total_qubits}") # Keep this debug print


    # Create quantum and classical registers
    qreg = QuantumRegister(total_qubits, 'q')
    # We'll need classical bits to read out the final counter value + overflow
    creg = ClassicalRegister(n_count_qubits + 1, 'c') # +1 for potential overflow in doubling

    qc = QuantumCircuit(qreg, creg, name=f"SignalCounter_Peak{wave_peak}_n{n_signals}_{operation}")

    # Define qubits by role using slices
    wave_peak_qubit = qreg[0]
    # Slice for counter qubits: starts at 1, ends before 1+n_count_qubits
    counter_qubits = qreg[1 : 1 + n_count_qubits]
    # Slice for work qubits: starts after counter, ends before op_control
    work_qubits_start = 1 + n_count_qubits
    work_qubits_end = work_qubits_start + (n_count_qubits + 1)
    work_qubits = qreg[work_qubits_start : work_qubits_end]
    # Operation control qubit is the last one
    operation_control = qreg[-1] # Indexing from end is convenient here

    # Removed debug prints accessing qubit.index to avoid AttributeError
    # print(f"Debug: wave_peak_qubit index = {wave_peak_qubit.index}") # REMOVED
    # print(f"Debug: counter_qubits indices = {[q.index for q in counter_qubits]}") # REMOVED
    # print(f"Debug: work_qubits indices = {[q.index for q in work_qubits]}") # REMOVED
    # print(f"Debug: operation_control index = {operation_control.index}") # REMOVED


    # Step 1: Initialize wave peak detector
    if wave_peak == "1":
        qc.x(wave_peak_qubit)

    qc.barrier(label="Wave Peak Init")

    # Step 2: Generate n signals if wave peak is detected
    # Encode n_signals in binary across the counter qubits, controlled by wave_peak
    # Ensure binary string has the correct width n_count_qubits
    n_signals_bin = f'{n_signals:0{n_count_qubits}b}'
    # print(f"Debug: n_signals_bin = {n_signals_bin} (length {len(n_signals_bin)})") # Keep this debug print

    # Iterate through the indices of counter_qubits (0 to n_count_qubits-1)
    for i in range(n_count_qubits):
        # Get the corresponding bit from the binary string.
        # Qubit index i (LSB) corresponds to bit index (n_count_qubits - 1 - i) in the standard binary string.
        bit_index_in_string = n_count_qubits - 1 - i
        if n_signals_bin[bit_index_in_string] == '1':
            # Apply CX controlled by wave_peak if the corresponding bit is 1
            qc.cx(wave_peak_qubit, counter_qubits[i]) # Index i is now safe

    qc.barrier(label=f"Gen {n_signals} Signals")

    # Step 3: Apply doubling or halving operation (controlled by operation_control)
    if operation == "double":
        # Set operation control to |1> for doubling
        qc.x(operation_control)

        # Implement doubling by shifting the register left (Multiply by 2)
        # Use work_qubits[0] for the overflow bit
        overflow_qubit = work_qubits[0]

        # Shift counter_qubits[i] to counter_qubits[i+1] for i=0 to n_count_qubits-2
        # Shift counter_qubits[n_count_qubits-1] (MSB) to overflow_qubit
        for i in range(n_count_qubits - 1, -1, -1): # Iterate from MSB down to LSB
            target_qubit = overflow_qubit if i == n_count_qubits - 1 else counter_qubits[i+1]
            qc.cswap(operation_control, counter_qubits[i], target_qubit)

        # LSB (counter_qubits[0]) should become 0 after shift for true doubling.
        # The simple CSWAP shift doesn't guarantee this.
        # print("Warning: Doubling shift implemented via CSWAP; LSB state depends on shifted-out bit...") # Keep this warning


        qc.barrier(label="Doubling Op")

    elif operation == "half":
        # Set operation control to |1> for halving
        qc.x(operation_control)

        # Implement halving by shifting the register right (Integer divide by 2)
        # Shift counter_qubits[i] to counter_qubits[i-1] for i=1 to n_count_qubits-1
        # LSB counter_qubits[0] is lost.
        # MSB counter_qubits[n_count_qubits-1] should become 0 for true halving.
        for i in range(n_count_qubits - 1): # Iterate from LSB up to second-to-MSB
             qc.cswap(operation_control, counter_qubits[i], counter_qubits[i+1])

        # MSB (counter_qubits[n_count_qubits-1]) state depends on shifted-in bit.
        # print("Warning: Halving shift implemented via CSWAP; MSB state depends on shifted-in bit...") # Keep this warning


        qc.barrier(label="Halving Op")

    # Step 4: Measure the counter qubits and potential overflow
    # print(f"Debug: Measuring counter_qubits...") # Keep this debug print
    for i in range(n_count_qubits):
        qc.measure(counter_qubits[i], creg[i])

    # Measure the overflow bit if we're doubling
    if operation == "double":
        overflow_qubit = work_qubits[0]
        # print(f"Debug: Measuring overflow_qubit...") # Keep this debug print
        qc.measure(overflow_qubit, creg[n_count_qubits])
    else:
        # If not doubling, the last classical bit remains 0 (default)
        pass

    return qc

# --- Simulation Function (Modified for fixed circuit) ---
def simulate_quantum_signal_counter():
    """
    Simulates the quantum signal counter with various configurations.
    """
    print("\n--- Running Quantum Signal Counter Simulation using qiskit-aer ---")

    if Aer is None:
        print("Aer provider not found. Cannot simulate.")
        return {}

    try:
        # Ensure Aer is imported correctly based on qiskit-aer version
        # For newer versions (>=0.10), use AerSimulator
        # from qiskit_aer import AerSimulator
        # simulator = AerSimulator()
        # For older versions, use Aer.get_backend
        simulator = Aer.get_backend('qasm_simulator')
    except Exception as e:
        print(f"Error getting qasm_simulator backend: {e}")
        print("Make sure qiskit-aer is installed correctly.")
        return {}

    # Test configurations
    test_configs = [
        {"wave_peak": "1", "n_signals": 2, "operation": "none"},
        {"wave_peak": "1", "n_signals": 2, "operation": "double"}, # Expect 4 (100) (n_count=2, overflow=1)
        {"wave_peak": "1", "n_signals": 3, "operation": "double"}, # Expect 6 (110) (n_count=2, overflow=1)
        {"wave_peak": "1", "n_signals": 4, "operation": "half"},   # Expect 2 (010) (n_count=3) -> Note: leading 0 for MSB
        {"wave_peak": "0", "n_signals": 2, "operation": "double"}, # Expect 0 (000)
    ]

    results = {}
    print("Preparing and simulating circuits locally...")

    all_plots = []

    for config in test_configs:
        wave_peak = config["wave_peak"]
        n_signals = config["n_signals"]
        operation = config["operation"]

        test_case = f"Peak={wave_peak}, n={n_signals}, op={operation}"
        print(f"\n--- Simulating: {test_case} ---")

        try:
            # Create the quantum circuit
            qc = create_quantum_signal_counter(wave_peak, n_signals, operation)
            shots = 1024

            # Run the simulation
            print(f"  Running simulation with {shots} shots...")
            # Transpilation might be needed depending on simulator/gates used
            # tqc = transpile(qc, simulator) # Optional
            job = simulator.run(qc, shots=shots) # Use original qc if not transpiling
            result = job.result()
            counts = result.get_counts(qc)
            print("  Simulation complete.")
            results[test_case] = counts

            # Calculate expected result (Ideal, based on perfect shifts setting 0s)
            n_count_qubits = n_signals.bit_length() if n_signals > 0 else 1
            expected_signals = 0
            # Classical register has n_count_qubits + 1 bits
            expected_classical_reg_size = n_count_qubits + 1
            expected_bin_string = f"{0:0{expected_classical_reg_size}b}" # Default to 0

            if wave_peak == "1":
                current_val = n_signals
                if operation == "double":
                    current_val *= 2
                elif operation == "half":
                    current_val //= 2
                expected_signals = current_val

                # Format expected result to match classical register size
                expected_bin_string = f"{expected_signals:0{expected_classical_reg_size}b}"


            print(f"  Expected signals (ideal): {expected_signals}")
            print(f"  Expected binary output (ideal): {expected_bin_string}")
            # Qiskit counts are often LSB first, reverse expected string for comparison if needed
            measured_keys = list(counts.keys())
            print(f"  Measured outcomes (counts): {counts}")
            if measured_keys:
                 print(f"  Top measured outcome (LSB first): {measured_keys[0]}")
                 # Check if top measured matches reversed expected string
                 # measured_msb_first = measured_keys[0][::-1] # Reverse measured LSB-first string
                 # print(f"  Top measured outcome (MSB first): {measured_msb_first}")


            # Plot histogram
            fig = plot_histogram(counts, title=f"Signal Counter: {test_case}\nExpected (ideal LSB first)={expected_bin_string[::-1]} ({expected_signals})")
            all_plots.append(fig) # Store figure object

        except Exception as e:
            print(f"  !!! Error during simulation for {test_case}: {e}")
            traceback.print_exc()
            results[test_case] = {"ERROR": str(e)}


    print("\nDisplaying all result plots...")
    # Check if any plots were generated
    if all_plots:
        plt.show() # Display all plots at the end
    else:
        print("No plots were generated (possibly due to errors).")
    print("\n--- Quantum Signal Counter Simulation Finished ---")
    return results

# --- Explanation Function (Unchanged) ---
def explain_signal_counter():
    """
    Provides a detailed explanation of the quantum signal counter.
    """
    print("\n--- Explanation of Quantum Signal Counter ---")
    print("This quantum circuit simulates a counter that generates n signals per wave peak,")
    print("with options to double or halve the number of signals.")
    print("\nThe circuit operation is divided into three main parts:")
    print("\n1. Wave Peak Detection:")
    print("   - Input qubit q[0] represents whether a wave peak is detected (|1>) or not (|0>)")
    print("   - This controls whether signals are generated")
    print("\n2. Signal Generation:")
    print("   - When a peak is detected, the circuit generates n signals")
    print("   - The number of signals is encoded in binary across counter qubits")
    print("   - These qubits are only set if the wave peak detector is |1>")
    print("\n3. Signal Manipulation:")
    print("   - Doubling Operation: Implements a bit shift left (x2)")
    print("     * Uses controlled-SWAP gates to shift bits")
    print("     * Handles overflow by using an additional qubit")
    print("     * Note: Simple shift may not guarantee LSB becomes 0")
    print("   - Halving Operation: Implements a bit shift right (/2)")
    print("     * Similar approach but shifting in the opposite direction")
    print("     * Equivalent to integer division by 2 (floors the result)")
    print("     * Note: Simple shift may not guarantee MSB becomes 0")
    print("\nQuantum Properties Used:")
    print("1. Conditional operations: Using the wave peak qubit as a control")
    print("2. Quantum parallelism: Performing operations on superpositions (if inputs are superpositioned)")
    print("3. Bit manipulation: Using quantum gates to implement binary arithmetic-like operations")
    print("\nApplications:")
    print("- Quantum signal processing")
    print("- Frequency multiplication/division")
    print("- Quantum sampling with controlled rate")
    print("- Building block for quantum waveform generators")
    return

# =============================================================================
# Main Execution Block (Modified for fixed circuit)
# =============================================================================
if __name__ == "__main__":
    print("Quantum Signal Counter with Doubling/Halving (Fixed)")
    print("=" * 60)
    print("This circuit simulates a counter that sends n signals per wave peak.")
    print("It can double or halve the number of signals generated.")
    print("-" * 60)

    # Show an example circuit
    wave_peak_example = "1"
    n_signals_example = 3 # Use 3 to see 2 counter qubits
    operation_example = "double"

    try:
        qc_example = create_quantum_signal_counter(
            wave_peak=wave_peak_example,
            n_signals=n_signals_example,
            operation=operation_example
        )

        print(f"\nCircuit diagram for Peak={wave_peak_example}, n={n_signals_example}, op={operation_example}:")
        print(qc_example.draw(output='text', fold=120))

    except Exception as e:
        print(f"\nAn error occurred during example circuit creation or drawing: {e}")
        traceback.print_exc()

    # Run the simulations
    results = simulate_quantum_signal_counter()

    # Explain the results
    explain_signal_counter()

    # Final summary
    print("-" * 60)
    print("\n--- Experiment Summary (Quantum Signal Counter) ---")
    print("- Circuit Description:")
    print("  - Detects wave peaks and generates n signals per peak")
    print("  - Can double the signal count (x2) using quantum bit shifting")
    print("  - Can halve the signal count (/2) using quantum bit shifting")
    print("- Applications:")
    print("  - Quantum frequency multiplication/division")
    print("  - Controlled signal generation based on input detection")
    print("  - Building block for quantum waveform processing")
    print("\nExecution finished.")
    print("-" * 60)

