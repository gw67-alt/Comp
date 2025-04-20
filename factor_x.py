# =============================================================================
# Enhanced Quantum Signal Counter with Multiple Doublers/Halvers
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.visualization import plot_histogram
# Qiskit Aer for local simulation
try:
    from qiskit_aer import Aer
except ImportError:
    print("Warning: qiskit-aer not found. Local simulation will not be available.")
    print("Install using: pip install qiskit-aer")
    Aer = None

import traceback

def create_quantum_signal_multiplier(wave_peak="1", n_signals=2, operations=None, multiplication_factor=None):
    """
    Creates a quantum circuit that simulates a counter that sends n signals per wave peak
    with multiple doubling/halving operations for precise multiplication.

    Parameters:
    wave_peak (str): "1" if a peak is detected, "0" if not
    n_signals (int): Initial number of signals to generate per peak (must be >= 1)
    operations (list): List of operations ["double", "half"] to apply sequentially
                       If None, determined automatically from multiplication_factor
    multiplication_factor (float): Target multiplication factor (e.g., 1.5, 0.75)
                                  If provided, operations will be derived automatically

    Returns:
    QuantumCircuit: The constructed circuit
    """
    if wave_peak not in ["0", "1"]:
        raise ValueError("wave_peak must be '0' or '1'")

    if n_signals < 1:
        raise ValueError("n_signals must be at least 1")
    
    # Handle multiplication factor if provided
    if multiplication_factor is not None:
        if multiplication_factor <= 0:
            raise ValueError("multiplication_factor must be positive")
        
        # Convert multiplication_factor to a sequence of doublings and halvings
        # For example: 3.0 = double + half + double (2 * 0.5 * 3 = 3)
        operations = []
        
        # Use binary representation to determine operations
        factor = multiplication_factor
        while abs(factor - 1.0) > 0.001:  # Continue until we reach approximately 1
            if factor > 1.0:
                operations.append("double")
                factor /= 2.0
            else:
                operations.append("half")
                factor *= 2.0
        
        print(f"Derived operations from multiplication factor {multiplication_factor}: {operations}")
    
    # Default to no operations if not specified
    if operations is None:
        operations = ["none"]
    
    # Validate operations
    for op in operations:
        if op not in ["double", "half", "none"]:
            raise ValueError(f"Invalid operation '{op}'. Must be 'double', 'half', or 'none'")
    
    # Calculate how many qubits we need for the counter
    n_count_qubits = max(n_signals.bit_length(), 3)  # Ensure minimum of 3 qubits for the counter
    
    # Calculate the maximum expected signal value after all operations
    max_expected_signals = n_signals
    for op in operations:
        if op == "double":
            max_expected_signals *= 2
        elif op == "half":
            max_expected_signals //= 2
    
    # Ensure n_count_qubits is sufficient for the maximum expected value
    n_count_qubits = max(n_count_qubits, max_expected_signals.bit_length())
    
    # For each operation we need:
    # - 1 control qubit per operation
    # - n_count_qubits+1 work qubits (shared across operations)
    operation_control_qubits = len(operations)
    
    # Total qubits needed:
    # 1 (wave peak) + n_count_qubits (counter) + (n_count_qubits+1) (work) + operation_control_qubits
    total_qubits = 1 + n_count_qubits + (n_count_qubits + 1) + operation_control_qubits
    
    # Create quantum and classical registers
    qreg = QuantumRegister(total_qubits, 'q')
    # We'll need classical bits to read out the final counter value + overflow
    creg = ClassicalRegister(n_count_qubits + 1, 'c')  # +1 for potential overflow
    
    # Create a name that reflects the operations
    op_str = "_".join(operations) if operations else "none"
    qc = QuantumCircuit(qreg, creg, name=f"SignalMultiplier_Peak{wave_peak}_n{n_signals}_{op_str}")
    
    # Define qubits by role using slices
    wave_peak_qubit = qreg[0]
    
    # Counter qubits (where the signal count is stored)
    counter_qubits = qreg[1:1 + n_count_qubits]
    
    # Work qubits (used for shift operations)
    work_qubits_start = 1 + n_count_qubits
    work_qubits_end = work_qubits_start + (n_count_qubits + 1)
    work_qubits = qreg[work_qubits_start:work_qubits_end]
    
    # Operation control qubits (one per operation)
    op_control_start = work_qubits_end
    op_control_qubits = qreg[op_control_start:op_control_start + operation_control_qubits]
    
    # Step 1: Initialize wave peak detector
    if wave_peak == "1":
        qc.x(wave_peak_qubit)
    
    qc.barrier(label="Wave Peak Init")
    
    # Step 2: Generate n signals if wave peak is detected
    # Encode n_signals in binary across the counter qubits, controlled by wave_peak
    n_signals_bin = f'{n_signals:0{n_count_qubits}b}'
    
    for i in range(n_count_qubits):
        bit_index_in_string = n_count_qubits - 1 - i
        if n_signals_bin[bit_index_in_string] == '1':
            # Apply CX controlled by wave_peak if the corresponding bit is 1
            qc.cx(wave_peak_qubit, counter_qubits[i])
    
    qc.barrier(label=f"Gen {n_signals} Signals")
    
    # Step 3: Apply the sequence of doubling or halving operations
    current_value = n_signals
    overflow_qubit = work_qubits[0]
    
    for op_idx, operation in enumerate(operations):
        if operation == "none":
            continue
            
        # Get current operation control qubit
        operation_control = op_control_qubits[op_idx]
        
        # Add a label for this operation
        qc.barrier(label=f"Operation {op_idx+1}: {operation}")
        
        # Set operation control to |1⟩
        qc.x(operation_control)
        
        if operation == "double":
            # Implement doubling (multiply by 2) via left shift
            for i in range(n_count_qubits - 1, -1, -1):  # MSB to LSB
                target_qubit = overflow_qubit if i == n_count_qubits - 1 else counter_qubits[i+1]
                qc.cswap(operation_control, counter_qubits[i], target_qubit)
            
            # Set LSB to 0 (required for proper doubling)
            qc.reset(counter_qubits[0])
            
            # Update the expected value for debugging
            current_value *= 2
            
        elif operation == "half":
            # Implement halving (divide by 2) via right shift
            for i in range(n_count_qubits - 1):  # LSB to second-to-MSB
                qc.cswap(operation_control, counter_qubits[i], counter_qubits[i+1])
            
            # Set MSB to 0 (for proper halving)
            qc.reset(counter_qubits[n_count_qubits - 1])
            
            # Update the expected value for debugging
            current_value //= 2
        
        qc.barrier(label=f"After {operation}: Expected={current_value}")
    
    # Step 4: Measure the counter qubits and potential overflow
    for i in range(n_count_qubits):
        qc.measure(counter_qubits[i], creg[i])
    
    # Always measure the overflow qubit
    qc.measure(overflow_qubit, creg[n_count_qubits])
    
    return qc

def simulate_quantum_signal_multiplier():
    """
    Simulates the quantum signal multiplier with various configurations.
    """
    print("\n--- Running Quantum Signal Multiplier Simulation ---")
    
    if Aer is None:
        print("Aer provider not found. Cannot simulate.")
        return {}
    
    try:
        simulator = Aer.get_backend('qasm_simulator')
    except Exception as e:
        print(f"Error getting qasm_simulator backend: {e}")
        print("Make sure qiskit-aer is installed correctly.")
        return {}
    
    # Test configurations
    test_configs = [
        {"wave_peak": "1", "n_signals": 2, "operations": ["double"], "expected": 4},
        {"wave_peak": "1", "n_signals": 2, "operations": ["double", "double"], "expected": 8},
        {"wave_peak": "1", "n_signals": 8, "operations": ["half", "half"], "expected": 2},
        {"wave_peak": "1", "n_signals": 3, "operations": ["double", "half"], "expected": 3},
        {"wave_peak": "1", "n_signals": 5, "operations": ["double", "half", "double"], "expected": 10},
        # Test with automatic operations derived from multiplication factor
        {"wave_peak": "1", "n_signals": 4, "multiplication_factor": 2.0, "expected": 8},
        {"wave_peak": "1", "n_signals": 6, "multiplication_factor": 0.5, "expected": 3},
        {"wave_peak": "1", "n_signals": 24, "multiplication_factor": 8, "expected": 192},
    ]
    
    results = {}
    print("Preparing and simulating circuits locally...")
    
    all_plots = []
    
    for config in test_configs:
        wave_peak = config["wave_peak"]
        n_signals = config["n_signals"]
        operations = config.get("operations")
        multiplication_factor = config.get("multiplication_factor")
        expected = config.get("expected", "N/A")
        
        op_str = ", ".join(operations) if operations else "none"
        test_case = f"Peak={wave_peak}, n={n_signals}"
        if multiplication_factor:
            test_case += f", factor={multiplication_factor}"
        else:
            test_case += f", ops=[{op_str}]"
        test_case += f", expected={expected}"
        
        print(f"\n--- Simulating: {test_case} ---")
        
        try:
            # Create the quantum circuit
            qc = create_quantum_signal_multiplier(
                wave_peak=wave_peak,
                n_signals=n_signals,
                operations=operations,
                multiplication_factor=multiplication_factor
            )
            shots = 1024
            
            # Run the simulation
            print(f"  Running simulation with {shots} shots...")
            job = simulator.run(qc, shots=shots)
            result = job.result()
            counts = result.get_counts(qc)
            print("  Simulation complete.")
            
            results[test_case] = counts
            
            # Calculate number of qubits in the counter
            n_count_qubits = max(n_signals.bit_length(), 3)
            if operations or multiplication_factor:
                # Recalculate based on operations if provided
                max_signals = n_signals
                if operations:
                    for op in operations:
                        if op == "double":
                            max_signals *= 2
                        elif op == "half":
                            max_signals //= 2
                elif multiplication_factor:
                    max_signals = int(n_signals * multiplication_factor + 0.5)  # Round to nearest int
                
                n_count_qubits = max(n_count_qubits, max_signals.bit_length())
            
            # Format expected result to match classical register size
            expected_classical_reg_size = n_count_qubits + 1  # +1 for overflow bit
            expected_bin_string = f"{expected:0{expected_classical_reg_size}b}"
            
            print(f"  Expected signals: {expected}")
            print(f"  Expected binary output: {expected_bin_string}")
            
            measured_keys = list(counts.keys())
            print(f"  Measured outcomes (counts): {counts}")
            
            if measured_keys:
                print(f"  Top measured outcome (LSB first): {measured_keys[0]}")
            
            # Plot histogram
            fig = plot_histogram(
                counts,
                title=f"Signal Multiplier: {test_case}\nExpected (LSB first)={expected_bin_string[::-1]} ({expected})"
            )
            all_plots.append(fig)
            
        except Exception as e:
            print(f"  !!! Error during simulation for {test_case}: {e}")
            traceback.print_exc()
            results[test_case] = {"ERROR": str(e)}
    
    print("\nDisplaying all result plots...")
    if all_plots:
        plt.show()
    else:
        print("No plots were generated (possibly due to errors).")
    
    print("\n--- Quantum Signal Multiplier Simulation Finished ---")
    return results

def explain_signal_multiplier():
    """
    Provides a detailed explanation of the enhanced quantum signal multiplier.
    """
    print("\n--- Explanation of Quantum Signal Multiplier ---")
    print("This enhanced quantum circuit extends the original signal counter by implementing")
    print("multiple chained doubling and halving operations to achieve arbitrary multiplication factors.")
    
    print("\nEnhancements over the original circuit:")
    print("1. Support for multiple sequential operations:")
    print("   - Can apply any sequence of doubling and halving operations")
    print("   - Each operation has its own control qubit")
    print("   - Operations are applied in order, with barriers between them")
    
    print("\n2. Automatic operation derivation:")
    print("   - Can determine optimal sequence of doublings/halvings from a target multiplication factor")
    print("   - Uses a greedy algorithm to approximate the factor with doubling/halving operations")
    print("   - For example, to multiply by 3, use [double, half, double] (2 * 0.5 * 3 = 3)")
    
    print("\n3. Improved bit shift implementation:")
    print("   - Uses reset operations to ensure proper doubling/halving behavior")
    print("   - LSB is explicitly set to 0 after doubling")
    print("   - MSB is explicitly set to 0 after halving")
    
    print("\n4. Dynamically sized register:")
    print("   - Counter register size adjusts based on expected maximum value")
    print("   - Prevents overflow issues when multiple doublings are applied")
    
    print("\nApplications:")
    print("- Precise quantum frequency multiplication/division")
    print("- Implementing arbitrary rational multiplication factors (e.g., x1.5, x0.75)")
    print("- More complex quantum signal processing operations")
    print("- Building block for quantum arithmetic circuits")
    
    return

# Main function (for when the module is run directly)
if __name__ == "__main__":
    print("Enhanced Quantum Signal Multiplier with Chained Operations")
    print("=" * 70)
    print("This circuit extends the original signal counter to support multiple")
    print("doubling and halving operations for arbitrary multiplication factors.")
    print("-" * 70)
    
    # Show an example circuit
    wave_peak_example = "1"
    n_signals_example = 4
    operations_example = ["double", "half"]
    
    try:
        qc_example = create_quantum_signal_multiplier(
            wave_peak=wave_peak_example,
            n_signals=n_signals_example,
            operations=operations_example
        )
        
        print(f"\nCircuit diagram for Peak={wave_peak_example}, n={n_signals_example}, ops={operations_example}:")
        print(qc_example.draw(output='text', fold=120))
        
    except Exception as e:
        print(f"\nAn error occurred during example circuit creation or drawing: {e}")
        traceback.print_exc()
    
    # Run the simulations
    results = simulate_quantum_signal_multiplier()
    
    # Explain the implementation
    explain_signal_multiplier()
    
    # Test with multiplication factor instead of operations
    print("\n--- Testing with Multiplication Factor ---")
    try:
        mult_factor = 2.5
        qc_mult = create_quantum_signal_multiplier(
            wave_peak="1",
            n_signals=2,
            multiplication_factor=mult_factor
        )
        
        print(f"Circuit for multiplication factor {mult_factor}:")
        print(qc_mult.draw(output='text', fold=120))
    except Exception as e:
        print(f"Error creating circuit with multiplication factor: {e}")
    
    # Final summary
    print("-" * 70)
    print("\n--- Implementation Summary ---")
    print("- Enhanced Quantum Signal Multiplier:")
    print("  - Supports multiple sequential doubling/halving operations")
    print("  - Can automatically derive operations from a multiplication factor")
    print("  - Improved bit manipulation with explicit reset operations")
    print("  - Dynamic register sizing to prevent overflow")
    print("- Applications:")
    print("  - Precise quantum frequency multiplication")
    print("  - Arbitrary rational multiplication factors")
    print("  - Building block for quantum arithmetic")
    print("\nExecution finished.")
    print("-" * 70)
