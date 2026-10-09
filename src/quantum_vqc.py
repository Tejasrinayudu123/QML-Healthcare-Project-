import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms import VQC
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.algorithms.optimizers import SPSA

from .metrics import evaluate_model

def build_vqc(n_qubits: int):
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2)
    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=2, entanglement="full")

    simulator = AerSimulator(method="statevector")
    qnn = EstimatorQNN(
        feature_map=feature_map,
        ansatz=ansatz,
        estimator=simulator,
    )

    optimizer = SPSA(maxiter=80)
    vqc = VQC(
        neural_network=qnn,
        optimizer=optimizer,
    )

    circuit = feature_map.compose(ansatz)
    return vqc, circuit

def run_vqc(X_train, y_train, X_test, y_test, n_qubits=4):
    vqc, circuit = build_vqc(n_qubits)
    depth = circuit.depth()
    shots = 0

    report = evaluate_model(
        vqc, X_train, y_train, X_test, y_test,
        n_qubits=n_qubits,
        circuit_depth=depth,
        n_shots=shots,
        name=f"VQC ({n_qubits} qubits)",
        notes="Estimator-based VQC with ZZFeatureMap + RealAmplitudes."
    )
    return report
