import sympy
import numpy as np

from cirq.circuits.circuit import Circuit
from cirq.devices.grid_qubit import GridQubit

from cirq.ops.common_gates import ry, rz, CZ
from cirq.ops.common_channels import amplitude_damp, depolarize
from cirq_google.devices.google_noise_properties import GoogleNoiseProperties

from typing import List
from collections import Counter
from qaisim.qrl.utils import load_device_noise_properties


def get_noisy_circuit(
    circuit: Circuit,
    noise_props: GoogleNoiseProperties,
    gate_time: float = 25,
    p_depol: float = 0.001,
):
    t1_values = list(noise_props.t1_ns.values())
    avg_t1 = float(np.mean(t1_values)) if t1_values else 20000.0

    counts = Counter()
    for moment in circuit:
        for op in moment:
            for q in op.qubits:
                counts[q] += 1

    noise_ops = []
    for q, n_ops in counts.items():
        total_time = n_ops * gate_time
        p_reset_total = 1 - np.exp(-total_time / avg_t1)
        p_depol_total = 1 - (1 - p_depol) ** n_ops

        noise_ops.append(amplitude_damp(p_reset_total).on(q))
        noise_ops.append(depolarize(p_depol_total).on(q))

    return Circuit(list(circuit.all_operations()) + noise_ops)


class ParametrizedQC:
    def __init__(
        self, num_qubits: int, num_layers: int, processor_id: str | None = None
    ) -> None:
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        self.qubits = GridQubit.rect(1, self.num_qubits)
        self.quantum_circuit = Circuit()

        self._create_circuit()

        self.noisy = False
        if processor_id is not None:
            try:
                noise_props = load_device_noise_properties(processor_id)
                device_qubits = list(noise_props.t1_ns.keys())
                logical_qubits = sorted(self.quantum_circuit.all_qubits())

                mapping = {lq: dq for lq, dq in zip(logical_qubits, device_qubits)}
                new_ops = [
                    op.with_qubits(*[mapping[q] for q in op.qubits])
                    for op in self.quantum_circuit.all_operations()
                ]

                qc = Circuit(new_ops)
                self.quantum_circuit = get_noisy_circuit(qc, noise_props)
                self.qubits = list(self.quantum_circuit.all_qubits())

                self.noisy = True
            except Exception as e:
                print(f"Error loading Noise Model: {e}")
                print("Continuing without noise")

    def _create_circuit(self) -> None:
        self.phi = sympy.symbols(
            f"phi[(0:{2 * (self.num_layers + 1) * self.num_qubits})]"
        )
        self.inputs = sympy.symbols(
            f"x[(0:{self.num_layers})][0:{2 * self.num_qubits}]"
        )

        for i in range(self.num_layers):
            start, end = i * (2 * self.num_qubits), (i + 1) * (2 * self.num_qubits)

            self._u_var(self.phi[start:end])
            self._entangling_layer()
            self._u_enc(self.inputs[start:end])

        self._u_var(self.phi[-2 * self.num_qubits :])
        self._entangling_layer()

    def _u_var(self, phi: List[sympy.Symbol]) -> None:
        rz_gates = [rz(phi[i])(self.qubits[i]) for i in range(self.num_qubits)]
        ry_gates = [
            ry(phi[self.num_qubits + i])(self.qubits[i]) for i in range(self.num_qubits)
        ]

        self.quantum_circuit += Circuit(rz_gates)
        self.quantum_circuit += Circuit(ry_gates)

    def _entangling_layer(self) -> None:
        cz_gates = [CZ(q0, q1) for q0, q1 in zip(self.qubits, self.qubits[1:])]
        if self.num_qubits > 2:
            cz_gates.append(CZ(self.qubits[0], self.qubits[-1]))

        self.quantum_circuit += Circuit(cz_gates)

    def _u_enc(self, inputs: List[sympy.Symbol]) -> None:
        ry_gates = [ry(inputs[i])(self.qubits[i]) for i in range(self.num_qubits)]
        rz_gates = [
            rz(inputs[self.num_qubits + i])(self.qubits[i])
            for i in range(self.num_qubits)
        ]

        self.quantum_circuit += Circuit(ry_gates)
        self.quantum_circuit += Circuit(rz_gates)

    def __repr__(self) -> str:
        return f"ParametrizedQC(num_qubits={self.num_qubits}, num_layers={self.num_layers})"
