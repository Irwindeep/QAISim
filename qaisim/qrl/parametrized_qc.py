import sympy

from cirq.circuits.circuit import Circuit
from cirq.devices.grid_qubit import GridQubit
from cirq.ops.common_gates import ry, rz, CZ

from typing import List


class ParametrizedQC:
    def __init__(self, num_qubits: int, num_layers: int) -> None:
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        self.qubits = GridQubit.rect(1, self.num_qubits)
        self.quantum_circuit = Circuit()

        self._create_circuit()

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
        if self.num_qubits != 2:
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
