import unittest, cirq
from qaisim.qrl.parametrized_qc import ParametrizedQC

class TestPQC(unittest.TestCase):
    def test_pqc_circuit(self):
        pqc = ParametrizedQC(num_qubits=4, num_layers=2)

        assert len(pqc.phi) == 24
        assert len(pqc.inputs) == 16

        assert len([op for op in pqc.quantum_circuit.all_operations()]) == 52
        assert len([
            op for op in pqc.quantum_circuit.findall_operations_with_gate_type(cirq.Rz)
        ]) == 20
        assert len([
            op for op in pqc.quantum_circuit.findall_operations_with_gate_type(cirq.Ry)
        ]) == 20
