import unittest
from qpysim.utils import TaskStatus
from qpysim.qtask import QTask

class TestQTask(unittest.TestCase):
    def test_qtask(self):
        qtask = QTask(
            id=1, num_qubits=5, circuit_layers=3,
            gate_counts=20, arrival_time=0
        )
        
        assert qtask.status == TaskStatus.INITIALIZING
