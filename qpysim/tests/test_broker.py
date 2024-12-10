from typing import List, Tuple
import unittest
import simpy
from qpysim.qtask import QTask
from qpysim.qnode import QNode, QNodeParams
from qpysim.broker import Broker

qtasks: List[QTask] = [
    QTask(
        id=1, num_qubits=5, circuit_layers=3,
        gate_counts=20, arrival_time=0
    ),
    QTask(
        id=2, num_qubits=3, circuit_layers=5,
        gate_counts=15, arrival_time=2
    ),
    QTask(
        id=3, num_qubits=15, circuit_layers=50,
        gate_counts=150, arrival_time=1
    )
]

qnodes: List[QNodeParams] = [
    QNodeParams(
        id=1, num_qubits=127, eplg=1.5,
        clops=32000, q_vol=128
    ),
    QNodeParams(id=2, num_qubits=133, eplg=1.5,
        clops=32000, q_vol=128
    ),
    QNodeParams(
        id=3, num_qubits=156, eplg=1.5,
        clops=32000, q_vol=128
    )
]

class TestBroker(unittest.TestCase):
    def test_broker(self):
        class SimpleBroker(Broker):
            def __init__(self,
                         qnodes: List[QNodeParams], qtasks: List[QTask]):
                super().__init__(qnodes, qtasks)

            def assign(self, qtask: QTask) -> Tuple[int, QNode]:
                return qtask.id - 1, self.qnodes[qtask.id - 1]
        
        broker = SimpleBroker(qnodes, qtasks)
        broker.run()

        assert broker.qnodes[0].total_completed_qtasks == 1
        assert broker.qnodes[1].total_completed_qtasks == 1
        assert broker.qnodes[2].total_completed_qtasks == 1

        self.assertAlmostEqual(broker.qnodes[0].total_running_time, 0.096)
        self.assertAlmostEqual(broker.qnodes[1].total_running_time, 0.16)
        self.assertAlmostEqual(broker.qnodes[2].total_running_time, 1.6)

        self.assertAlmostEqual(qtasks[0].exec_start_time, 0)
        self.assertAlmostEqual(qtasks[0].exec_end_time, 0.096)
        self.assertAlmostEqual(qtasks[1].exec_start_time, 2)
        self.assertAlmostEqual(qtasks[1].exec_end_time, 2.16)
        self.assertAlmostEqual(qtasks[2].exec_start_time, 1)
        self.assertAlmostEqual(qtasks[2].exec_end_time, 2.6)
