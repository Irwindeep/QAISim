from typing import Generator, List
import unittest
import simpy
from qpysim.qtask import QTask
from qpysim.qnode import QNode

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

def exec_qtask(env: simpy.Environment, qnode: QNode, qtask: QTask) -> Generator:
    yield env.timeout(qtask.arrival_time - env.now)
    with qnode.capacity.request() as request:
        yield request
        yield env.process(qnode.process_qtask(qtask))

class TestQNode(unittest.TestCase):
    def test_qnode(self):
        env = simpy.Environment()
        qnode = QNode(
            env=env, id=1, num_qubits=127, eplg=1.5,
            clops=32000, q_vol=128
        )

        for qtask in qtasks:
            qnode.qtasks.append(qtask)

        for qtask in qtasks:
            env.process(exec_qtask(env, qnode, qtask))

        env.run()
        
        self.assertAlmostEqual(qtasks[0].exec_start_time, 0)
        self.assertAlmostEqual(qtasks[0].exec_end_time, 0.096)
        self.assertAlmostEqual(qtasks[2].exec_start_time, 1)
        self.assertAlmostEqual(qtasks[2].exec_end_time, 2.6)
        self.assertAlmostEqual(qtasks[1].exec_start_time, 2.6)
        self.assertAlmostEqual(qtasks[1].exec_end_time, 2.76)
