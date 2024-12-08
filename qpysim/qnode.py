from typing import List, Any
from qpysim.utils import TaskStatus
from qpysim.qtask import QTask
import simpy

class QNode:
    def __init__(
        self,
        env: simpy.Environment,
        id: int,
        num_qubits: int,
        eplg: float,
        clops: int,
        q_vol: int,
        capacity: int = 1,
        qtasks: List[QTask] = []
    ) -> None:
        self.env = env
        self.id = id
        self.num_qubits = num_qubits
        self.eplg = eplg
        self.clops = clops
        self.q_vol = q_vol
        
        self.capacity = simpy.Resource(self.env, capacity)
        self.qtasks = qtasks

    def process_qtask(self, qtask: QTask) -> Any:
        qtask.status = TaskStatus.RUNNING
        qtask.waiting_time = self.env.now - qtask.arrival_time

        exec_time = (qtask.circuit_layers/self.clops)*qtask.shots
        qtask.exec_start_time = self.env.now

        yield self.env.timeout(exec_time)

        qtask.status = TaskStatus.DONE
        qtask.exec_end_time = self.env.now
        
        self.qtasks.remove(qtask)

    def __repr__(self) -> str:
        return f"QNode(id={self.id}, capacity={self.capacity.capacity})"
