from typing import List, Generator, NamedTuple, Tuple
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

        self.total_completed_qtasks = 0
        self.total_running_time = 0.0

        self.busy_time: List[Tuple[float, float]] = []

    def add_qtask(self, qtask: QTask) -> None:
        self.qtasks.append(qtask)
        qtask.status == TaskStatus.QUEUED

    def process_qtask(self, qtask: QTask) -> Generator:
        qtask.status = TaskStatus.RUNNING
        qtask.waiting_time = self.env.now - qtask.arrival_time

        exec_time = (qtask.circuit_layers/self.clops)*qtask.shots
        qtask.exec_start_time = self.env.now

        yield self.env.timeout(exec_time)

        qtask.status = TaskStatus.DONE
        qtask.exec_end_time = self.env.now
        
        self.qtasks.remove(qtask)
        self.total_completed_qtasks += 1
        self.total_running_time += exec_time

        self.busy_time.append((qtask.exec_start_time, qtask.exec_end_time))

    def __repr__(self) -> str:
        return f"QNode(id={self.id}, capacity={self.capacity.capacity})"

class QNodeParams(NamedTuple):
    id: int
    num_qubits: int
    eplg: float
    clops: int
    q_vol: int
