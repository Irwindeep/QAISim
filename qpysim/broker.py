from typing import List, Generator, Union, Tuple
import simpy
from qpysim.qtask import QTask
from qpysim.qnode import QNode, QNodeParams

class Broker:
    def __init__(
        self,
        qnode_params: List[QNodeParams],
        qtasks: List[QTask] = []
    ) -> None:
        self.envs = [simpy.Environment() for _ in range(len(qnode_params))]

        self.qnodes = [QNode(self.envs[i], *params) for i, params in enumerate(qnode_params)]

        self.qtasks = qtasks

    def add_qtasks(self, new_qtasks: List[QTask]) -> None:
        self.qtasks += new_qtasks

    def assign_qtasks(self) -> None:
        for qtask in self.qtasks:
            idx, qnode = self.assign(qtask)
            qnode.add_qtask(qtask)
            self.envs[idx].process(self.task_executor(self.envs[idx], qnode, qtask))

    def assign(self, qtask: QTask) -> Tuple[int, QNode]:
        raise NotImplementedError(
            f"Function to map QTask to QNode is not implemented"
        )
    
    def task_executor(self, env: simpy.Environment, qnode: QNode, qtask: QTask) -> Generator:
        yield env.timeout(abs(qtask.arrival_time - env.now))
        with qnode.capacity.request() as request:
            yield request
            yield env.process(qnode.process_qtask(qtask))

    def run(self, until: Union[int, None] = None) -> None:
        self.assign_qtasks()
        for env in self.envs: env.run(until=until)
