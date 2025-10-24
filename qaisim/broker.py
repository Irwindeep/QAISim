import simpy
from qaisim.qtask import QTask
from qaisim.qnode import QNode, QNodeParams

from typing import List, Generator, Union


class Broker:
    def __init__(
        self,
        env: simpy.Environment,
        qnode_params: List[QNodeParams],
        qtasks: List[QTask] = [],
    ) -> None:
        self.env = env

        self.qnodes = [QNode(self.env, *params) for params in qnode_params]
        self.qtasks = qtasks

    def add_qtasks(self, new_qtasks: List[QTask]) -> None:
        self.qtasks += new_qtasks

    def assign_qtasks(self) -> None:
        for qtask in self.qtasks:
            qnode = self.assign(qtask)
            qnode.add_qtask(qtask)
            self.env.process(self.task_executor(qnode, qtask))

    def assign(self, qtask: QTask) -> QNode:
        raise NotImplementedError(
            f"Function to map QTask to QNode is not implemented. Input {qtask} not assigned."
        )

    def task_executor(self, qnode: QNode, qtask: QTask) -> Generator:
        yield self.env.timeout(max(0, qtask.arrival_time - self.env.now))
        with qnode.capacity.request() as request:
            yield request
            yield self.env.process(qnode.process_qtask(qtask))

    def run(self, until: Union[int, None] = None) -> None:
        self.assign_qtasks()
        self.env.run(until=until)
