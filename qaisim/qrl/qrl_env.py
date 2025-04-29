from typing import (
    Optional,
    Dict,
    Any,
    Tuple,
    List,
    Union
)
import gym, simpy
import numpy as np
from numpy.typing import NDArray
from qaisim import QTask, QNode
from qaisim.utils import Dataset
from qaisim.broker import Broker
from .env_qnodes import ibm_qnodes
from gym import spaces

MAX_CIRCUIT_LAYERS=200000
MAX_GATE_COUNTS=250000
MAX_QUEUED_TASKS=30
MAX_RESCHEDULING=10

Observation = Dict[str, NDArray]

class QRLEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        dataset: Union[str, Dataset],
    ) -> None:
        super().__init__()

        if isinstance(dataset, str):
            try:
                self.dataset = Dataset(dataset)
            except Exception as e:
                raise RuntimeError(f"Error in creating Dataset Object: {e}")
        else:
            self.dataset = dataset

        self.sim_env = simpy.Environment()
        self.broker = Broker(
            env=self.sim_env,
            qnode_params=[param for _, param in ibm_qnodes.items()]
        )

        self.qnodes = self.broker.qnodes
        self.num_qnodes = len(self.qnodes)

        self.action_space = spaces.Discrete(self.num_qnodes)
        self.observation_space = spaces.Dict(
            {
                "qnode_queued_tasks": spaces.Box(low=0, high=MAX_QUEUED_TASKS, shape=(self.num_qnodes, )),
                "qtask_arrival_time": spaces.Box(low=0, high=60),
                "qtask_num_qubits": spaces.Box(low=2, high=156),
                "qtask_circuit_layers": spaces.Box(low=0, high=MAX_CIRCUIT_LAYERS),
                "qtask_gate_counts": spaces.Box(low=0, high=MAX_GATE_COUNTS)
            }
        )

        self.qtasks: List[QTask] = []
        self.ibm_qtasks: List[Dict[int, QTask]] = []

        self.current_qtask: Optional[QTask] = None
        self.current_ibm_qtask: Optional[Dict[int, QTask]] = None

    def reset(self, *, seed = None, return_info = False, options = None) -> Any:
        super().reset(seed=seed, return_info=return_info, options=options)

        self.sim_env = simpy.Environment()
        self.broker = Broker(
            env=self.sim_env,
            qnode_params=[param for _, param in ibm_qnodes.items()]
        )

        self.qnodes = self.broker.qnodes
        self._generate_qtasks()

        return self._get_obs(), self._get_info()
    
    def step(self, action: int) -> Tuple[Any, float, bool, bool, dict]:
        if self.current_qtask is None or self.current_ibm_qtask is None:
            raise RuntimeError("Called `step` on NoneType QTask")
        
        ibm_qtask = self.current_qtask
        if self.qnodes[action].num_qubits in [127, 133, 156]:
            ibm_qtask = self.current_ibm_qtask[self.qnodes[action].num_qubits]

        reward = self._assign_qtask_to_qnode(self.current_qtask, self.qnodes[action], ibm_qtask)

        if len(self.qtasks) > 0:
            self.current_qtask = self.qtasks.pop(0)
            self.current_ibm_qtask = self.ibm_qtasks.pop(0)
            terminated = False
        else:
            self.current_qtask = None
            self.current_ibm_qtask = None
            terminated = True

        return self._get_obs(), reward, terminated, False, {"scheduled_qtask": ibm_qtask}

    def _assign_qtask_to_qnode(self, qtask: QTask, qnode: QNode, ibm_qtask: QTask) -> float:
        if self.current_qtask is None or self.current_ibm_qtask is None:
            raise RuntimeError("Cannot assign empty task")
        
        if qtask.num_qubits > qnode.num_qubits:
            qtask.num_rescheduled += 1

            if qtask.num_rescheduled <= MAX_RESCHEDULING:
                self.qtasks.append(self.current_qtask)
                self.ibm_qtasks.append(self.current_ibm_qtask)
            
            return -10.0
        
        qnode.add_qtask(ibm_qtask)
        qtask_execution = self.broker.task_executor(qnode, ibm_qtask)
        self.sim_env.process(qtask_execution)
        self.sim_env.run()

        waiting_time = ibm_qtask.exec_start_time - ibm_qtask.arrival_time
        exec_time = ibm_qtask.exec_end_time - ibm_qtask.exec_start_time

        assert waiting_time + exec_time > 0

        return 1/(waiting_time + exec_time)
    
    def _generate_qtasks(self) -> None:
        qtask_ids = list(range(MAX_QUEUED_TASKS))
        arrival_times = np.sort(
            np.random.uniform(
                low=self.sim_env.now,
                high=self.sim_env.now+60,
                size=MAX_QUEUED_TASKS
            )
        )

        qtasks_dict = self.dataset.random_qtasks(MAX_QUEUED_TASKS).values()
        self.qtasks = [
            QTask(i+1, *list(qtasks_dict)[i]["original"])
            for i in qtask_ids
        ]

        assert all([task.circuit_layers > 0 for task in self.qtasks])

        self.ibm_qtasks = [
            {
                127: QTask(i+1, *list(qtasks_dict)[i]["ibm127"]),
                133: QTask(i+1, *list(qtasks_dict)[i]["ibm133"]),
                156: QTask(i+1, *list(qtasks_dict)[i]["ibm156"])
            }
            for i in qtask_ids
        ]

        for i, qtask in enumerate(self.qtasks):
            qtask.arrival_time = arrival_times[i]
            self.ibm_qtasks[i][127].arrival_time = arrival_times[i]
            self.ibm_qtasks[i][133].arrival_time = arrival_times[i]
            self.ibm_qtasks[i][156].arrival_time = arrival_times[i]

        self.current_qtask = self.qtasks.pop(0)
        self.current_ibm_qtask = self.ibm_qtasks.pop(0)

    def _get_obs(self) -> Observation:
        if self.current_qtask is None:
            return {}

        qnode_queued_tasks = []
        for qnode in self.qnodes:
            count = 0
            for _, exec_end in qnode.busy_time:
                if self.current_qtask.arrival_time < exec_end: count += 1
            
            qnode_queued_tasks.append(count)

        obs = {
            "qnode_queued_tasks": np.array(qnode_queued_tasks),
            "qtask_arrival_time": np.array([self.current_qtask.arrival_time]),
            "qtask_num_qubits": np.array([self.current_qtask.num_qubits]),
            "qtask_circuit_layers": np.array([self.current_qtask.circuit_layers]),
            "qtask_gate_counts": np.array([self.current_qtask.gate_counts])
        }

        return obs

    def _get_info(self) -> Any:
        if self.sim_env.now == 0: return {}

        return {"scheduled_qtask": self.current_qtask}
