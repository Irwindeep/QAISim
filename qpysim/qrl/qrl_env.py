from typing import (
    Optional,
    Dict,
    Any,
    Tuple,
    List
)
import simpy, gym
import numpy as np
from gym import spaces
from qpysim import QTask, QNode
from qpysim.broker import Broker
from qpysim.utils import Dataset
from qpysim.qrl.env_qnodes import ibm_qnodes

MAX_TIME=10000000
MAX_CIRCUIT_LAYERS=2000000
MAX_GATE_COUNTS=250000

class QRLEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        qtasks_dataset: Dataset,
        num_qnodes: int = 11,
        num_qtasks: int = 30,
        qnode_capacity: int = 1
    ) -> None:
        super(QRLEnv, self).__init__()

        self.sim_env = simpy.Environment()

        self.qtasks_dataset = qtasks_dataset
        self.num_qnodes = num_qnodes
        self.num_qtasks = num_qtasks
        self.qnode_capacity = qnode_capacity

        self.broker = Broker(qnode_params=[param for _, param in ibm_qnodes.items()])
        self.broker.env = self.sim_env

        self.qtasks = self.broker.qtasks
        self.target_specific_qtasks: List[Dict[int, QTask]] = []
        self.qnodes = self.broker.qnodes

        self.action_space = spaces.Discrete(self.num_qnodes)
        self.observation_space = spaces.Dict({
            "qnode_num_qubits": spaces.Box(low=0, high=156, shape=(self.num_qnodes,)),
            "qnode_eplg": spaces.Box(low=0.0, high=1.0, shape=(self.num_qnodes,)),
            "qnode_clops": spaces.Box(low=20000, high=250000, shape=(self.num_qnodes,)),
            "qtask_arrival_time": spaces.Box(low=0, high=MAX_TIME, shape=(1,)),
            "qtask_num_qubits": spaces.Box(low=0, high=156, shape=(1,)),
            "qtask_circuit_layers": spaces.Box(low=0, high=MAX_CIRCUIT_LAYERS, shape=(1,)),
            "qtask_gate_counts": spaces.Box(low=0, high=MAX_GATE_COUNTS, shape=(1,))
        })

        self.current_qtask: Optional[QTask] = None
        self.current_target_specific_qtask: Optional[Dict[int, QTask]] = None
        self.current_time = 0.0

    def assign_qtask_to_qnode(self, qtask: QTask, qnode: QNode, target_specific_qtask: QTask) -> float:
        # A condition never attainable at our dataset but for scalability
        if qtask.num_qubits > qnode.num_qubits:
            qtask.num_rescheduled += 1
            return -10.0
        
        qnode.add_qtask(target_specific_qtask)
        qtask_execution = self.broker.task_executor(qnode, target_specific_qtask)
        self.sim_env.process(qtask_execution)

        waiting_time = target_specific_qtask.exec_end_time - target_specific_qtask.arrival_time
        exec_time = target_specific_qtask.exec_end_time - target_specific_qtask.exec_start_time

        self.current_time += waiting_time + exec_time
        return (1/waiting_time + exec_time) - qnode.eplg
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self.current_qtask is None or self.current_target_specific_qtask is None:
            raise RuntimeError("Called step on None qtask")
        
        target_specific_qtask = self.current_target_specific_qtask[self.qnodes[action].num_qubits]
    
        reward = self.assign_qtask_to_qnode(self.current_qtask, self.qnodes[action], target_specific_qtask)
        scheduled_qtask = self.current_qtask

        if len(self.qtasks) > 0:
            self.current_qtask = self.qtasks.pop(0)
            self.current_target_specific_qtask = self.target_specific_qtasks.pop(0)
            terminated = False
        else:
            self.current_qtask = None
            self.current_target_specific_qtask = None
            terminated = True

        return self._get_obs(), reward, terminated, False, {"scheduled_qtask": scheduled_qtask}
    
    def reset(self, *, seed=None, return_info: bool = False, options=None) -> Tuple[Dict[str, Any], Dict]:
        super().reset(seed=seed)
        self._generate_qtasks()
        self.current_time = 0.0

        return self._get_obs(), {}

    def _generate_qtasks(self) -> None:
        qtask_ids = list(range(self.num_qtasks))
        arrival_times = np.sort(np.random.uniform(self.current_time, self.current_time+60, size=self.num_qtasks))

        qtasks_dict = self.qtasks_dataset.random_qtasks(self.num_qtasks).values()
        self.qtasks = [
            QTask(i+1, *list(qtasks_dict)[i]["original"])
            for i in qtask_ids
        ]

        self.target_specific_qtasks = [
            {
                127: QTask(i+1, *list(qtasks_dict)[i]["ibm127"]),
                133: QTask(i+1, *list(qtasks_dict)[i]["ibm133"]),
                156: QTask(i+1, *list(qtasks_dict)[i]["ibm156"])
            }
            for i in qtask_ids
        ]

        for i, qtask in enumerate(self.qtasks):
            qtask.arrival_time = arrival_times[i]
            self.target_specific_qtasks[i][127].arrival_time = arrival_times[i]
            self.target_specific_qtasks[i][133].arrival_time = arrival_times[i]
            self.target_specific_qtasks[i][156].arrival_time = arrival_times[i]

        self.current_qtask = self.qtasks.pop(0)
        self.current_target_specific_qtask = self.target_specific_qtasks.pop(0)

    def _get_obs(self) -> Dict[str, Any]:
        if self.current_qtask is None: return {}

        obs = {
            "qnode_num_qubits": np.array([qnode.num_qubits for qnode in self.qnodes]),
            "qnode_eplg": np.array([qnode.eplg for qnode in self.qnodes]),
            "qnode_clops": np.array([qnode.clops for qnode in self.qnodes]),
            "qtask_arrival_time": np.array([self.current_qtask.arrival_time]),
            "qtask_num_qubits": np.array([self.current_qtask.num_qubits]),
            "qtask_circuit_layers": np.array([self.current_qtask.circuit_layers]),
            "qtask_gate_counts": np.array([self.current_qtask.gate_counts])
        }

        return obs
    
    def close(self):
        # Not implementing as of now
        pass
