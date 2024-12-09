from typing import Dict
import simpy
import gym
import random
from gym import spaces
from qpysim.qnode import QNode, QNodeParams
from qpysim.broker import Broker
from qpysim.utils import Dataset
from qpysim.qrl.env_qnodes import ibm_qnodes

class QRLEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        qtasks_dataset: Dataset,
        num_qnodes: int = 11,
        qnode_capacity: int = 1
    ) -> None:
        super(QRLEnv, self).__init__()

        self.env = simpy.Environment()

        self.qtasks_dataset = qtasks_dataset
        self.num_qnodes = num_qnodes
        self.qnode_capacity = qnode_capacity

        self.broker = Broker(qnode_params=[param for _, param in ibm_qnodes.items()])
