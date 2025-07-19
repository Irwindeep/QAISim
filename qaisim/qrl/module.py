# Base class for qrl agents

from typing import Tuple, Optional, List, Callable, Dict, Union, Any
from numpy.typing import NDArray
from qaisim.qrl.parametrized_qc import ParametrizedQC
import cirq
import numpy as np
import tensorflow as tf  # type: ignore[import-untyped]

Observable = List[cirq.PauliString]
EpisodicOutcome = List[Dict[str, Any]]
EpisodeCallable = Callable[
    [tf.keras.Model, int, Union[int, float], Optional[NDArray[np.float32]]],
    EpisodicOutcome,
]


class Module:
    def __init__(
        self,
        parametrized_qc: ParametrizedQC,
        num_actions: int,
        gamma: float = 0.99,
        lrs: Tuple[float, float, float] = (0.01, 0.1, 0.1),
    ) -> None:
        self.parametrized_qc = parametrized_qc
        self.quantum_circuit = parametrized_qc.quantum_circuit
        self.qubits = parametrized_qc.qubits
        self.num_qubits = parametrized_qc.num_qubits
        self.num_layers = parametrized_qc.num_layers

        self.num_actions = num_actions
        self.gamma = gamma

        self.observables: Optional[Observable] = None
        self.model: Optional[tf.keras.Model] = None

        self.w_in, self.w_var, self.w_out = 1, 0, 2

        self.optimizer_in = tf.keras.optimizers.Adam(
            learning_rate=lrs[self.w_in], amsgrad=True
        )
        self.optimizer_var = tf.keras.optimizers.Adam(
            learning_rate=lrs[self.w_var], amsgrad=True
        )
        self.optimizer_out = tf.keras.optimizers.Adam(
            learning_rate=lrs[self.w_out], amsgrad=True
        )

        self.episode_reward_history: List[float] = []
        self.episode_length: List[int] = []

        self.eval_episode_reward_history: List[float] = []
        self.eval_episode_length: List[int] = []
        self.eval_waiting_time: List[float] = []

    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"Module [{type(self).__name__}] is missing `train` function implementation"
        )

    def eval(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"Module [{type(self).__name__}] is missing `train` function implementation"
        )
