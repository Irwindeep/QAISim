from typing import List
from qpysim.qrl.parametrized_qc import ParametrizedQC
import cirq
import tensorflow as tf # type: ignore[import-untyped]
import tensorflow_quantum as tfq # type: ignore[import-untyped]
import numpy as np

class ReUploading(tf.keras.layers.Layer):
    def __init__(
        self,
        parametrized_qc: ParametrizedQC,
        observables: List[cirq.PauliString],
        activation: str = "linear",
        name: str = "re-uploading_pqc"
    ) -> None:
        super(ReUploading, self).__init__(name=name)

        self.num_qubits = parametrized_qc.num_qubits
        self.num_layers = parametrized_qc.num_layers

        phi_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        phi_init = phi_init(shape=(1, len(parametrized_qc.phi)))
        self.phi = tf.Variable(initial_value=phi_init, trainable=True, name="phi")

        lmbd_init = tf.ones(shape=(len(parametrized_qc.inputs), ))
        self.lmbd = tf.Variable(initial_value=lmbd_init, trainable=True, name="lambda")

        symbols = [str(symb) for symb in parametrized_qc.phi + parametrized_qc.inputs]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])

        self.activation = activation
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(
            parametrized_qc.quantum_circuit, observables
        )
    
    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        batch_dim = tf.gather(tf.shape(inputs[0]), 0)

        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_phis = tf.tile(self.phi, multiples=[batch_dim, 1])
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, 2*self.num_layers])

        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)
        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)

        joined_vars = tf.concat([tiled_up_phis, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        return self.computation_layer([tiled_up_circuits, joined_vars])

class Alternating(tf.keras.layers.Layer):
    def __init__(self, output_dim: int) -> None:
        super(Alternating, self).__init__()

        self.output_dim = output_dim
        self.w = tf.Variable(
            initial_value=tf.constant([[(-1.)**i for i in range(output_dim)]]),
            trainable=True, name="observable-weights"
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.matmul(inputs, self.w)
