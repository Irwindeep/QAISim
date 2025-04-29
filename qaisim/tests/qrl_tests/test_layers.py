import unittest, cirq
from qaisim.qrl.parametrized_qc import ParametrizedQC
from qaisim.qrl.layers import ReUploading
from functools import reduce
import tensorflow as tf # type: ignore[import-untyped]

class TestLayers(unittest.TestCase):
    def test_reuploading_layer(self):
        num_qubits, num_layers = 4, 2
        parametrized_qc = ParametrizedQC(num_qubits=num_qubits, num_layers=num_layers)

        ops = [cirq.Z(q) for q in parametrized_qc.qubits]
        observables = [reduce(lambda x,y: x*y, ops)]

        re_uploading_pqc = ReUploading(
            parametrized_qc=parametrized_qc,
            observables=observables
        )

        inputs = tf.keras.Input(shape=(4, ), dtype=tf.dtypes.float32, name='input')
        outputs = re_uploading_pqc([inputs])

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        test_input = tf.constant([[1, 2, 3, 4]])
        output = model.predict(test_input)

        assert output.shape == (1, 1)
