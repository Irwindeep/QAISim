import unittest, os
from qpysim.utils import Dataset

class TestDataset(unittest.TestCase):
    def test_data_loading(self):
        data_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../data/qtasks_test.csv"
        )

        dataset = Dataset(data_file)
        dataset.load_data()

        assert len(dataset.data) == 22800
        assert len(list(dataset.data.values())[0]) == 4

    def test_qtask_collection(self):
        data_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../data/qtasks_test.csv"
        )

        dataset = Dataset(data_file)
        qtasks = dataset.random_qtasks(num_qtasks=5)
        qtasks = list(qtasks.values())

        assert len(qtasks) == 5

        assert all([isinstance(qtask["original"].num_qubits, int) for qtask in qtasks])
        assert all([isinstance(qtask["ibm127"].num_qubits, int) for qtask in qtasks])
        assert all([isinstance(qtask["ibm133"].num_qubits, int) for qtask in qtasks])
        assert all([isinstance(qtask["ibm156"].num_qubits, int) for qtask in qtasks])

        assert all([isinstance(qtask["original"].circuit_layers, int) for qtask in qtasks])
        assert all([isinstance(qtask["ibm127"].circuit_layers, int) for qtask in qtasks])
        assert all([isinstance(qtask["ibm133"].circuit_layers, int) for qtask in qtasks])
        assert all([isinstance(qtask["ibm156"].circuit_layers, int) for qtask in qtasks])

        assert all([isinstance(qtask["original"].gate_counts, int) for qtask in qtasks])
        assert all([isinstance(qtask["ibm127"].gate_counts, int) for qtask in qtasks])
        assert all([isinstance(qtask["ibm133"].gate_counts, int) for qtask in qtasks])
        assert all([isinstance(qtask["ibm156"].gate_counts, int) for qtask in qtasks])
