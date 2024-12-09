import unittest, os
from qpysim.qrl.qrl_env import QRLEnv
from qpysim.utils import Dataset

class TestQRLEnv(unittest.TestCase):
    def test_env_initialisation(self):
        data_file = data_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../../data/qtasks_test.csv"
        )

        dataset = Dataset(data_file)
        qrl_env = QRLEnv(dataset)

        assert qrl_env.current_time == 0.0
        assert qrl_env.current_qtask is None

    def test_env_reset(self):
        data_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../../data/qtasks_test.csv"
        )

        dataset = Dataset(data_file)
        qrl_env = QRLEnv(dataset)

        state, _ = qrl_env.reset()

        assert len(state) == 7
    
    def test_env_step(self):
        data_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../../data/qtasks_test.csv"
        )

        dataset = Dataset(data_file)
        qrl_env = QRLEnv(dataset)

        state, _ = qrl_env.reset()
        new_state, reward, terminated, _, _ = qrl_env.step(0)

        assert len(state) == 7
        assert len(new_state) == 7
        assert terminated == False
        assert isinstance(reward, float)
