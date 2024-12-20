import unittest, os
from qpysim.qrl.qrl_env import QRLEnv
from qpysim.utils import Dataset

class TestQRLEnv(unittest.TestCase):
    def test_env_initialisation(self):
        data_file = data_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../testing_dataset.csv"
        )

        dataset = Dataset(data_file)
        qrl_env = QRLEnv(dataset)

        assert qrl_env.sim_env.now == 0.0
        assert qrl_env.current_qtask is None

    def test_env_reset(self):
        data_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../testing_dataset.csv"
        )

        dataset = Dataset(data_file)
        qrl_env = QRLEnv(dataset)

        state, _ = qrl_env.reset()

        assert len(state) == 5
    
    def test_env_step(self):
        data_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../testing_dataset.csv"
        )

        dataset = Dataset(data_file)
        qrl_env = QRLEnv(dataset)

        state, _ = qrl_env.reset()
        new_state, reward, terminated, _, qtask = qrl_env.step(0)

        assert len(state) == 5
        assert len(new_state) == 5
        assert terminated == False
        assert isinstance(reward, float)

        assert qtask["scheduled_qtask"].num_rescheduled == 0
        assert qtask["scheduled_qtask"].exec_start_time > -1
        assert qtask["scheduled_qtask"].exec_end_time > qtask["scheduled_qtask"].exec_start_time

    def test_termination(self):
        data_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../testing_dataset.csv"
        )

        dataset = Dataset(data_file)
        qrl_env = QRLEnv(dataset)

        state, _ = qrl_env.reset()
        for i in range(30):
            new_state, reward, terminated, _, _ = qrl_env.step(i%qrl_env.num_qnodes)
            if i == 15:
                assert any(new_state["qnode_queued_tasks"] > 0)

        assert terminated == True
        assert reward > 0
