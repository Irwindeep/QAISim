from qpysim.utils import TaskStatus

class QTask:
    def __init__(
        self,
        id: int,
        num_qubits: int,
        circuit_layers: int,
        gate_counts: int,
        arrival_time: float,
        shots: int = 1024,
        qasm_file: str = ""
    ) -> None:
        self.id = id
        self.num_qubits = num_qubits
        self.circuit_layers = circuit_layers
        self.gate_counts = gate_counts
        self.arrival_time = arrival_time

        self.shots = shots
        self.qasm_file = qasm_file

        self._status = TaskStatus.INITIALIZING
        self.waiting_time = -1.0

        self.exec_start_time = -1.0
        self.exec_end_time = -1.0

    @property
    def status(self) -> TaskStatus:
        return self._status
    
    @status.setter
    def status(self, new_status: TaskStatus) -> None:
        self._status = new_status

    def __repr__(self) -> str:
        return f"QTask(id={self.id}, status={self.status})"
