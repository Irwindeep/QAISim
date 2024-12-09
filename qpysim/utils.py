from typing import Dict, Tuple, NamedTuple
import enum
import pandas as pd
import random

class TaskStatus(enum.Enum):
    INITIALIZING = "QTask is being initialized"
    QUEUED = "QTask is queued"
    VALIDATING = "QTask is being validated"
    RUNNING = "QTask is actively running"
    CANCELLED = "QTask has been cancelled"
    DONE = "QTask has successfully run"
    ERROR = "QTask incurred error"

class QTaskParams(NamedTuple):
    num_qubits: int
    circuit_layers: int
    gate_counts: int

class Dataset:
    def __init__(self, file_name: str) -> None:
        self.data: Dict[Tuple, Dict] = {}
        self.file_name = file_name

    def load_data(self) -> None:
        self.df = pd.read_csv(self.file_name)

        for idx, row in self.df.iterrows():
            row_data = {
                "original": QTaskParams(
                    num_qubits=row["original_num_qubits"], circuit_layers=row["original_circuit_layers"],
                    gate_counts=row["original_gate_counts"]
                ),
                "ibm127": QTaskParams(
                    num_qubits=row["ibm127_num_qubits"], circuit_layers=row["ibm127_circuit_layers"],
                    gate_counts=row["ibm127_gate_counts"]
                ),
                "ibm133": QTaskParams(
                    num_qubits=row["ibm133_num_qubits"], circuit_layers=row["ibm133_circuit_layers"],
                    gate_counts=row["ibm133_gate_counts"]
                ),
                "ibm156": QTaskParams(
                    num_qubits=row["ibm156_num_qubits"], circuit_layers=row["ibm156_circuit_layers"],
                    gate_counts=row["ibm156_gate_counts"]
                )
            }

            key = (row["subset"], row["Algorithm"], row["original_num_qubits"], idx)
            self.data[key] = row_data

    def subset_data(self, subset_id):
        if len(self.data) == 0: self.load_data()
        return {key: value for key, value in self.data.items() if key[0] == subset_id}
    
    def random_qtasks(self, num_qtasks):
        if len(self.data) == 0: self.load_data()
        return {key: value for key, value in random.sample(list(self.data.items()), num_qtasks)}
