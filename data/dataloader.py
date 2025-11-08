import re
import zipfile

from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2

from typing import List, Dict, Any

backends = [
    {"num_qubits": 127, "backend": GenericBackendV2(num_qubits=127)},
    {"num_qubits": 133, "backend": GenericBackendV2(num_qubits=133)},
    {"num_qubits": 156, "backend": GenericBackendV2(num_qubits=156)},
]

pattern = r"(.+?)_indep_qiskit_"


def extract_circuit_feats(qasm_str: str, backends: List[Dict[str, Any]]) -> List[int]:
    feats_data = []
    original_circuit = QuantumCircuit.from_qasm_str(qasm_str)

    original_num_qubits = original_circuit.num_qubits
    original_circuit_layers = original_circuit.depth()
    original_gate_counts = sum(
        [gate_count for _, gate_count in original_circuit.count_ops().items()]
    )

    feats_data += [original_num_qubits, original_circuit_layers, original_gate_counts]

    for backend in backends:
        num_qubits, circuit_layers, gate_counts = -1, -1, -1

        if backend["num_qubits"] >= original_num_qubits:
            transplied_circuit = transpile(
                original_circuit, backend["backend"], optimization_level=3
            )

            num_qubits = transplied_circuit.num_qubits
            circuit_layers = transplied_circuit.depth()
            gate_counts = sum(
                [gate_count for _, gate_count in transplied_circuit.count_ops().items()]
            )

        feats_data += [num_qubits, circuit_layers, gate_counts]

    return feats_data


def qtasks_analysis(zip_file_path: str) -> None:
    with open("qtasks_analysis.csv", "w") as file:
        file.write("Algorithm,#Instances\n")

        alg_counts = {}
        with zipfile.ZipFile(zip_file_path, "r") as zip_file:
            for file_name in zip_file.namelist():
                # skipping qwalk with a lot of layers
                if "qwalk" in file_name:
                    continue

                # skipping circuits with more than 50 qubits
                if int(file_name.split("_")[-1][:-5]) > 50:
                    continue

                alg_name = re.search(pattern, file_name).group(1)  # pyright: ignore

                if alg_name not in alg_counts:
                    alg_counts[alg_name] = 0
                alg_counts[alg_name] += 1

        for alg, inst in alg_counts.items():
            file.write(f"{alg},{inst}\n")


def zipped_qasm_to_csv(
    zip_file_path: str,
    backends: List[Dict[str, Any]],
    csv_file: str = "qtasks_indep.csv",
) -> None:
    print("Extracting QTasks...")

    with open(csv_file, "w") as file:
        header = "Algorithm,original_num_qubits,original_circuit_layers,original_gate_counts,"
        for backend in backends:
            model = f"ibm{backend['num_qubits']}"
            header += f"{model}_num_qubits,{model}_circuit_layers,{model}_gate_counts,"

        file.write(f"{header[:-1]}\n")

        file_count = 0
        with zipfile.ZipFile(zip_file_path, "r") as zip_file:
            for file_name in zip_file.namelist():
                # skipping qwalk with a lot of layers
                if "qwalk" in file_name:
                    continue

                # skipping circuits with more than 50 qubits
                if int(file_name.split("_")[-1][:-5]) > 50:
                    continue

                algorithm = re.search(pattern, file_name).group(1)  # pyright: ignore
                with zip_file.open(file_name) as qasm_file:
                    qasm_str = qasm_file.read().decode()
                    feats = extract_circuit_feats(qasm_str, backends)

                row_data = ",".join([str(feat) for feat in feats])
                row = algorithm + "," + row_data

                file.write(f"{row}\n")
                file_count += 1

    print(f"Extracted QTasks from {file_count} files.")


if __name__ == "__main__":
    zip_file_path = "qtasks_indep.zip"
    qtasks_analysis(zip_file_path)

    # Executing this will take some time (actually a lot, kindly use the csv only)
    zipped_qasm_to_csv(zip_file_path, backends)
