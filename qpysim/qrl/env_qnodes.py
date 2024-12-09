from qpysim.qnode import QNodeParams

# IBM QPUs as of 8th dec 2024
ibm_qnodes = {
    "ibm_marrakesh": QNodeParams(
        id=1, num_qubits=156, eplg=3.71e-3,
        clops=180000, q_vol=128
    ),
    "ibm_fez": QNodeParams(
        id=2, num_qubits=156, eplg=4.78e-3,
        clops=180000, q_vol=128
    ),
    "ibm_torino": QNodeParams(
        id=3, num_qubits=133, eplg=8.95e-3,
        clops=200000, q_vol=128
    ),
    "ibm_quebec": QNodeParams(
        id=4, num_qubits=127, eplg=1.67e-2,
        clops=32000, q_vol=128
    ),
    "ibm_kyiv": QNodeParams(
        id=5, num_qubits=127, eplg=2.12e-2,
        clops=30000, q_vol=128
    ),
    "ibm_brisbane": QNodeParams(
        id=6, num_qubits=127, eplg=1.82e-2,
        clops=170000, q_vol=128
    ),
    "ibm_sherbrooke": QNodeParams(
        id=7, num_qubits=127, eplg=2.12e-2,
        clops=30000, q_vol=128
    ),
    "ibm_kawasaki": QNodeParams(
        id=8, num_qubits=127, eplg=2.19e-2,
        clops=29000, q_vol=128
    ),
    "ibm_rensselaer": QNodeParams(
        id=9, num_qubits=127, eplg=2.72e-2,
        clops=32000, q_vol=128
    ),
    "ibm_brussels": QNodeParams(
        id=10, num_qubits=127, eplg=2.32e-2,
        clops=37000, q_vol=128
    ),
    "ibm_strasbourg": QNodeParams(
        id=11, num_qubits=127, eplg=3.52e-2,
        clops=205000, q_vol=128
    )
}
