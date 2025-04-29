from qaisim.qnode import QNodeParams

# IBM QPUs as of 8th dec 2024
ibm_qnodes = {
    "ibm_marrakesh": QNodeParams(
        id=0, num_qubits=156, eplg=3.71e-3,
        clops=180000, q_vol=128
    ),
    "ibm_torino": QNodeParams(
        id=1, num_qubits=133, eplg=8.95e-3,
        clops=200000, q_vol=128
    ),
    "ibm_quebec": QNodeParams(
        id=2, num_qubits=127, eplg=1.67e-2,
        clops=32000, q_vol=128
    ),
    "ibm_brisbane": QNodeParams(
        id=3, num_qubits=127, eplg=1.82e-2,
        clops=170000, q_vol=128
    ),
    "ibm_kolkata": QNodeParams(
        id=4, num_qubits=27, eplg=1.5e-2,
        clops=66000, q_vol=128
    )
}
