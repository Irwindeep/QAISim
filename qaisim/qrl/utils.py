## helper functions from v1.6.1 to get device noise properties

from cirq_google.devices import grid_device, google_noise_properties
from cirq_google.engine.virtual_engine_factory import (
    create_device_from_processor_id,
    load_median_device_calibration,
    load_sample_device_zphase,
    ZPHASE_DATA,
)
from cirq_google.engine import calibration_to_noise_properties
from cirq_google.ops import fsim_gate_family

from cirq.ops.raw_types import Gate
from cirq.ops.identity import IdentityGate
from cirq.ops.wait_gate import WaitGate


def extract_gate_times_ns_from_device(
    device: grid_device.GridDevice,
) -> dict[type[Gate], float]:
    """Extract a dictionary of gate durations in nanoseconds from GridDevice object.

    The durations are obtained from `GridDevice.metadata` field which is
    provided for devices obtained with `create_device_from_processor_id`.

    Args:
        device: Object representing Google devices with a grid qubit layout.

    Returns:
        A dictionary of gate durations versus supported gate types.  Returns an
        empty dictionary when `device.metadata` do not provide gate durations.
    """
    gate_times_ns: dict[type[Gate], float] = {}
    if not device.metadata.gate_durations:
        return gate_times_ns
    gate_type: type[Gate]  # pragma: no cover
    for gate_family, duration in device.metadata.gate_durations.items():
        if isinstance(gate_family, fsim_gate_family.FSimGateFamily):
            for g in gate_family.gates_to_accept:
                gate_type = g if isinstance(g, type) else type(g)
                gate_times_ns[gate_type] = duration.total_nanos()
            continue
        # ordinary GateFamily here
        gate_type = (
            gate_family.gate
            if isinstance(gate_family.gate, type)
            else type(gate_family.gate)
        )
        gate_times_ns[gate_type] = duration.total_nanos()
    # cirq.IdentityGate can leak from FSimGateFamily.  Exclude to default to zero duration.
    _ = gate_times_ns.pop(IdentityGate, None)
    # cirq.WaitGate has variable duration and should not be included here.
    _ = gate_times_ns.pop(WaitGate, None)
    return gate_times_ns


def load_device_noise_properties(
    processor_id: str,
) -> google_noise_properties.GoogleNoiseProperties:
    """Loads NoiseProperties for the given device.

    This combines calibration data for the device with gate times from its specification, and
    the Z phases data, if available, to construct NoiseProperties for device simulation.

    Args:
        processor_id: name of the processor to simulate.

    Raises:
        ValueError: if processor_id is not a supported QCS processor.
    """
    device = create_device_from_processor_id(processor_id)
    calibration = load_median_device_calibration(processor_id)
    zphase_data = (
        load_sample_device_zphase(processor_id) if processor_id in ZPHASE_DATA else None
    )
    if processor_id in ("rainbow", "weber"):
        gate_times_ns = calibration_to_noise_properties.DEFAULT_GATE_NS
    else:
        gate_times_ns = extract_gate_times_ns_from_device(device)
    return calibration_to_noise_properties.noise_properties_from_calibration(
        calibration=calibration, gate_times_ns=gate_times_ns, zphase_data=zphase_data
    )
