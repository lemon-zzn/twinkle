from typing import Any, Dict, Optional, Sequence


def _hostapi_name(hostapis: Sequence[Dict[str, Any]], hostapi_index: Optional[int]) -> Optional[str]:
    if hostapi_index is None:
        return None
    try:
        return str(hostapis[int(hostapi_index)]["name"])
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def _device_metadata(
    device_index: Optional[int],
    devices: Sequence[Dict[str, Any]],
    hostapis: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if device_index is None:
        return None
    device = devices[int(device_index)]
    return {
        "index": int(device_index),
        "name": str(device.get("name", "")),
        "hostapi": _hostapi_name(hostapis, device.get("hostapi")),
        "max_input_channels": int(device.get("max_input_channels", 0)),
        "max_output_channels": int(device.get("max_output_channels", 0)),
    }


def resolve_audio_device_selection(
    requested_input_device: Optional[int],
    requested_output_device: Optional[int],
    default_device,
    devices: Sequence[Dict[str, Any]],
    hostapis: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    default_input = None
    default_output = None
    if isinstance(default_device, (list, tuple)) and len(default_device) >= 2:
        default_input = default_device[0]
        default_output = default_device[1]
    else:
        try:
            default_input = default_device[0]
            default_output = default_device[1]
        except (IndexError, TypeError, KeyError):
            pass

    input_index = requested_input_device if requested_input_device is not None else default_input
    output_index = requested_output_device if requested_output_device is not None else default_output

    return {
        "requested": {
            "input_device": requested_input_device,
            "output_device": requested_output_device,
        },
        "defaults": {
            "input_device": default_input,
            "output_device": default_output,
        },
        "resolved": {
            "input": _device_metadata(input_index, devices, hostapis),
            "output": _device_metadata(output_index, devices, hostapis),
        },
    }


def _format_device(device: Optional[Dict[str, Any]]) -> str:
    if device is None:
        return "unavailable"
    return (
        f"[{device['index']}] {device['name']} "
        f"({device.get('hostapi')}, in={device['max_input_channels']}, out={device['max_output_channels']})"
    )


def format_audio_device_selection(selection: Dict[str, Any]) -> str:
    resolved = selection["resolved"]
    return "\n".join(
        [
            "Audio devices:",
            f"  input : {_format_device(resolved.get('input'))}",
            f"  output: {_format_device(resolved.get('output'))}",
        ]
    )
