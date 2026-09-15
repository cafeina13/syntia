"""Windows-specific tweaks the voice assistant needs."""

import os


def disable_power_throttling() -> bool:
    # Windows 11 puts background processes into "efficiency mode" (EcoQoS): slow
    # efficiency cores, lower speed. A bot whose console isn't the focused window
    # is exactly that — spike 4 measured Whisper running 2.5x slower because of
    # it. Opt this process out. Returns True if Windows accepted.
    if os.name != "nt":
        return False
    import ctypes
    from ctypes import wintypes

    class ThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetProcessInformation.restype = wintypes.BOOL
    PROCESS_POWER_THROTTLING = 4  # ProcessPowerThrottling
    EXECUTION_SPEED = 0x1  # control execution-speed throttling...
    state = ThrottlingState(1, EXECUTION_SPEED, 0)  # ...and turn it OFF
    return bool(kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), PROCESS_POWER_THROTTLING,
                                               ctypes.byref(state), ctypes.sizeof(state)))
