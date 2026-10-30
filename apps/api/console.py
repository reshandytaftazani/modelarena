"""Restore Ctrl+C handling in dedicated Windows console entrypoints."""

import signal
import sys


def enable_console_interrupt() -> None:
    if sys.platform == "win32":
        import ctypes
        if not ctypes.windll.kernel32.SetConsoleCtrlHandler(None, False):
            raise RuntimeError("Cannot enable console interruption")
        signal.signal(signal.SIGINT, signal.default_int_handler)
