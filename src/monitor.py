"""
System monitor: GPU (via powermetrics), CPU, and RAM stats.

powermetrics requires root. Run the server with `sudo python main.py serve`
to get GPU stats. CPU% and RAM are available without sudo via psutil.
"""
import plistlib
import subprocess
import threading
import time

import psutil

_lock = threading.Lock()
_stats = {
    "gpu_util": None,   # 0-100 float
    "gpu_w": None,      # float, watts
    "cpu_util": None,   # 0-100 float
    "cpu_w": None,      # float, watts (powermetrics only)
    "ram_used_gb": None,
    "ram_total_gb": None,
}
_proc = None


def get_stats():
    with _lock:
        return dict(_stats)


def _update_psutil():
    vm = psutil.virtual_memory()
    with _lock:
        _stats["ram_used_gb"] = round(vm.used / 1e9, 1)
        _stats["ram_total_gb"] = round(vm.total / 1e9, 1)
        _stats["cpu_util"] = round(psutil.cpu_percent(), 1)


def _parse_plist(chunk: bytes):
    try:
        data = plistlib.loads(chunk.lstrip(b"\x00"))
    except Exception:
        return

    with _lock:
        # GPU utilization: idle_ratio is fraction of time idle
        gpu = data.get("gpu") or {}
        idle = gpu.get("idle_ratio")
        if idle is not None:
            _stats["gpu_util"] = round((1.0 - float(idle)) * 100, 1)

        # Power figures are in mW under "processor"
        proc = data.get("processor") or {}
        gpu_mw = proc.get("gpu_power")
        cpu_mw = proc.get("cpu_power")
        if gpu_mw is not None:
            _stats["gpu_w"] = round(float(gpu_mw) / 1000, 2)
        if cpu_mw is not None:
            _stats["cpu_w"] = round(float(cpu_mw) / 1000, 2)


def _powermetrics_thread(proc):
    buf = b""
    for line in proc.stdout:
        buf += line
        if b"</plist>" in line:
            _parse_plist(buf)
            _update_psutil()
            buf = b""


def _psutil_only_thread():
    while _proc is None and _running:
        _update_psutil()
        time.sleep(1)


_running = False


def start():
    global _proc, _running
    _running = True
    _update_psutil()   # immediate first read

    import os
    prefix = [] if os.geteuid() == 0 else ["sudo", "-n"]
    cmd = prefix + [
        "powermetrics",
        "--samplers", "gpu_power,cpu_power",
        "-f", "plist",
        "-i", "1000",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        time.sleep(0.3)
        if proc.poll() is not None:
            raise RuntimeError("exited immediately")
        _proc = proc
        t = threading.Thread(target=_powermetrics_thread, args=(proc,), daemon=True)
        t.start()
        print("GPU monitor: powermetrics active (GPU+CPU+RAM)")
    except Exception as e:
        print(f"GPU monitor: powermetrics unavailable ({e}) — run with sudo for GPU stats; showing CPU+RAM only")
        t = threading.Thread(target=_psutil_poll, daemon=True)
        t.start()


def _psutil_poll():
    while _running:
        _update_psutil()
        time.sleep(1)


def stop():
    global _running, _proc
    _running = False
    if _proc:
        _proc.terminate()
        _proc = None
