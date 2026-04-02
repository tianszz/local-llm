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
        data = plistlib.loads(chunk)
    except Exception:
        return

    with _lock:
        # GPU utilization — key varies by macOS version
        gpu = data.get("gpu") or data.get("GPU") or {}
        active = gpu.get("active_ratio") or gpu.get("Active Ratio")
        if active is not None:
            _stats["gpu_util"] = round(float(active) * 100, 1)

        # Power figures live under "processor"
        proc = data.get("processor") or {}
        gpu_w = proc.get("gpu_w")
        cpu_w = proc.get("cpu_w")
        if gpu_w is not None:
            _stats["gpu_w"] = round(float(gpu_w), 1)
        if cpu_w is not None:
            _stats["cpu_w"] = round(float(cpu_w), 1)


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

    cmd = [
        "sudo", "-n", "powermetrics",
        "--samplers", "gpu_power,cpu_power",
        "-f", "plist",
        "-i", "1000",
        "-n", "0",
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
