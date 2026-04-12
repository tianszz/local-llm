"""
nvidia-smi-style GPU/CPU/RAM monitor for Apple Silicon.
Usage: python main.py monitor [--server URL] [--no-server]
"""
import importlib.metadata
import json
import platform
import subprocess
import time
import urllib.request
from datetime import datetime

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src import monitor


def _chip_name():
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return platform.processor() or "Apple Silicon"


def _mlx_version():
    try:
        return importlib.metadata.version("mlx-lm")
    except Exception:
        return "unknown"


def _tps_from_server(url):
    """Background thread: keep _last_tps updated from server SSE."""
    while True:
        try:
            resp = urllib.request.urlopen(url + "/metrics", timeout=5)
            for raw in resp:
                line = raw.decode().strip()
                if line.startswith("data:"):
                    d = json.loads(line[5:].strip())
                    if d.get("tps") is not None:
                        _last_tps["value"] = d["tps"]
        except Exception:
            time.sleep(2)


_last_tps = {"value": None}


def _make_source(server_url, no_server):
    """Always use local powermetrics for GPU/CPU/RAM. Optionally pull tps from server."""
    import threading

    monitor.start()
    server_active = False

    if not no_server:
        try:
            urllib.request.urlopen(server_url + "/info", timeout=1).close()
            t = threading.Thread(target=_tps_from_server, args=(server_url,), daemon=True)
            t.start()
            server_active = True
        except Exception:
            pass

    label = "local powermetrics" + (f"  +  server ({server_url})" if server_active else "")
    if not server_active and not no_server:
        print(f"Server not reachable at {server_url} — tok/s unavailable")

    def _source():
        try:
            while True:
                stats = monitor.get_stats()
                stats["tps"] = _last_tps["value"]
                yield stats
                time.sleep(1)
        finally:
            monitor.stop()

    return _source(), label


def _fmt(val, unit="", fmt=".0f"):
    if val is None:
        return "—"
    return f"{val:{fmt}} {unit}".strip()


def _render(stats, chip, mlx_ver, source_label):
    gpu_u  = stats.get("gpu_util")
    gpu_w  = stats.get("gpu_w")
    cpu_u  = stats.get("cpu_util")
    cpu_w  = stats.get("cpu_w")
    ram_u  = stats.get("ram_used_gb")
    ram_t  = stats.get("ram_total_gb")
    tps    = stats.get("tps")

    ram_pct = f"({ram_u / ram_t * 100:.0f} %)" if ram_u and ram_t else ""

    # ── Header ──────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    total_mem = f"{int(ram_t)} GB unified memory" if ram_t else "? GB unified memory"
    header = Table.grid(expand=True)
    header.add_column(ratio=1)
    header.add_column(justify="right")
    header.add_row(
        Text("local-llm monitor", style="bold white"),
        Text(ts, style="dim"),
    )
    header.add_row(
        Text(f"{chip}  ·  MLX {mlx_ver}  ·  {total_mem}", style="dim"),
        Text(""),
    )

    # ── Left: GPU / CPU ──────────────────────────────────────────────────────
    left = Table(box=None, show_header=False, padding=(0, 1))
    left.add_column(style="bold cyan", width=16)
    left.add_column()

    left.add_row(Text("GPU", style="bold white underline"), "")
    left.add_row("Utilization", _fmt(gpu_u, "%"))
    left.add_row("Power", _fmt(gpu_w, "W", ".1f"))
    left.add_row("", "")
    left.add_row(Text("CPU", style="bold white underline"), "")
    left.add_row("Utilization", _fmt(cpu_u, "%"))
    left.add_row("Power", _fmt(cpu_w, "W", ".1f"))

    # ── Right: Memory / Inference ────────────────────────────────────────────
    right = Table(box=None, show_header=False, padding=(0, 1))
    right.add_column(style="bold cyan", no_wrap=True)
    right.add_column()

    right.add_row(Text("Memory (Unified)", style="bold white underline"), "")
    right.add_row("Used", _fmt(ram_u, "GB", ".1f"))
    right.add_row("Total", f"{_fmt(ram_t, 'GB', '.1f')}  {ram_pct}")
    right.add_row("", "")
    right.add_row(Text("Inference", style="bold white underline"), "")
    right.add_row("tok/s", _fmt(tps, "", ".1f"))
    src_hint = "(via server)" if "server" in source_label else "(local only)"
    right.add_row("", Text(src_hint, style="dim"))

    body = Columns([left, right], expand=True)

    # ── Footer hints ─────────────────────────────────────────────────────────
    hint_parts = [Text(f"Ctrl-C to quit  ·  1 Hz  ·  {source_label}")]
    if gpu_u is None:
        hint_parts.append(Text("GPU stats unavailable — run with sudo for powermetrics", style="dim"))
    if tps is None and "local" in source_label:
        hint_parts.append(Text("tok/s unavailable — start server (python main.py serve) to enable", style="dim"))

    from rich.console import Group
    content = Group(header, Text(""), body, Text(""), *hint_parts)
    return Panel(content, box=box.DOUBLE, border_style="bright_black")


def run(server_url, no_server):
    chip    = _chip_name()
    mlx_ver = _mlx_version()
    source, label = _make_source(server_url, no_server)

    console = Console()
    with Live(console=console, refresh_per_second=1, screen=False) as live:
        try:
            for stats in source:
                live.update(_render(stats, chip, mlx_ver, label))
        except KeyboardInterrupt:
            pass
