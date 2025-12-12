# resource_monitor.py   | SET (leave it as it is)

import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

import psutil

try:
    import pynvml
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False


@dataclass
class ResourceMonitor:
    sample_interval_s: float = 5.0
    device_index: int = 0
    enabled: bool = True

    _thread: Optional[threading.Thread] = None
    _stop: Optional[threading.Event] = None
    _rows: Optional[List[Dict[str, Any]]] = None

    def start(self) -> None:
        if not self.enabled:
            self._rows = []
            return
        self._rows = []
        self._stop = threading.Event()

        # init psutil baselines
        psutil.cpu_percent(interval=None)

        # init NVML if possible
        self._nvml_ok = False
        if _HAS_NVML:
            try:
                pynvml.nvmlInit()
                self._h = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
                self._nvml_ok = True
            except Exception:
                self._nvml_ok = False

        self._proc = psutil.Process()
        self._t0 = time.time()
        self._net0 = psutil.net_io_counters()
        self._disk0 = psutil.disk_io_counters()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> pd.DataFrame:
        if not self.enabled:
            return pd.DataFrame(self._rows or [])

        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

        if getattr(self, "_nvml_ok", False):
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

        return pd.DataFrame(self._rows or [])

    def _run(self) -> None:
        while self._stop is not None and not self._stop.is_set():
            self._rows.append(self._sample_once())
            time.sleep(self.sample_interval_s)

    def _sample_once(self) -> Dict[str, Any]:
        t = time.time()
        row: Dict[str, Any] = {
            "t_seconds": t - self._t0,
            "t_hours": (t - self._t0) / 3600.0,
        }

        # ----- system CPU / RAM -----
        vm = psutil.virtual_memory()
        row.update({
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_mb": (vm.total - vm.available) / (1024**2),
            "ram_available_mb": vm.available / (1024**2),
            "ram_percent": vm.percent,
        })

        # ----- process stats -----
        try:
            mem = self._proc.memory_info()
            row.update({
                "proc_rss_mb": mem.rss / (1024**2),
                "proc_vms_mb": mem.vms / (1024**2),
                "proc_threads": self._proc.num_threads(),
                "proc_cpu_percent": self._proc.cpu_percent(interval=None),
            })
        except Exception:
            row.update({
                "proc_rss_mb": np.nan,
                "proc_vms_mb": np.nan,
                "proc_threads": np.nan,
                "proc_cpu_percent": np.nan,
            })

        # ----- disk / net deltas -----
        try:
            net = psutil.net_io_counters()
            disk = psutil.disk_io_counters()
            row.update({
                "net_sent_bytes": net.bytes_sent - self._net0.bytes_sent,
                "net_recv_bytes": net.bytes_recv - self._net0.bytes_recv,
                "disk_read_bytes": disk.read_bytes - self._disk0.read_bytes,
                "disk_write_bytes": disk.write_bytes - self._disk0.write_bytes,
            })
        except Exception:
            row.update({
                "net_sent_bytes": np.nan,
                "net_recv_bytes": np.nan,
                "disk_read_bytes": np.nan,
                "disk_write_bytes": np.nan,
            })

        # ----- GPU (NVML if available) -----
        if getattr(self, "_nvml_ok", False):
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._h)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(self._h)
                temp = pynvml.nvmlDeviceGetTemperature(self._h, pynvml.NVML_TEMPERATURE_GPU)
                pwr_mw = pynvml.nvmlDeviceGetPowerUsage(self._h)  # milliwatts
                pwr_limit_mw = pynvml.nvmlDeviceGetEnforcedPowerLimit(self._h)

                sm_clk = pynvml.nvmlDeviceGetClockInfo(self._h, pynvml.NVML_CLOCK_SM)
                mem_clk = pynvml.nvmlDeviceGetClockInfo(self._h, pynvml.NVML_CLOCK_MEM)

                # ECC errors (may fail on some consumer GPUs)
                try:
                    ecc_corr = pynvml.nvmlDeviceGetTotalEccErrors(
                        self._h, pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                        pynvml.NVML_VOLATILE_ECC
                    )
                    ecc_unc = pynvml.nvmlDeviceGetTotalEccErrors(
                        self._h, pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                        pynvml.NVML_VOLATILE_ECC
                    )
                except Exception:
                    ecc_corr, ecc_unc = np.nan, np.nan

                row.update({
                    "gpu_util_percent": util.gpu,
                    "gpu_mem_util_percent": util.memory,
                    "gpu_mem_used_mb": meminfo.used / (1024**2),
                    "gpu_mem_total_mb": meminfo.total / (1024**2),
                    "gpu_mem_used_percent": 100.0 * meminfo.used / max(1.0, meminfo.total),
                    "gpu_temp_c": temp,
                    "gpu_power_w": pwr_mw / 1000.0,
                    "gpu_power_limit_w": pwr_limit_mw / 1000.0,
                    "gpu_sm_clock_mhz": sm_clk,
                    "gpu_mem_clock_mhz": mem_clk,
                    "gpu_ecc_corrected": ecc_corr,
                    "gpu_ecc_uncorrected": ecc_unc,
                })
            except Exception:
                # If NVML gets cranky mid-run
                row.update({
                    "gpu_util_percent": np.nan,
                    "gpu_mem_util_percent": np.nan,
                    "gpu_mem_used_mb": np.nan,
                    "gpu_mem_total_mb": np.nan,
                    "gpu_mem_used_percent": np.nan,
                    "gpu_temp_c": np.nan,
                    "gpu_power_w": np.nan,
                    "gpu_power_limit_w": np.nan,
                    "gpu_sm_clock_mhz": np.nan,
                    "gpu_mem_clock_mhz": np.nan,
                    "gpu_ecc_corrected": np.nan,
                    "gpu_ecc_uncorrected": np.nan,
                })
        else:
            # no NVML: still log NaNs so CSV columns are consistent
            row.update({
                "gpu_util_percent": np.nan,
                "gpu_mem_util_percent": np.nan,
                "gpu_mem_used_mb": np.nan,
                "gpu_mem_total_mb": np.nan,
                "gpu_mem_used_percent": np.nan,
                "gpu_temp_c": np.nan,
                "gpu_power_w": np.nan,
                "gpu_power_limit_w": np.nan,
                "gpu_sm_clock_mhz": np.nan,
                "gpu_mem_clock_mhz": np.nan,
                "gpu_ecc_corrected": np.nan,
                "gpu_ecc_uncorrected": np.nan,
            })

        return row
