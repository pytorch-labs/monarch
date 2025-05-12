# pyre-unsafe
import logging
import time
from typing import Any, Dict, List, Protocol, Tuple

import monarch

import torch
from monarch.opaque_object import OpaqueObject
from torchtitan import utils
from torchtitan.metrics import build_device_memory_monitor

logger = logging.getLogger(__name__)


class MetricsLogger(Protocol):
    def log(self, step: int, metrics: Dict[str, Any]):
        pass

    def observe(self, kind: str, value: torch.Tensor):
        pass


class Report:
    def __init__(
        self,
        log_freq: int,
        num_flop_per_token: int,
        preserved_metrics: Tuple[str, ...] = (
            "loss_metrics/global_avg_loss",
            "loss_metrics/global_max_loss",
        ),
    ):
        self.log_freq = log_freq
        self.num_flop_per_token = num_flop_per_token
        # initialize device memory monitor and get peak flops for MFU calculation
        self.device_memory_monitor = OpaqueObject(build_device_memory_monitor)
        self.gpu_peak_flops = utils.get_peak_flops(torch.cuda.get_device_name())
        logger.info(f"Peak FLOPS used for computing MFU: {self.gpu_peak_flops:.3e}")
        self.losses_since_last_log = []
        self.ntokens_since_last_log = 0
        self.data_loading_times = []
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.call_method_on_shard_and_fetch("reset_peak_stats")
        self.metrics_loggers: List[MetricsLogger] = []
        self.preserved_metrics = preserved_metrics
        self.pending_observes = []
        self.metrics_state: List[Tuple[int, Dict[str, Any]]] = []
        self.observations_since_last_log = []

    def add_metrics_logger(self, metric: MetricsLogger):
        self.metrics_loggers.append(metric)

    def report_metrics(self, step: int, metrics: Dict[str, Any]):
        for logger in self.metrics_loggers:
            logger.log(step, metrics)

        preserved = {k: v for k, v in metrics.items() if k in self.preserved_metrics}
        if preserved:
            self.metrics_state.append((step, preserved))

    def observe(self, name: str, value: torch.Tensor):
        if not self.metrics_loggers:
            return
        value = value.detach().cpu()
        self.pending_observes.append((name, value))

    def step_state(self):
        """
        Anything that the reporter wants to do and save during a step,
        but then bring to the host when doing reporting goes into step_state.
        After the step, we will run step_state()/load_step_state() before
        report_completed_step is run. This is necessary to work with the limitations
        of monarch.compile, where any remote value produced within a compiled function
        has to be a return value of the function.
        """
        return self.pending_observes

    def load_step_state(self, state):
        self.pending_observes = state

    def report_memory_usage(self, topic: str):
        device_mem_stats = self.device_memory_monitor.call_method_on_shard_and_fetch(
            "get_peak_stats"
        ).result()
        logger.info(
            f"CUDA memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

    def state_dict(self):
        return {"metrics_state": self.metrics_state}

    def load_state_dict(self, d: Dict[str, Any]):
        for step, metrics in d["metrics_state"]:
            self.report_metrics(step, metrics)

    def report_completed_step_from_previous_run(self, step: int, avg_loss, max_loss):
        metrics = {
            "loss_metrics/global_avg_loss": avg_loss,
            "loss_metrics/global_max_loss": max_loss,
        }
        self.report_metrics(step, metrics)

    def report_completed_step(
        self, step: int, loss, tokens: int, data_load_time: float
    ):
        self.ntokens_since_last_log += tokens
        self.data_loading_times.append(data_load_time)
        self.losses_since_last_log.append(loss)
        self.observations_since_last_log.extend(self.pending_observes)
        self.pending_observes.clear()

        # log metrics
        if step == 1 or step % self.log_freq == 0:
            for observe in monarch.inspect(self.observations_since_last_log):
                with monarch.no_mesh.activate():
                    for metric_logger in self.metrics_loggers:
                        metric_logger.observe(*observe)
            losses = torch.stack(self.losses_since_last_log)
            avg_loss, max_loss = (
                losses.sum(0) / len(self.losses_since_last_log),
                losses.max(0).values,
            )
            global_avg_loss, global_max_loss = (
                avg_loss.reduce("dp", "avg"),
                max_loss.reduce("dp", "max"),
            )

            avg_loss, max_loss, global_avg_loss, global_max_loss = [
                x.item()
                for x in monarch.inspect(
                    (avg_loss, max_loss, global_avg_loss, global_max_loss)
                )
            ]

            time_delta = time.perf_counter() - self.time_last_log

            # tokens per second per device, abbreviated as tps
            tps = self.ntokens_since_last_log / (time_delta)
            # model FLOPS utilization
            # For its definition and calculation, please refer to the PaLM paper:
            # https://arxiv.org/abs/2204.02311
            mfu = 100 * self.num_flop_per_token * tps / self.gpu_peak_flops

            time_end_to_end = time_delta / self.log_freq
            time_data_loading = sum(self.data_loading_times) / len(
                self.data_loading_times
            )
            time_data_loading_pct = 100 * sum(self.data_loading_times) / time_delta

            device_mem_stats = (
                self.device_memory_monitor.call_method_on_shard_and_fetch(
                    "get_peak_stats"
                ).result()
            )

            metrics = {
                "loss_metrics/global_avg_loss": global_avg_loss,
                "loss_metrics/global_max_loss": global_max_loss,
                "throughput(tps)": tps,
                "mfu(%)": mfu,
                "time_metrics/end_to_end(s)": time_end_to_end,
                "time_metrics/data_loading(s)": time_data_loading,
                "time_metrics/data_loading(%)": time_data_loading_pct,
                "memory/max_active(GiB)": device_mem_stats.max_active_gib,
                "memory/max_active(%)": device_mem_stats.max_active_pct,
                "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
                "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
                "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
                "memory/num_ooms": device_mem_stats.num_ooms,
            }
            self.report_metrics(step, metrics)
            logger.info(
                f"step: {step:2}  "
                f"loss: {global_avg_loss:7.4f}  "
                f"memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
                f"({device_mem_stats.max_reserved_pct:.2f}%)  "
                f"tps: {round(tps):,}  "
                f"mfu: {mfu:.2f}%"
            )

            self.losses_since_last_log.clear()
            self.observations_since_last_log.clear()
            self.ntokens_since_last_log = 0
            self.data_loading_times.clear()
            self.time_last_log = time.perf_counter()
            self.device_memory_monitor.call_method_on_shard_and_fetch(
                "reset_peak_stats"
            )
