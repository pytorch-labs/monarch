# pyre-unsafe
import logging
import time
from typing import Optional

import monarch

import torch

from monarch.common._coalescing import compile
from morpho.activation_checkpointing import apply_ac
from morpho.checkpointer import Checkpointer, CheckpointState
from morpho.config import ParallelismConfig

from morpho.dataloader import Cursor, Dataloader
from morpho.droppable import null_droppable
from morpho.grad_clipping import clip_grad_norm_
from morpho.optimizer import Optimizer
from morpho.report import Report

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        mesh,
        model,
        optimizer: Optimizer,
        checkpointer: Checkpointer,
        report: Report,
        parallelism: ParallelismConfig,
        max_norm: float,  # TODO: should grad norms get their own input class?
        starting_checkpoint: Optional[CheckpointState] = None,
    ):
        self.mesh: monarch.DeviceMesh = mesh
        self.model = model
        self.report = report
        self.optimizer = optimizer
        self.checkpointer = checkpointer
        self.max_norm = max_norm

        model.to_empty(device="cuda")
        if starting_checkpoint is not None:
            self.model.load_state_dict(starting_checkpoint.model_state)
            self.report.load_state_dict(starting_checkpoint.report_state)
        else:
            # XXX: titan sets the random number state the same for all replicas,
            # which does not seem right, but we have to copy it to match results
            # so we just init_weights and get the same weights everywhere.
            model.init_weights(buffer_device="cuda")
            model.train()
        report.report_memory_usage("model")

        for layer in self.model.layers.values():
            layer.register_forward_hook(
                lambda module, args, output: self.report.observe("layer", output)
            )
        param_values = dict(self.model.named_parameters())
        self.named_parameters = []
        self.parameters = []
        # we use the parameter_names in the optimizer rather than directly from
        # the model, so we can
        # (1) use it as a way to specify which parameters to optimize
        # (2) use the _order_ of the parameter_names as defined in the optimizer
        #     object as the suggested order of use for the purpose of sharding
        #     and gradient steps.
        for name in self.optimizer.parameter_names:
            param = param_values[name]
            self.named_parameters.append((name, param))
            self.parameters.append(param)

        if starting_checkpoint is None:
            self.optimizer.initialize_state(self.named_parameters)
        else:
            self.optimizer.load_state_dict(starting_checkpoint.optimizer_state)

        apply_ac(model, parallelism.ac_freq)
        self.checkpointer_borrow = null_droppable

    def train(
        self,
        data_loader: Dataloader,
        starting_cursor: Optional[Cursor],
        start_step: int,
        end_step: int,
    ):
        data_iterator = iter(data_loader.generate(starting_cursor))
        # train loop
        logger.info(
            f"Training starts at step {start_step + 1}, "
            f"with local batch size {data_loader.batch_size}, "
            f"global batch size {data_loader.batch_size}, "
            f"sequence length {data_loader.seq_len}, "
            f"total steps {end_step} "
        )

        for step in range(start_step, end_step):
            data_load_start = time.perf_counter()
            input_ids, labels, cursor = next(data_iterator)
            data_load_time = time.perf_counter() - data_load_start
            # TODO: either the model or data loader should know
            # the device
            input_ids = input_ids.to("cuda")
            labels = labels.to("cuda")

            tstep = torch.tensor(step, dtype=torch.double)
            loss, report_step_state = self.step(tstep, input_ids, labels)
            self.report.load_step_state(report_step_state)

            self.report.report_completed_step(
                step + 1, loss, labels.numel(), data_load_time
            )

            if self.checkpointer.should_save(step, step + 1 == end_step):
                self.checkpointer_borrow = self.checkpointer.save(
                    CheckpointState(
                        step=step,
                        model_state=self.model.state_dict(),
                        optimizer_state=self.optimizer.state_dict(),
                        report_state=self.report.state_dict(),
                        dataloader_cursor=cursor,
                    )
                )

    @compile
    def step(self, step: torch.Tensor, input_ids, labels):
        # Non-PP forward / backward
        pred = self.model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            pred.flatten(0, 1).float(), labels.flatten(0, 1)
        )
        self.report.observe("layer", loss)
        # pred.shape=(bs, seq_len, vocab_size)
        # need to free to before bwd to avoid peaking memory
        del pred
        gradients = torch.autograd.grad(loss, self.parameters)

        # clip gradients
        clip_grad_norm_(gradients, self.max_norm, dims=("dp",))

        for grad in reversed(gradients):
            grad.reduce_("dp", "avg")  # type: ignore
            self.report.observe("grad", grad)

        # make sure the checkpointer returns any state it was borrowing while
        # we trained
        self.checkpointer_borrow.drop()
        # optimizer step
        with torch.no_grad():
            self.optimizer.step(step, self.named_parameters, gradients)
        for p in self.parameters:
            self.report.observe("param", p)
        return loss.detach(), self.report.step_state()

    # metric_logger.close()
    logger.info("Training completed")
