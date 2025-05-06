"""Learner instances, modelled as a UDF."""

from typing import Dict, List, Tuple

# Required for OpaqueRef
import monarch.common.tree  # noqa: F401
import torch
from monarch.common.opaque_ref import OpaqueRef
from rl import model
from rl.config import Config
from rl.data import Trajectory


class Learner:
    def __init__(self, config: Config):
        self.model = model.create_model(config).to("cuda")
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            eps=config.epsilon,
            fused=True,
        )

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()

    def train_step(
        self, trajectories: List[Trajectory]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        queries_and_response, logits, rewards = zip(*trajectories)
        queries_and_responses = torch.stack(queries_and_response).to("cuda")
        logits = torch.stack(logits).to("cuda")
        rewards = torch.stack(rewards).to("cuda")

        logits = self.model(queries_and_responses)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = -torch.mean(rewards.detach().squeeze() * log_probs.sum(dim=[1, 2]))
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optim.step()
        return loss, rewards


def create(config: Config) -> OpaqueRef:
    return OpaqueRef(Learner(config))


def get_state_dict(ref: OpaqueRef) -> Dict[str, torch.Tensor]:
    return ref.value.get_state_dict()


def step(
    ref: OpaqueRef, trajectories: List[Trajectory]
) -> Tuple[torch.Tensor, torch.Tensor]:
    return ref.value.train_step(trajectories)
