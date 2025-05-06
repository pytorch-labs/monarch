"""
1. This file implements a GRPO algorithm, originally from the paper
    "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
    (https://arxiv.org/abs/2402.03300).
2. Some functions are simplified from the Huggingface TRL library
    (https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py).
3. Some functions are simplified from nanoGPT
    (https://github.com/karpathy/nanoGPT/blob/master/sample.py).
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Union

import numpy as np

import torch
import torch.nn as nn

import torch.nn.functional as F
from monarch import fetch_shard

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """
    Configuration class for GRPO model settings.

    Attributes:
        max_prompt_length (int): Maximum length of the prompt.
        max_completion_length (int): Maximum length of the completion.
        num_generations (int): Number of generations to sample.
        beta (float): Regularization parameter.
        temperature (float): Sampling temperature for generation.
    """

    max_prompt_length: int = 64
    max_completion_length: int = 128
    num_generations: int = 4
    beta: float = 0.01
    temperature: float = 0.7
    top_k: int = 100


class GRPO:
    def __init__(
        self,
        model: Union[str, nn.Module],
        ref_model: Union[str, nn.Module],
        reward_funcs: list[Callable],
        args: GRPOConfig,
        encoder: Callable,
        decoder: Callable,
        device: torch.device,
        mesh=None,
        policy_mesh=None,
        ref_mesh=None,
    ):
        """
        Initializes the GRPO class with the given model, reward functions, configuration, and optional encoder, decoder, and device.

        Args:
            model (Union[str, nn.Module]): The model to be used, which can be a string identifier or a PyTorch module.
            reward_funcs (list[Callable]): A list of reward functions to evaluate the model's performance.
            args (GRPOConfig): Configuration settings for the GRPO model.
            encoder A function for converting input prompt strings to token index lists.
            decoder A functionfor converting model output token index lists to strings.
            device torch.device: The device on which the model will be run.
        """
        self.model = model
        self.ref_model = ref_model
        self.reward_funcs = reward_funcs
        self.args = args
        self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)
        self.reward_processing_classes = [None] * len(reward_funcs)
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = (
            args.max_completion_length
        )  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.beta = args.beta

        self.mesh = mesh
        self.policy_mesh = policy_mesh
        self.ref_mesh = ref_mesh

        self.update_reference_model()
        logger.info("GRPO initialized")

    def generate(
        self,
        prompt: str | torch.Tensor,
        num_generations: int = 1,
        max_length: int = 1024,
        temperature: float = 0.7,
        top_k: int = 100,
    ):
        """
        Generates multiple completions for a given prompt using the model.

        Args:
            prompt (str | torch.Tensor): The input prompt, which can be a string or a tensor.
            num_generations (int): Number of completions to generate. Default is 1.
            max_length (int): Maximum length of each generated completion. Default is 1024.
            temperature (float): Sampling temperature for generation. Default is 0.7.
            top_k (int): Number of top tokens to consider during sampling. Default is 100.

        Returns:
            tuple: A tuple containing the prompt tensor, list of response tensors, and list of response strings.
        """
        with self.policy_mesh.activate():
            logger.debug(
                f"Generating {num_generations=} completions with size{max_length=} for prompt: {prompt}"
            )
            if not isinstance(prompt, torch.Tensor):
                start_ids = self.encoder(prompt)
                x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[
                    None, ...
                ]
            else:
                x = prompt.unsqueeze(0) if prompt.dim() == 1 else prompt
            ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

            y_list = []
            y_tensor_list = []
            # run generation
            with torch.no_grad():
                with ctx:
                    for k in range(num_generations):
                        y = self.model.generate(
                            x, max_length, temperature=temperature, top_k=top_k
                        )
                        logger.debug(f"Sample {k} returns: {y.shape} {y[0]=}")
                        y0_local = fetch_shard(y[0]).result()
                        y_ids = y0_local.tolist()
                        y_decode = self.decoder(y_ids)

                        y_list.append(y_decode)
                        y_tensor_list.append(y[0])

                        logger.debug(f"Sample {k} returns: {y0_local}")
                        logger.debug(y_decode)
                        logger.debug("---------------")

            prompt_id_tensor = x
            response_tensor_list = y_tensor_list
            reponse_list = y_list
            return prompt_id_tensor, response_tensor_list, reponse_list

    def _prepare_inputs(
        self,
        prompts,
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Prepares the input data for the model by generating completions and calculating rewards.

        Args:
            prompts: A list of input prompts for which completions are to be generated.

        Returns:
            A dictionary containing:
                - 'prompt_ids': Tensor of prompt token IDs.
                - 'completion_ids': Padded tensor of completion token IDs.
                - 'ref_per_token_logps': Log probabilities per token from the reference model.
                - 'advantages': Calculated advantages for the generated completions.
        """
        device = self.device
        # PART 1: define prompts
        prompt = prompts[0]
        assert len(prompts) == 1, "Only support single prompt for now"
        with self.policy_mesh.activate():
            prompt_ids_tensor, response_tensor_list, reponse_text_list = self.generate(
                prompt,
                num_generations=self.args.num_generations,
                max_length=self.max_completion_length,
                temperature=self.args.temperature,
                top_k=self.args.top_k,
            )
            prompt_ids_tensor_size = prompt_ids_tensor.size(-1)
            response_tensor_padded = pad(
                response_tensor_list,
                padding_value=0,
                max_length=prompt_ids_tensor_size + self.max_completion_length,
            )
            logits_to_keep = response_tensor_padded.size(1)
            response_tensor_padded_to_mesh_ref = response_tensor_padded.to_mesh(
                self.ref_mesh
            )

        with self.ref_mesh.activate():
            with torch.inference_mode():
                ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                with ctx:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model,
                        # response_tensor_padded,
                        response_tensor_padded_to_mesh_ref,
                        attention_mask=None,
                        logits_to_keep=logits_to_keep,
                    )
            completions = reponse_text_list

        rewards_per_func = torch.zeros(
            len(completions), len(self.reward_funcs), device=device
        )
        for i, reward_func in enumerate(self.reward_funcs):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            output_reward_func = reward_func(prompts=prompts, completions=completions)
            rewards_per_func[:, i] = output_reward_func[:, 0]

        rewards = rewards_per_func
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        logger.debug(f"{rewards=} {advantages=}")

        return {
            "prompt_ids": prompt_ids_tensor,
            "completion_ids": response_tensor_padded,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    def compute_loss(self, model, inputs):
        """
        Computes the loss for the model based on the provided inputs.

        Args:
            model: The model for which the loss is being computed.
            inputs: A dictionary containing the necessary inputs for loss computation,
                including completion IDs, reference log probabilities, and advantages.

        Returns:
            tuple: A tuple containing the computed loss, advantages, and mean KL divergence.
        """
        # Compute the per-token log probabilities for the model
        completion_ids = inputs["completion_ids"]
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        with self.policy_mesh.activate():
            per_token_logps = self._get_per_token_logps(
                model,
                completion_ids,
                attention_mask=None,
                logits_to_keep=logits_to_keep,
            )

            # Compute the KL divergence between the model and the reference model
            ref_per_token_logps = inputs["ref_per_token_logps"]
            ref_per_token_logps_to_mesh_policy = ref_per_token_logps.to_mesh(
                self.policy_mesh
            )
            per_token_kl = (
                torch.exp(ref_per_token_logps_to_mesh_policy - per_token_logps)
                - (ref_per_token_logps_to_mesh_policy - per_token_logps)
                - 1
            )
            advantages = inputs["advantages"]
            advantages_mtensor = advantages
            per_token_loss = torch.exp(
                per_token_logps - per_token_logps.detach()
            ) * advantages_mtensor.unsqueeze(1)
            per_token_loss = -(per_token_loss - self.beta * per_token_kl)
            loss = per_token_loss.mean()
            loss = loss.mean()
            mean_kl = per_token_kl.mean()
            logger.debug(f"{loss=} {advantages_mtensor=} {mean_kl=}")
            return loss, advantages_mtensor, mean_kl

    def prediction_step(
        self,
        inputs,
    ):
        """
        Performs a prediction step by preparing inputs and computing the loss.

        Args:
            inputs: The input data required for the prediction step.

        Returns:
            tuple: A tuple containing the computed loss, and two None values as placeholders.
        """
        model = self.model
        inputs = self._prepare_inputs(inputs)
        ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        with ctx:
            loss, _, _ = self.compute_loss(model, inputs)
        return loss, None, None

    def update_reference_model(self):
        if self.mesh is None:
            policy_model_state_dict = self.model.state_dict()
            self.ref_model.load_state_dict(policy_model_state_dict)
        else:
            with self.mesh.activate():
                policy_model_state_dict = self.model.state_dict()
                ref_model_state_dict = {}
                for k, v in policy_model_state_dict.items():
                    if k in self.ref_model.state_dict():
                        v = v.to_mesh(self.ref_mesh)
                        ref_model_state_dict[k] = v
                    else:
                        logger.warning(
                            f"Key {k} not found in reference model state dict"
                        )
                with self.ref_mesh.activate():
                    self.ref_model.load_state_dict(ref_model_state_dict)

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        logits, loss = model(
            idx=input_ids,
            return_full_logits=True,
            # logits_to_keep=logits_to_keep + 1,
        )
        logits = logits[
            :, :logits_to_keep, :
        ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids.clone()[:, :logits_to_keep]

        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(
            logits, input_ids
        )  #  compute logprobs for the input tokens


# from .utils import generate_model_card, get_comet_experiment_url, pad, selective_log_softmax
def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(
            logits, dim=-1, index=index.unsqueeze(-1)
        ).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = (
            selected_logits - logsumexp_values
        )  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(
            logits, index
        ):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(
                dim=-1, index=row_labels.unsqueeze(-1)
            ).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def pad(
    tensors: list[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "right",
    max_length: int = None,
) -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension, with a maximum length for the second dimension.
    Args:
        tensors (`list[torch.Tensor]`): List of input tensors to pad.
        padding_value (`int`): Value to use for padding. Default is 0.
        padding_side (`str`): Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.
        max_length (`int`): Maximum length for the second dimension. If None, no limit is applied.
    Returns:
        `torch.Tensor`: A single tensor containing the padded tensors.
    Examples:
        >>> import torch
        >>> pad([torch.tensor([1, 2, 3]), torch.tensor([4, 5])], max_length=3)
        tensor([[1, 2, 3],
                [4, 5, 0]])
    """
    # Determine the maximum shape for each dimension
    max_shape = np.max([t.shape for t in tensors], axis=0).tolist()
    # Apply max_length constraint to the second dimension
    if max_length is not None:
        max_shape[0] = min(max_shape[0], max_length)
    # Create an output tensor filled with the padding value
    output = torch.full(
        (len(tensors), *max_shape),
        padding_value,
        dtype=tensors[0].dtype,
        device=tensors[0].device,
    )
    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        seq_length = min(t.shape[0], max_shape[0])
        if padding_side == "left":
            seq_slice = slice(max_shape[0] - seq_length, max_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, seq_length)
        else:
            raise ValueError("padding_side must be 'left' or 'right'")
        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t[:seq_length]
    return output
