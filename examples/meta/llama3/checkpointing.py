# pyre-unsafe
from collections import defaultdict

import torch
from monarch.common.remote import remote
from monarch.common.tensor import Tensor as MonarchTensor
from monarch.common.tree import tree_map


save_checkpoint = remote(
    "llama3.remote_functions.worker_save_checkpoint", propagate="inspect"
)


def checkpoint(path, iter, pp_stages, pp_meshes, optimizers):
    per_pp_mesh_model_dict = {}
    for p_mesh, p_stage in zip(pp_meshes, pp_stages):
        stage_state_dict = {}
        for module in p_stage:
            module.state_dict(destination=stage_state_dict, prefix=module.name + ".")
        per_pp_mesh_model_dict[p_mesh] = stage_state_dict

    per_pp_mesh_optim_dict = {}
    for p_mesh, o in zip(pp_meshes, optimizers):
        per_pp_mesh_optim_dict[p_mesh] = o.state_dict()

    for (pp_mesh_model, per_mesh_model_dict), (
        pp_mesh_optim,
        per_mesh_optim_dict,
    ) in zip(per_pp_mesh_model_dict.items(), per_pp_mesh_optim_dict.items()):
        assert pp_mesh_model == pp_mesh_optim
        with pp_mesh_model.activate():
            save_checkpoint(
                path,
                per_mesh_model_dict,
                per_mesh_optim_dict,
                iter,
                pp_mesh_model.process_idx(),
                None,
            )


def transform_checkpoint_for_te(checkpoint, model, rank, tp_size):
    def slice_tensor_by_rank(tensor, dim, rank, n_ranks):
        M, N = tensor.shape
        rank = rank.reshape(1, 1)
        if dim == 0:
            tensor = tensor.reshape(n_ranks, M // n_ranks, N)
            return tensor[rank].squeeze(dim=(0, 1))
        elif dim == 1:
            tensor = tensor.reshape(M, n_ranks, N // n_ranks)
            return tensor[:, rank, :].squeeze(dim=(1, 2))

    new_checkpoint = {}
    for idx, layer in enumerate(model.layers):
        extra_state = layer.feed_forward.get_extra_state()
        k = f"layers.{idx}.feed_forward._extra_state"
        extra_state.seek(0)
        if torch.load(extra_state) is None:
            new_checkpoint[k] = extra_state

    for k, v in checkpoint.items():
        if "feed_forward" in k:
            # layers.0.feed_forward.w1.weight
            _, layer, _, weight, _ = k.split(".")
            k = f"layers.{layer}.feed_forward.fc{weight[1:]}_weight"
            if weight == "w1":
                v = slice_tensor_by_rank(v, 0, rank, tp_size)
                w3_weight = checkpoint[f"layers.{layer}.feed_forward.w3.weight"]
                w3_weight = slice_tensor_by_rank(w3_weight, 0, rank, tp_size)
                v = torch.cat([v, w3_weight])
            elif weight == "w3":
                continue
            else:
                v = slice_tensor_by_rank(v, 1, rank, tp_size)

        elif "ffn_norm" in k:
            k = k.replace("ffn_norm.weight", "feed_forward.layer_norm_weight")
        new_checkpoint[k] = v

    return new_checkpoint
