from __future__ import annotations

import logging

# import os
import sys

import torch

import torch.nn as nn
import torch.optim as optim
from monarch import fetch_shard, get_active_stream, remote, Simulator, Stream
from monarch.parallel import PipelineParallelism

from monarch_supervisor.logging import initialize_logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.WARNING)


class Net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, device="cuda"):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, device=device)
        self.fc2 = nn.Linear(hidden_dim, output_dim, device=device)

    def forward(self, input_tensor):
        x = torch.relu(self.fc1(input_tensor))
        x = self.fc2(x)
        return x


def make_input_output(batch_size, input_dim, output_dim, num_batches=10):
    inputs_list = []
    labels_list = []
    for _ in range(num_batches):
        inputs = torch.randn(batch_size, input_dim).cuda()
        labels = torch.randn(batch_size, output_dim).cuda()
        inputs_list.append(inputs)
        labels_list.append(labels)
    return inputs_list, labels_list


set_worker_logging_level = remote(
    "monarch.worker.worker.set_worker_logging_level", propagate="inspect"
)

# Add current directory to sys.path on the workers.
# Args:
#     new_directory (str): The directory to be added to sys.path.
add_sys_path = remote(
    "monarch.parallel.pipelining.runtime.add_sys_path_impl", propagate="inspect"
)

set_worker_random_seed = remote(
    "monarch.worker.worker.set_random_seed_impl", propagate="inspect"
)


def generate_model_and_optimizer(
    mesh, pp_meshes, input_dim, output_dim, hidden_dim, lr
):
    with mesh.activate():
        model_chunks = torch.nn.ModuleList()
        optimizers = []
        for _, pp_mesh in enumerate(pp_meshes):
            with pp_mesh.activate():
                model_chunk = Net(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dim=hidden_dim,
                )
                model_chunk.train()
                model_chunks.append(model_chunk)
        model = model_chunks
        for module in model:
            optimizer = optim.SGD(module.parameters(), lr=lr)
            optimizers.append(optimizer)
        loss_layer = nn.MSELoss()
        return model, optimizers, loss_layer, []


def train(world_size, pp_dim, batch_size, num_steps):
    simulator = Simulator(
        hosts=1, gpus=world_size, trace_mode="stream_only", build_ir=True
    )
    orig_mesh = simulator.mesh

    input_dim = 2
    output_dim = 2
    hidden_dim = 2
    lr = 0.01

    mesh = orig_mesh.flatten("gpu").split(
        gpu=(
            "dp",
            "pp",
        ),
        pp=pp_dim,
    )
    torch.set_default_device("cuda")

    pp_meshes = [mesh(pp=i) for i in range(pp_dim)]

    with mesh.activate():
        # set_worker_logging_level(logging.DEBUG) # Error @ fbcode/monarch/python/monarch/common/fake.py
        # new_directory = os.path.dirname(os.path.abspath(__file__))
        # add_sys_path(new_directory) # Error @ fbcode/monarch/python/monarch/common/fake.py

        model, optimizers, loss_layer_ref, loss_list_ref = generate_model_and_optimizer(
            mesh=mesh,
            pp_meshes=pp_meshes,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            lr=lr,
        )

        model_loss_list = []
        # ref_model_loss_list = []

        set_worker_random_seed(12345, 0)
        with pp_meshes[0].activate():
            x_list, y_list = make_input_output(
                batch_size, input_dim, output_dim, num_batches=num_steps
            )

        # NOTE: Removed reference model creation.

        pp_stages = [[] for _ in range(world_size)]
        for idx in range(world_size):
            pp_stages[idx].append(model[idx])

        pp_class = PipelineParallelism(
            meshes=pp_meshes,
            stages=pp_stages,
            schedule="dora-dfs",
            batch_size=batch_size,
            loss_fn=loss_layer_ref,
            p2p_stream=Stream("self.p2p_stream"),
            compute_stream=get_active_stream(),
            loss_list=loss_list_ref,
        )

        # NOTE: Removed copying parameters to reference model
        # NOTE: Removed refernce model optimizer creation

        # NOTE: Removed profiling
        for i in range(num_steps):
            x = x_list[i]
            y = y_list[i]

            for j, optimizer in enumerate(optimizers):
                with pp_meshes[j].activate():
                    optimizer.zero_grad()
            remote_loss = pp_class.run(x, y)

            for j, optimizer in enumerate(optimizers):
                with pp_meshes[j].activate():
                    optimizer.step()

            loss = fetch_shard(remote_loss).result()
            model_loss_list.append(loss)

        # NOTE: Removed reference model training
        # NOTE: Removed comparing results from model and reference model

    logging.info("PP example finished successfully")

    orig_mesh.exit()

    ir_pkl_path = "examples/simulator_ir_pp/ir_pp.pkl"
    dag_json_path = "examples/simulator_ir_pp/ir_pp_dag.json"
    filtered_dag_json_path = "examples/simulator_ir_pp/ir_pp_filtered_dag.json"
    data_csv_path = "examples/simulator_ir_pp/ir_pp_data.csv"
    data_timeline_csv_path = "examples/simulator_ir_pp/ir_pp_data_timeline.csv"
    borrows_csv_path = "examples/simulator_ir_pp/ir_pp_borrows.csv"
    sendtensors_csv_path = "examples/simulator_ir_pp/ir_pp_sendtensors.csv"

    simulator.export_ir(ir_pkl_path)
    with open(ir_pkl_path, "rb") as f:
        ir = torch.load(f, weights_only=False)
    ir.export_dag_json(dag_json_path)
    ir.export_data_csv(data_csv_path)
    ir.export_data_timeline_csv(data_timeline_csv_path)
    ir.export_borrows_csv(borrows_csv_path)
    ir.export_sendtensors_csv(sendtensors_csv_path)

    # Testing out DAG filtering
    ir.remove_dag_item_type("Borrow")
    ir.export_dag_json(filtered_dag_json_path)


def main() -> None:
    torch.set_default_device("cuda")
    initialize_logging()
    torch.manual_seed(23456)
    return train(world_size=4, pp_dim=4, batch_size=4, num_steps=5)


if __name__ == "__main__":
    # run the training loop
    ret = main()
    sys.exit(ret)
