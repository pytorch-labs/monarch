# pyre-unsafe

import monarch
import torch

from torch import nn as nn


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers = []
        for _ in range(2):
            layers.append(nn.Linear(4, 4))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        output = self.layers(input)
        return torch.nn.functional.cross_entropy(output, target)


def test_streams() -> None:
    comms = monarch.Stream("comms")
    t1 = torch.rand(3, 4, device="cuda")
    t2 = t1 @ t1.T
    comm_t2, borrow = comms.borrow(t2, mutable=True)
    with comms.activate():
        comm_t2.reduce_("gpu", "sum")
    borrow.drop()
    t2 *= 2


def train_one_stream(
    model: nn.Module, input: torch.Tensor, target: torch.Tensor
) -> None:
    loss = model(input, target)
    rparameters = list(reversed(list(model.parameters())))
    grads = monarch.grad_generator(loss, rparameters)
    with torch.no_grad():
        it = iter(zip(rparameters, grads))
        todo = next(it, None)
        while todo is not None:
            param, grad = todo
            grad.reduce_("gpu", "sum")
            todo = next(it, None)
            param += 0.01 * grad


def train_two_streams(
    model: nn.Module, input: torch.Tensor, target: torch.Tensor
) -> None:
    comms = monarch.Stream("comms")
    loss = model(input, target)
    rparameters = list(reversed(list(model.parameters())))
    grads = monarch.grad_generator(loss, rparameters)
    with torch.no_grad():
        # NEW: iter also produces the tensor borrowed to the comm stream
        it = iter(
            (param, grad, *comms.borrow(grad, mutable=True))
            for param, grad in zip(rparameters, grads)
        )

        todo = next(it, None)
        while todo is not None:
            param, grad, comm_grad, borrow = todo
            # NEW: compute the reduce on the comm stream
            with comms.activate():
                comm_grad.reduce_("gpu", "sum")
            borrow.drop()
            todo = next(it, None)
            param += 0.01 * grad


def train_two_streams_overlap(
    model: nn.Module, input: torch.Tensor, target: torch.Tensor
) -> None:
    comms = monarch.Stream("comms")
    loss = model(input, target)
    rparameters = list(reversed(list(model.parameters())))
    grads = monarch.grad_generator(loss, rparameters)
    with torch.no_grad():
        it = iter(
            (param, grad, *comms.borrow(grad, mutable=True))
            for param, grad in zip(rparameters, grads)
        )

        todo = next(it, None)
        while todo is not None:
            param, grad, comm_grad, borrow = todo
            with comms.activate():
                comm_grad.reduce_("gpu", "sum")
            todo = next(it, None)
            # NEW: delay the borrow as late as possible
            borrow.drop()
            param += 0.01 * grad


def simulate() -> None:
    """
    Testbench for testing simulator IR functionality.
    """
    simulator = monarch.Simulator(
        hosts=1, gpus=4, trace_mode="stream_only", build_ir=True
    )
    simulator_mesh = simulator.mesh
    with simulator_mesh.activate():
        test_streams()

        """
        model = Net()

        train_one_stream(
            model, torch.rand(3, 4), torch.full((3,), 1, dtype=torch.int64)
        )
        train_two_streams(
            model, torch.rand(3, 4), torch.full((3,), 1, dtype=torch.int64)
        )
        train_two_streams_overlap(
            model, torch.rand(3, 4), torch.full((3,), 1, dtype=torch.int64)
        )
        """

    simulator.mesh.exit()

    ir_pkl_path = "examples/simulator_ir/ir_test.pkl"
    dag_json_path = "examples/simulator_ir/ir_test_dag.json"
    filtered_dag_json_path = "examples/simulator_ir/ir_test_filtered_dag.json"
    data_csv_path = "examples/simulator_ir/ir_test_data.csv"
    data_timeline_csv_path = "examples/simulator_ir/ir_test_data_timeline.csv"
    borrows_csv_path = "examples/simulator_ir/ir_test_borrows.csv"
    sendtensors_csv_path = "examples/simulator_ir/ir_test_sendtensors.csv"

    # Testing out serialization and exports
    simulator.export_ir(ir_pkl_path)
    with open(ir_pkl_path, "rb") as f:
        ir = torch.load(f, weights_only=False)
    ir.export_dag_json(dag_json_path)
    ir.export_data_csv(data_csv_path)
    ir.export_data_timeline_csv(data_timeline_csv_path)
    ir.export_borrows_csv(borrows_csv_path)
    ir.export_sendtensors_csv(sendtensors_csv_path)

    # Testing out DAG filtering
    ir.remove_dag_item_type("UndefinedType")
    ir.remove_dag_item_type(["CallFunction"])
    ir.export_dag_json(filtered_dag_json_path)


def simulate_reduce() -> None:
    simulator = monarch.Simulator(
        hosts=3, gpus=4, trace_mode="stream_only", build_ir=True
    )
    simulator_mesh = simulator.mesh
    simulator_meshes = [
        simulator_mesh(host=i) for i in range(simulator_mesh.size("host"))
    ]

    with simulator_meshes[0].activate():
        x = torch.rand(3, 4, device="cuda")
        x = x @ x.T

    with simulator_mesh.activate():
        x = torch.rand(3, 4, device="cuda")
        x.reduce_("gpu", "sum")

    simulator.mesh.exit()

    ir_pkl_path = "examples/simulator_ir_reduce/ir_test_reduce.pkl"
    dag_json_path = "examples/simulator_ir_reduce/ir_test_reduce_dag.json"
    data_csv_path = "examples/simulator_ir_reduce/ir_test_reduce_data.csv"
    data_timeline_csv_path = (
        "examples/simulator_ir_reduce/ir_test_reduce_data_timeline.csv"
    )
    borrows_csv_path = "examples/simulator_ir_reduce/ir_test_reduce_borrows.csv"
    sendtensors_csv_path = "examples/simulator_ir_reduce/ir_test_reduce_sendtensors.csv"

    simulator.export_ir(ir_pkl_path)
    with open(ir_pkl_path, "rb") as f:
        ir = torch.load(f, weights_only=False)
    ir.export_dag_json(dag_json_path)
    ir.export_data_csv(data_csv_path)
    ir.export_data_timeline_csv(data_timeline_csv_path)
    ir.export_borrows_csv(borrows_csv_path)
    ir.export_sendtensors_csv(sendtensors_csv_path)


def simulate_send() -> None:
    simulator = monarch.Simulator(
        hosts=3, gpus=4, trace_mode="stream_only", build_ir=True
    )
    simulator_mesh = simulator.mesh
    simulator_meshes = [
        simulator_mesh(host=i) for i in range(simulator_mesh.size("host"))
    ]

    for i in range(len(simulator_meshes) - 1):
        with simulator_meshes[len(simulator_meshes) - 1 - i].activate():
            comms = monarch.Stream("comms")
            x = torch.rand(3, 4, device="cuda")
            comm_x, borrow = comms.borrow(x, mutable=True)
            with comms.activate():
                comm_x.reduce_("gpu", "sum")
                borrow.drop()
            x.to_mesh(simulator_meshes[len(simulator_meshes) - 1 - (i + 1)])

    simulator.mesh.exit()

    ir_pkl_path = "examples/simulator_ir_send/ir_test_send.pkl"
    dag_json_path = "examples/simulator_ir_send/ir_test_send_dag.json"
    data_csv_path = "examples/simulator_ir_send/ir_test_send_data.csv"
    data_timeline_csv_path = "examples/simulator_ir_send/ir_test_send_data_timeline.csv"
    borrows_csv_path = "examples/simulator_ir_send/ir_test_send_borrows.csv"
    sendtensors_csv_path = "examples/simulator_ir_send/ir_test_send_sendtensors.csv"

    simulator.export_ir(ir_pkl_path)
    with open(ir_pkl_path, "rb") as f:
        ir = torch.load(f, weights_only=False)

    ir.export_dag_json(dag_json_path)
    ir.export_data_csv(data_csv_path)
    ir.export_data_timeline_csv(data_timeline_csv_path)
    ir.export_borrows_csv(borrows_csv_path)
    ir.export_sendtensors_csv(sendtensors_csv_path)


def simulate_real() -> None:
    """
    Testbench for testing equivalent functionality on real Monarch/workers.
    """
    mesh = monarch.python_local_mesh()
    with mesh.activate():
        test_streams()
    mesh.exit()


# TODO: Make these into separate unit tests.
def main() -> None:
    torch.set_default_device("cuda")

    simulate()
    # simulate_reduce()
    # simulate_send()
    # simulate_real()


if __name__ == "__main__":
    main()
