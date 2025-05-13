# pyre-unsafe
from unittest import main, TestCase

import monarch.common.mock_cuda

import torch


def simple_forward_backward(device: str) -> None:
    torch.manual_seed(123)
    m = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.ReLU()).to(device)
    x = torch.rand(10, 3).to(device)
    y = m(x)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(y, torch.randint(3, (10,)).to(device))
    # Under the hood, enabling/disabling CUDA mocking is done with a thread-local
    # flag. By default, backward() executes ops on a different thread than the one
    # we enabled mocking on, which would lead to an invalid memory access. So we need
    # to disable multithreading for backward.
    with torch.autograd.set_multithreading_enabled(False):
        loss.backward()
    return y, m[0].weight.grad, m[0].bias.grad


class TestMockCuda(TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_output_is_garbage(self):
        with monarch.common.mock_cuda.mock_cuda_guard():
            x = torch.arange(9, device="cuda", dtype=torch.float32).reshape(3, 3)
            y = 2 * torch.eye(3, device="cuda")
            true_output = torch.tensor(
                [[0, 2, 4], [6, 8, 10], [12, 14, 16]], dtype=torch.float32
            )
            self.assertFalse(torch.equal((x @ y).cpu(), true_output))

    def test_simple_forward_backward(self):
        # This test just makes sure that the forward and backward pass work
        # and don't crash.
        simple_forward_backward("cuda")

    def test_turn_mock_on_and_off(self):
        cpu_y, cpu_dw, cpu_db = simple_forward_backward("cpu")

        real_y, real_dw, real_db = simple_forward_backward("cuda")
        self.assertTrue(torch.allclose(cpu_y, real_y.cpu()))
        self.assertTrue(torch.allclose(cpu_dw, real_dw.cpu()))
        self.assertTrue(torch.allclose(cpu_db, real_db.cpu()))

        with monarch.common.mock_cuda.mock_cuda_guard():
            mocked_y, mocked_dw, mocked_db = simple_forward_backward("cuda")
            self.assertFalse(torch.allclose(cpu_y, mocked_y.cpu()))
            self.assertFalse(torch.allclose(cpu_dw, mocked_dw.cpu()))
            self.assertFalse(torch.allclose(cpu_db, mocked_db.cpu()))

        real_y, real_dw, real_db = simple_forward_backward("cuda")
        self.assertTrue(torch.allclose(cpu_y, real_y.cpu()))
        self.assertTrue(torch.allclose(cpu_dw, real_dw.cpu()))
        self.assertTrue(torch.allclose(cpu_db, real_db.cpu()))


if __name__ == "__main__":
    main()
