import torch
from examples.te import rf_comms_existing_functions

from monarch import local_mesh
from monarch.cached_remote_function import remote_autograd_function
from monarch.common import remote


log = remote("monarch.worker.worker.log", propagate="inspect")


def main():
    o_mesh = local_mesh(gpus=2)
    device_mesh = o_mesh.flatten("gpu")
    device_mesh.activate()
    # Have to set default device instead of initialize on cpu and cast to CUDA
    # because T194391401
    torch.set_default_device("cuda")
    pg = device_mesh.process_group(("gpu",))

    rf_comms_existing_functions._MultiplyAllReduce = remote_autograd_function(
        rf_comms_existing_functions._MultiplyAllReduce
    )
    m = rf_comms_existing_functions.MultiplyAllReduce(pg).cuda()
    x = torch.rand((3), requires_grad=True).cuda()
    y = torch.rand((3), requires_grad=True).cuda()
    x.retain_grad()
    y.retain_grad()
    z = m(x, y)
    z.sum().backward()
    log(x.grad)
    log(y.grad)
    o_mesh.exit()


if __name__ == "__main__":
    main()
