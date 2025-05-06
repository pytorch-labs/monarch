import sys

from examples.nanoGPT.data_loader import get_batch

from monarch import local_mesh
from monarch.common.device_mesh import remote

device_mesh = local_mesh(hosts=2, gpus=2)


log = remote("monarch.worker._testing_function.log", propagate="inspect")


# you can select a subset of the devices with
# name-based indexing
data_loaders = device_mesh(host=0)
trainers = device_mesh(host=1)

with data_loaders.activate():
    x, y = get_batch(
        split="train",
        data_dir=sys.argv[1],
        device=None,
        device_type=None,
        block_size=5,
        batch_size=3,
    )

    x = x.cuda()
    y = y.cuda()
    log("orig x:\n%s", x)
    log("orig y:\n%s", y)

    a = x.to_mesh(trainers)
    b = y.to_mesh(trainers)

with trainers.activate():
    log(a)
    log(b)

device_mesh.exit()
