# pyre-unsafe
import argparse
from typing import List

import torch

from monarch import OpaqueRef, remote
from monarch.rust_local_mesh import local_mesh, LoggingLocation, SocketType

initialize_dlrm_model = remote(
    "dlrm.remote_functions.initialize_model",
    propagate=lambda rank,
    world_size,
    float_features_count,
    dense_arch_layers,
    over_arch_layers,
    num_sprase_features,
    embedding_dim,
    embedding_hash_size: OpaqueRef(None),
)

dlrm_train_step = remote(
    "dlrm.remote_functions.dlrm_train_setp",
    propagate=lambda model_ref, batch_size: torch.ones(1),
)

log = remote("monarch.worker._testing_function.log", propagate="inspect")


def main() -> None:
    """DOC_STRING"""
    parser = argparse.ArgumentParser(description="DLRM on Monarch argparser")
    parser.add_argument(
        "--hosts",
        type=int,
        default=1,
        help="Nbmber of hosts for local_mesh.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=2,
        help="Number of GPUs per host for local_mesh.",
    )
    parser.add_argument(
        "--float-features-count",
        type=int,
        default=13,
        help="Number of dense/float features.",
    )
    parser.add_argument(
        "--dense-arch-layers",
        type=List[int],
        default=[32, 16],
        help="Dense arch layers.",
    )
    parser.add_argument(
        "--over-arch-layers",
        type=List[int],
        default=[64, 1],
        help="Over arch layers.",
    )
    parser.add_argument(
        "--num-sprase-features",
        type=int,
        default=2,
        help="Number of sprase features.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=16,
        help="Embedding dimension for sprase features.",
    )
    parser.add_argument(
        "--embedding-hash-size",
        type=int,
        default=10,
        help="Number of embeddings per sparse feature.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="The batch size.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=100,
        help="The number of batches.",
    )

    args = parser.parse_args()

    device_mesh = local_mesh(
        hosts=args.hosts,
        gpus_per_host=args.gpus,
        socket_type=SocketType.UNIX,
        logging_location=LoggingLocation.DEFAULT,
    )

    with device_mesh as device_mesh:
        with device_mesh.activate():
            torch.set_default_device("cuda")

            # initalize and shards the dlrm model
            model = initialize_dlrm_model(
                device_mesh.rank("gpu"),
                device_mesh.numdevices(),
                args.float_features_count,
                args.dense_arch_layers,
                args.over_arch_layers,
                args.num_sprase_features,
                args.embedding_dim,
                args.embedding_hash_size,
            )

            for _ in range(args.num_batches):
                # run model train step
                loss = dlrm_train_step(model, args.batch_size)
                log("loss at rank: %s is %s", device_mesh.rank("gpu"), loss)
        device_mesh.exit()


if __name__ == "__main__":
    main()
