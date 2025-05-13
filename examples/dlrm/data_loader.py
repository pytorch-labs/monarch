# pyre-unsafe
import torch
from monarch import OpaqueRef, remote_generator
from monarch.common.pipe import FakePipe
from monarch.worker.worker import ProcessPipe
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


@remote_generator("dlrm.data_loader.generate_dlrm_random_batch", max_messages=50)
def data_loader_pipe(
    p: FakePipe,
    float_features_count,
    num_sprase_features,
    embedding_hash_size,
    batch_size,
):
    # generate_dlrm_random_batch is the actual function called on the worker.
    while True:
        x = torch.zeros(
            (batch_size, float_features_count),
            dtype=torch.int64,
        )
        y = OpaqueRef(None)
        yield x, y


def generate_id_list_features(
    num_sprase_features,
    embedding_hash_size,
    batch_size,
    feature_name_prefix="id_list_feature_",
):
    feature_keys = [f"{feature_name_prefix}{i}" for i in range(num_sprase_features)]
    values = []
    lengths = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _ in range(num_sprase_features):
        num_values = torch.randint(0, embedding_hash_size + 1, (batch_size,))
        total_values = num_values.sum()
        feature_values = torch.randint(0, 100, (total_values,))
        values.append(feature_values)
        lengths.append(num_values)
    kjt = KeyedJaggedTensor.from_lengths_sync(
        feature_keys,
        torch.cat(values),
        torch.cat(lengths),
    ).to(device)
    return kjt


def generate_dlrm_random_batch(
    p: ProcessPipe,
    float_features_count,
    num_sprase_features,
    embedding_hash_size,
    batch_size,
    feature_name_prefix="id_list_feature_",
):
    while True:
        kjt = generate_id_list_features(
            num_sprase_features, embedding_hash_size, batch_size, feature_name_prefix
        )
        # (dense, sparse)
        p.send(
            (
                torch.rand(batch_size, float_features_count),
                OpaqueRef(kjt),
            )
        )
