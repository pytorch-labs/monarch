from torchrec.models.dlrm import DLRM
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingBagConfig,
)


def get_dlrm(
    float_features_count,
    dense_arch_layers,
    over_arch_layers,
    num_embedding_tables,
    embedding_dim,
    embedding_hash_size,
):
    return DLRM(
        embedding_bag_collection=_create_embedding_bag_collection(
            num_embedding_tables, embedding_dim, embedding_hash_size
        ),
        dense_in_features=float_features_count,
        dense_arch_layer_sizes=dense_arch_layers,
        over_arch_layer_sizes=over_arch_layers,
    )


def _create_embedding_bag_collection(num_tables, embedding_dim, hash_size):
    feature_name_prefix = "id_list_feature_"
    configs = []
    for i in range(num_tables):
        configs.append(
            EmbeddingBagConfig(
                name=f"{feature_name_prefix}{i}",
                embedding_dim=embedding_dim,
                num_embeddings=hash_size,
                feature_names=[f"{feature_name_prefix}{i}"],
            )
        )
    return EmbeddingBagCollection(tables=configs)
