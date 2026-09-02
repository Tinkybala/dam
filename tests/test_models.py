import numpy as np
import pandas as pd
import pytest
import torch

from src.models import BPRMatrixFactorization, GMF, MLP, MostPopular, NeuMF
from src.training import (
    iter_bpr_batches,
    iter_bpr_batches_torch,
    iter_pointwise_batches,
    normalized_confidence_weights,
)


def test_popular_counts_distinct_training_users():
    train = pd.DataFrame(
        [(1, 10), (1, 10), (2, 10), (3, 20)],
        columns=["user_id", "anime_id"],
    )
    model = MostPopular().fit(train)

    assert model.predict(pd.Series([10, 20, 30])).tolist() == [2.0, 1.0, 0.0]


def test_bpr_loss_prefers_larger_positive_score():
    model = BPRMatrixFactorization(1, 2, 1)
    with torch.no_grad():
        model.user_embedding.weight.fill_(1.0)
        model.item_embedding.weight.copy_(torch.tensor([[2.0], [-1.0]]))
        model.item_bias.weight.zero_()

    good = model.pairwise_loss(torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
    bad = model.pairwise_loss(torch.tensor([0]), torch.tensor([1]), torch.tensor([0]))

    assert float(good) < float(bad)


def test_bpr_sampler_never_returns_observed_pairs():
    users = np.array([0, 0, 1, 1], dtype=np.int64)
    positives = np.array([0, 1, 1, 2], dtype=np.int64)
    item_count = 5
    observed_codes = np.array([0, 1, 6, 7], dtype=np.int64)
    batches = list(
        iter_bpr_batches(
            users,
            positives,
            observed_codes,
            item_count,
            batch_size=2,
            negatives_per_positive=3,
            rng=np.random.default_rng(42),
        )
    )

    sampled_codes = np.concatenate(
        [batch_users * item_count + negatives for batch_users, _, negatives in batches]
    )
    assert len(sampled_codes) == len(users) * 3
    assert not np.isin(sampled_codes, observed_codes).any()


def test_torch_bpr_sampler_never_returns_observed_pairs():
    users = np.array([0, 0, 1, 1], dtype=np.int64)
    positives = np.array([0, 1, 1, 2], dtype=np.int64)
    observed_codes = np.array([0, 1, 6, 7], dtype=np.int64)
    batches = list(
        iter_bpr_batches_torch(
            users,
            positives,
            observed_codes,
            item_count=5,
            batch_size=2,
            negatives_per_positive=3,
            device=torch.device("cpu"),
        )
    )
    sampled_codes = torch.cat(
        [batch_users * 5 + negatives for batch_users, _, negatives in batches]
    )
    assert len(sampled_codes) == len(users) * 3
    assert not torch.isin(sampled_codes, torch.from_numpy(observed_codes)).any()


def test_bpr_sampler_validates_lengths():
    with pytest.raises(ValueError, match="same length"):
        next(
            iter_bpr_batches(
                np.array([0, 1]),
                np.array([0]),
                np.array([], dtype=np.int64),
                item_count=3,
                batch_size=2,
                negatives_per_positive=1,
                rng=np.random.default_rng(42),
            )
        )


@pytest.mark.parametrize("model_class", [GMF, MLP, NeuMF])
def test_pointwise_models_return_one_logit_per_pair(model_class):
    kwargs = {"user_count": 3, "item_count": 4, "embedding_dim": 2}
    if model_class is not GMF:
        kwargs["hidden_layers"] = [4, 2]
    model = model_class(**kwargs)

    logits = model(torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]))

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_confidence_weights_have_mean_one_and_alpha_zero_is_control():
    ratings = np.array([7, 8, 10])

    control = normalized_confidence_weights(ratings, 7, 10, alpha=0)
    weighted = normalized_confidence_weights(ratings, 7, 10, alpha=2)

    assert control.tolist() == [1.0, 1.0, 1.0]
    assert float(weighted.mean()) == pytest.approx(1.0)
    assert weighted[0] < weighted[1] < weighted[2]


def test_pointwise_sampler_labels_and_unseen_negatives():
    users = np.array([0, 1], dtype=np.int64)
    positives = np.array([0, 1], dtype=np.int64)
    weights = np.array([0.75, 1.25], dtype=np.float32)
    observed_codes = np.array([0, 5], dtype=np.int64)
    batch = next(
        iter_pointwise_batches(
            users,
            positives,
            weights,
            observed_codes,
            item_count=4,
            positive_batch_size=2,
            negatives_per_positive=2,
            rng=np.random.default_rng(42),
        )
    )
    batch_users, batch_items, labels, batch_weights = batch
    negative_codes = batch_users[labels == 0] * 4 + batch_items[labels == 0]

    assert labels.sum() == 2
    assert len(labels) == 6
    assert not np.isin(negative_codes, observed_codes).any()
    assert sorted(batch_weights[labels == 1]) == pytest.approx([0.75, 1.25])
