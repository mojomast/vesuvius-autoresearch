import pytest

from src.autoresearch.villa_samplers import GroupStratifiedBatchSampler, StatefulShuffledSampler


# Samplers adapted from ScrollPrize/villa: https://github.com/ScrollPrize/villa/blob/main/ink-detection/samplers.py


def test_stateful_shuffled_sampler_cursor_persists_without_repeats() -> None:
    sampler = StatefulShuffledSampler(5, seed=123)
    iterator = iter(sampler)
    first = [next(iterator) for _ in range(3)]
    second_iterator = iter(sampler)
    second = [next(second_iterator) for _ in range(2)]
    assert len(set(first + second)) == 5


def test_stateful_shuffled_sampler_is_deterministic() -> None:
    left = list(StatefulShuffledSampler(8, seed=7))
    right = list(StatefulShuffledSampler(8, seed=7))
    assert left == right


def test_group_stratified_batch_sampler_represents_all_groups() -> None:
    groups = [0, 0, 0, 0, 1, 1, 1, 1]
    sampler = GroupStratifiedBatchSampler(groups, batch_size=4, seed=3)
    batch = next(iter(sampler))
    assert {groups[index] for index in batch} == {0, 1}
    assert len(batch) == 4


def test_group_stratified_batch_sampler_validates_batch_size() -> None:
    with pytest.raises(ValueError, match="divisible"):
        GroupStratifiedBatchSampler([0, 1, 2, 0, 1, 2], batch_size=4)
