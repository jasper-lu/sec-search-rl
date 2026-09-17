from sec_rl.environment import RetrievalDataset


def _dataset(n_builders: int, groups: int, epochs: int) -> RetrievalDataset:
    return RetrievalDataset(list(range(n_builders)), groups, epochs=epochs, seed=7)  # type: ignore[arg-type]


def test_one_epoch_is_the_plain_batch_order() -> None:
    ds = _dataset(10, 4, epochs=1)
    assert len(ds) == 3
    assert list(ds.get_batch(0)) == [0, 1, 2, 3]
    assert list(ds.get_batch(2)) == [8, 9]


def test_second_epoch_reshuffles_and_covers_every_query() -> None:
    ds = _dataset(16, 4, epochs=2)
    assert len(ds) == 8
    epoch1 = [b for i in range(4) for b in ds.get_batch(i)]
    epoch2 = [b for i in range(4, 8) for b in ds.get_batch(i)]
    assert epoch1 == list(range(16))
    assert sorted(epoch2) == list(range(16))
    assert epoch2 != epoch1


def test_epoch_order_is_deterministic() -> None:
    a = _dataset(16, 4, epochs=2)
    b = _dataset(16, 4, epochs=2)
    assert [list(a.get_batch(i)) for i in range(8)] == [list(b.get_batch(i)) for i in range(8)]
