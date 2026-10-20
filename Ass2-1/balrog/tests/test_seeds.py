from balrog.utils import get_episode_seed


def test_episode_seed_is_reproducible_and_task_specific():
    first = get_episode_seed(42, "babyai", "goto", 0)

    assert first == get_episode_seed(42, "babyai", "goto", 0)
    assert first != get_episode_seed(42, "babyai", "goto", 1)
    assert first != get_episode_seed(42, "babyai", "pickup", 0)
    assert first != get_episode_seed(42, "crafter", "goto", 0)
    assert 0 <= first < 2**32
