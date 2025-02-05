import pytest
from config.config import Config
from utils.dataset_actor.dataset_actor import DatasetActor


@pytest.fixture
def conf_init():
    Config.init("config/config.yaml")


@pytest.fixture
def dataset_actor(conf_init):
    dataset_actor = DatasetActor(Config.config.dataset)
    return dataset_actor


def test_get_next_function(dataset_actor):
    first_function = dataset_actor.get_next_function()
    assert first_function is not None

    second_function = dataset_actor.get_next_function()
    assert second_function is not None

    third_function = dataset_actor.get_next_function()
    assert third_function is not None

    dataset_actor = DatasetActor(Config.config.dataset)
    first_function_again = dataset_actor.get_next_function()
    assert first_function_again is not None and first_function_again == first_function

    second_function_again = dataset_actor.get_next_function()
    assert (
        second_function_again is not None and second_function_again == second_function
    )

    # random
    third_function_random = dataset_actor.get_next_function(True)
    assert third_function_random is not None and third_function_random != third_function


def test_tags(conf_init, dataset_actor):
    tag = "10k"
    all_same_tag = True
    for i in range(10):
        _, function_data, _ = dataset_actor.get_next_function()
        if tag not in function_data.tags:
            all_same_tag = False
            break

    assert not all_same_tag

    Config.config.dataset.tags = [tag]
    dataset_actor = DatasetActor(Config.config.dataset)
    for i in range(10):
        _, function_data, _ = dataset_actor.get_next_function()
        assert tag in function_data.tags
