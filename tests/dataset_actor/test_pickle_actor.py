import pytest
import ray
from config.config import Config
from utils.dataset_actor.dataset_actor import DatasetActor


@pytest.fixture
def conf_init():
    Config.init("config/config.yaml")


@pytest.fixture
def dataset_actor(conf_init):
    dataset_actor = DatasetActor.remote(Config.config.dataset)
    return dataset_actor


def test_get_next_function(dataset_actor):
    first_function = ray.get(dataset_actor.get_next_function.remote())
    assert first_function is not None

    second_function = ray.get(dataset_actor.get_next_function.remote())
    assert second_function is not None

    third_function = ray.get(dataset_actor.get_next_function.remote())
    assert third_function is not None

    dataset_actor = DatasetActor.remote(Config.config.dataset)
    first_function_again = ray.get(dataset_actor.get_next_function.remote())
    assert first_function_again is not None and first_function_again == first_function

    second_function_again = ray.get(dataset_actor.get_next_function.remote())
    assert (
        second_function_again is not None and second_function_again == second_function
    )

    # random
    third_function_random = ray.get(dataset_actor.get_next_function.remote(True))
    assert third_function_random is not None and third_function_random != third_function


def test_tags(conf_init, dataset_actor):
    tag = "10k"
    all_same_tag = True
    for i in range(10):
        _, function_data, _ = ray.get(dataset_actor.get_next_function.remote())
        if tag not in function_data["tags"]:
            all_same_tag = False
            break

    assert not all_same_tag

    Config.config.dataset.tags = [tag]
    dataset_actor = DatasetActor.remote(Config.config.dataset)
    for i in range(10):
        _, function_data, _ = ray.get(dataset_actor.get_next_function.remote())
        assert tag in function_data["tags"]
