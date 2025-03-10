from unittest.mock import patch
import pytest
from config.config import Config
from utils.dataset_actor import DatasetActor


@pytest.fixture
def conf_init():
    Config.init("tests/config.yaml")


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


def test_get_next_function_random_same_seed(dataset_actor):
    first_function = dataset_actor.get_next_function()
    assert first_function is not None

    second_function = dataset_actor.get_next_function()
    assert second_function is not None

    third_function = dataset_actor.get_next_function()
    assert third_function is not None

    sequence = [first_function[0], second_function[0], third_function[0]]

    dataset_actor = DatasetActor(Config.config.dataset)
    first_function_again = dataset_actor.get_next_function()
    second_function_again = dataset_actor.get_next_function()
    third_function_again = dataset_actor.get_next_function()

    assert sequence == [
        first_function_again[0],
        second_function_again[0],
        third_function_again[0],
    ]


def test_get_next_function_random_different_seed(dataset_actor):
    first_function = dataset_actor.get_next_function()
    assert first_function is not None

    second_function = dataset_actor.get_next_function()
    assert second_function is not None

    third_function = dataset_actor.get_next_function()
    assert third_function is not None

    sequence = [first_function[0], second_function[0], third_function[0]]

    Config.config.dataset.seed = 42
    dataset_actor = DatasetActor(Config.config.dataset)
    first_function_again = dataset_actor.get_next_function()
    second_function_again = dataset_actor.get_next_function()
    third_function_again = dataset_actor.get_next_function()

    assert sequence != [
        first_function_again[0],
        second_function_again[0],
        third_function_again[0],
    ]

    Config.config.dataset.seed = None
    dataset_actor = DatasetActor(Config.config.dataset)
    first_function_again = dataset_actor.get_next_function()
    second_function_again = dataset_actor.get_next_function()
    third_function_again = dataset_actor.get_next_function()

    assert sequence != [
        first_function_again[0],
        second_function_again[0],
        third_function_again[0],
    ]


def test_get_next_function_not_random(conf_init):
    Config.config.dataset.shuffle = False
    dataset_actor = DatasetActor(Config.config.dataset)
    first_function = dataset_actor.get_next_function()
    assert first_function is not None

    second_function = dataset_actor.get_next_function()
    assert second_function is not None

    third_function = dataset_actor.get_next_function()
    assert third_function is not None

    sequence = [first_function[0], second_function[0], third_function[0]]

    dataset_actor = DatasetActor(Config.config.dataset)
    first_function_again = dataset_actor.get_next_function()
    second_function_again = dataset_actor.get_next_function()
    third_function_again = dataset_actor.get_next_function()

    assert sequence == [
        first_function_again[0],
        second_function_again[0],
        third_function_again[0],
    ]
    Config.config.dataset.seed = 13
    dataset_actor = DatasetActor(Config.config.dataset)
    first_function_again = dataset_actor.get_next_function()
    second_function_again = dataset_actor.get_next_function()
    third_function_again = dataset_actor.get_next_function()

    assert sequence == [
        first_function_again[0],
        second_function_again[0],
        third_function_again[0],
    ]


def test_get_next_function_random(conf_init):
    dataset_actor = DatasetActor(Config.config.dataset)
    first_function = dataset_actor.get_next_function()
    assert first_function is not None

    second_function = dataset_actor.get_next_function()
    assert second_function is not None

    third_function = dataset_actor.get_next_function()
    assert third_function is not None

    sequence = [first_function[0], second_function[0], third_function[0]]

    dataset_actor = DatasetActor(Config.config.dataset)
    first_function_again = dataset_actor.get_next_function(random=True)
    second_function_again = dataset_actor.get_next_function(random=True)
    third_function_again = dataset_actor.get_next_function(random=True)

    assert sequence != [
        first_function_again[0],
        second_function_again[0],
        third_function_again[0],
    ]


def test_dataset_filtered_by_tags(conf_init):
    tag = "icml"
    Config.config.dataset.tags = [tag]
    dataset_actor = DatasetActor(Config.config.dataset)

    assert (
        dataset_actor.dataset_size == 8
    ), f"Expected 8, got {dataset_actor.dataset_size}"

    assert all(
        tag in dataset_actor.dataset[function]["tags"]
        for function in dataset_actor.function_names
    )


def test_tags(conf_init, dataset_actor):
    tag = "full"
    all_same_tag = True
    for i in range(dataset_actor.dataset_size):
        _, function_data, _ = dataset_actor.get_next_function()
        if tag not in function_data.tags:
            all_same_tag = False
            break

    assert not all_same_tag


def test_save_dataset(dataset_actor):
    with (
        patch("pickle.dump") as pickle_dump,
        patch("pathlib.Path.open") as path_open,
    ):
        dataset_actor.save_dataset_to_disk()
        path_open.assert_called_once_with("wb")
        pickle_dump.assert_called_once()


def test_update_dataset(dataset_actor):
    function_name, function_cache, _ = dataset_actor.get_next_function()
    with patch(
        "utils.dataset_actor.DatasetActor.save_dataset_to_disk"
    ) as save_dataset_to_disk:
        dataset_actor.update_dataset(function_name, function_cache.to_dict())
        save_dataset_to_disk.assert_not_called()
        dataset_actor.nbr_updates = dataset_actor.saving_frequency - 1
        dataset_actor.update_dataset(function_name, function_cache.to_dict())
        save_dataset_to_disk.assert_called_once_with(version=2)
        dataset_actor.nbr_updates = 2 * dataset_actor.saving_frequency - 1
        dataset_actor.update_dataset(function_name, function_cache.to_dict())
        save_dataset_to_disk.call_count == 2
        save_dataset_to_disk.call_args == {"version": 1}
