from sre.datasets import load_housing_data


def test_dataset_not_is_empty():
    assert False == load_housing_data().empty
