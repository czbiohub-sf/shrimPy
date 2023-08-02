import pytest
import yaml


@pytest.fixture(scope="function")
def demo_settings():
    with open("./mantis/acquisition/settings/demo_acquisition_settings.yaml") as file:
        settings = yaml.safe_load(file)
    yield settings


@pytest.fixture(scope="function")
def example_settings():
    with open("./mantis/acquisition/settings/example_acquisition_settings.yaml") as file:
        settings = yaml.safe_load(file)
    yield settings
