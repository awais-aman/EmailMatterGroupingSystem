import pytest

from matter_grouping import generate_emails


@pytest.fixture(scope="session")
def emails():
    return generate_emails(count=25, replies=3, seed=7)
