import pytest

from nexus.config import Settings


@pytest.mark.parametrize("token", ["", "short", "x" * 32 + " ", "x" * 32 + "\n", "é" * 32])
def test_invalid_tokens_fail_configuration(token):
    with pytest.raises(ValueError):
        Settings(api_token=token)


def test_secret_is_not_in_configuration_repr():
    token = "secret-" + "x" * 32
    assert token not in repr(Settings(api_token=token))


@pytest.mark.parametrize("reviewer", ["", "with spaces", "é", "x" * 65])
def test_invalid_reviewer_configuration(reviewer):
    with pytest.raises(ValueError):
        Settings(reviewer_id=reviewer)
