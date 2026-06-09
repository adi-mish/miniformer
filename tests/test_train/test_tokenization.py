import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "src"))

import pytest

from miniformer.utils.tokenization import stable_token_id


def test_stable_token_id_is_reproducible():
    assert stable_token_id("hello", 1000) == stable_token_id("hello", 1000)
    assert 0 <= stable_token_id("hello", 1000) < 1000


def test_stable_token_id_rejects_invalid_vocab_size():
    with pytest.raises(ValueError, match="vocab_size"):
        stable_token_id("hello", 0)
