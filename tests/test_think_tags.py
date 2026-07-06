"""Tests for think-tag state machine."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.utils import strip_think_tags


def test_no_think_tag():
    text = "This is a normal response."
    out, state = strip_think_tags(text, False)
    assert out == text
    assert state is False


def test_simple_think_block():
    text = "Hello <think>internal reasoning</think> world"
    out, state = strip_think_tags(text, False)
    assert out == "Hello  world"
    assert state is False


def test_unclosed_think():
    text = "Hello <think>still thinking"
    out, state = strip_think_tags(text, False)
    assert out == "Hello "
    assert state is True


def test_close_in_second_chunk():
    out1, state = strip_think_tags("Hello <think>thinking", False)
    assert out1 == "Hello "
    assert state is True
    out2, state = strip_think_tags(" still thinking</think> world", state)
    assert out2 == " world"
    assert state is False


def test_multiple_think_blocks():
    text = "A<think>hidden</think>B<think>hidden2</think>C"
    out, state = strip_think_tags(text, False)
    assert out == "ABC"
    assert state is False


def test_empty_content():
    out, state = strip_think_tags("", False)
    assert out == ""
    assert state is False
