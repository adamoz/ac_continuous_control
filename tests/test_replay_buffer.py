import pytest
from ac_continuous_control.replay_buffer import ReplayBuffer


def test_check_ready_for_samplig():
    replay_buffer = ReplayBuffer(buffer_size=10, batch_size=5)
    for value in range(4):
        value = [value]
        replay_buffer.add(value, value, value, value, value)
    assert replay_buffer.is_ready_to_sample() is False

    for value in range(4):
        value = [value]
        replay_buffer.add(value, value, value, value, value)
    assert replay_buffer.is_ready_to_sample() is True
