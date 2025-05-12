import pickle

from morpho.config import DebugConfig, MachineConfig, TrainingConfig
from morpho.titan import train


def test_titan():
    debug = train(machine=MachineConfig(ngpu=1), training=TrainingConfig(steps=3))
    assert debug is None


def test_titan_debug():
    debug = train(
        machine=MachineConfig(ngpu=1),
        training=TrainingConfig(steps=3),
        debug=DebugConfig(variant="noise", observe=True),
    )
    with open(debug, "rb") as f:
        pickle.load(f)


def test_titan_one():
    debug = train(machine=MachineConfig(ngpu=1), training=TrainingConfig(steps=3))
    assert debug is None
