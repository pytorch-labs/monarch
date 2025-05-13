Install
=======

Install torchtitan

    git clone https://github.com/pytorch/torchtitan
    cd torchtitan
    # TODO: update to current API
    git checkout 82f7387f8c7af6b4285869f5daf2f52e523a2774
    pip install .

Install morpho:

    cd fbsource/fbcode/monarch/examples/morpho
    pip install -e .


Use
===

Run an example debug training run:

    # 'morpho.train.train' is the name of the python function that will
    # be invoked, use --help to get full arguments
    morpho morpho.train.train --machine.ngpu 2

Compare morpho's values to torchtitan for correctness:

    morpho morpho.debug.sweep.local_debug_sweep --training.steps 3  output.png --machine.ngpu 2

`output.png` will contain a plot of the numeric differences at each measurement.
