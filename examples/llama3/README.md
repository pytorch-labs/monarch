Run with ssh proxy if on devserver

    cd examples/llama3 && python data/shakespeare/prepare.py

Run llama3 locally
    cd .. (to examples folder)
    python -m llama3.train ./llama3/configs/llama8b_local.py

## Install TransformerEngine (--use_te=True) in conda env
monarch_conda env is the simplest conda env to use. Follow https://fburl.com/code/8q2rkyvg

## Train locally on xlformers data loaded from warm storage

First, determine which warm storage cluster to load data from for your devgpu region. Go into ./xlformers/core/params/cluster.py
and look at `_LARGE_EXPERIMENT_CLUSTER_MAP`. Find your devgpu region and note that the associated path starts with /home/{USER}/{CLUSTER}-wsf.

Now, set up [OILFS](https://www.internalfb.com/intern/wiki/GenAI_Storage/Storage_Services/OILFS/). On your devgpu:
```
mkdir ~/{CLUSTER}-wsf
feature install oilfs
```

If `CLUSTER == pci`, then run:
```
oilfs ws://ws.ai.pci0ai/checkpoint/infra ~/pci-wsf
```

Alternatively, if `CLUSTER == eag`, then run:
```
oilfs ws://ws.ai.eag0genai/checkpoint/infra ~/eag-wsf
```

If that succeeded, then running `ls ~/{CLUSTER}-wsf` should show something like:
```
aidev  dajang  david_test  fair_llm_v2  fair_llm_v3  finetune  genai_media  lakshyagarg  nextgen_mm  outputs  tgeorgiou_test  tokenizers  wenyinfu
```

You also need to create the directory /tmp/xlformers/{USER}/xldumps, because xlformers expects this directory to exist.

You can then run training with either buck or python. You will likely need to install additional packages in your conda environment if running
with python. Although xlformers has prepackaged conda environments distributed via fbpkg like `xlformers_multimodal_conda_unified`, the torch
version in that environment doesn't work with A100 gpus.

To run with buck:
```
buck2 run @//mode/opt //monarch/examples/llama3:train -- /path/to/llama3/configs/xlformers.py
```

To run with python, from inside the fbcode/monarch/examples directory:
```
python llama3/train.py llama3/configs/xlformers.py
```

To run on MAST on N hosts, first install monarch via `python setup.py install` (as opposed to `develop`). Then, from inside the fbcode/monarch/examples directory:
```
torchx run --scheduler_args="localityConstraints=region;eag,rmAttribution=gen_ai_rf_nextgen_infra" mast.py:train --nodes=N --script llama3/train_mast.py -- [args...]
```

It may be necessary to increase the heartbeat timeout (the amount of time before supervisor considers a host dead) when running on MAST. You can do this by adding `--env "TORCH_SUPERVISOR_HEARTBEAT_LIVENESS=15.0"` after `mast.py:train`.

## launching training on mast

```
torchx run --scheduler_args="localityConstraints=region;eag" mast.py:train --script llama3/train.py -- <script_args>
```
The above command should be run with the conda env currently active in the monarch/examples directory and will package up the conda env that is active and launch training. If you have a prebuilt conda env then adding
`conda_fbpkg_id=<fbpkg_id>` will use it directly. We will have a standard env setup for use soon so this will be a no-op. The locality constraint will control which region training is launched in and `-h` option can be added to change the hardware type. Current default is a H100.

You can use the following commands in that directory for more info:
- torchx run --help
- torchx runopts mast_conda
- torchx run mast.py:train --help

Note: You will need to be part of the permission group [infra_genai_testing](https://www.internalfb.com/amp/group/infra_genai_testing) for warm storage to work. Please
check and request permission if not part of it. The monarch oncall should already have been added.
