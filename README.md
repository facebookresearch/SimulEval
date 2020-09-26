# SimulEval
[![](https://github.com/fairinternal/SimulEval/workflows/build/badge.svg)](https://github.com/fairinternal/SimulEval/actions)

SimulEval is a general evaluation framework for simultanteuous translation on text and speech.

## Installation
(Will modify after public)
### Requirement
* python >= 3.7.0
```
git clone git@github.com:fairinternal/SimulEval.git
cd SimulEval
pip install -e .
```

## Quick Start
Following is the evaluation of a [dummy agent](examples/dummy/dummy_waitk_text_agent.py) which operates wait-k (k = 3) policy and generates random words until the length of the generated words is the same as the number of all the source words.
```shell
cd examples
simuleval \
  --agent dummy/dummy_waitk_agent.py \
  --source data/src.txt \
  --target data/tgt.txt \
```
(TODO) More tutorial coming.

# License

SimulEval is licensed under Creative Commons BY-SA 4.0.
