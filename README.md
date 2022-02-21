# SimulEval
[![](https://github.com/facebookresearch/SimulEval/workflows/build/badge.svg)](https://github.com/facebookresearch/SimulEval/actions)

SimulEval is a general evaluation framework for simultaneous translation on text and speech.

### Requirement
* python >= 3.7.0

## Installation
```
git clone https://github.com/facebookresearch/SimulEval.git
cd SimulEval
pip install -e .
```

## Quick Start
Following is the evaluation of a [dummy agent](examples/dummy/dummy_waitk_text_agent.py) which operates wait-k (k = 3) policy and generates random words until the length of the generated words is the same as the number of all the source words. A tutorial can be found [here](docs/get_started.md).
```shell
cd examples
simuleval \
  --agent dummy/dummy_waitk_text_agent.py \
  --source data/src.txt \
  --target data/tgt.txt
```

# License

SimulEval is licensed under Creative Commons BY-SA 4.0.

# Citation

Please cite as:

```bibtex
@inproceedings{simuleval2020,
  title = {Simuleval: An evaluation toolkit for simultaneous translation},
  author = {Xutai Ma, Mohammad Javad Dousti, Changhan Wang, Jiatao Gu, Juan Pino},
  booktitle = {Proceedings of the EMNLP},
  year = {2020},
}
```
