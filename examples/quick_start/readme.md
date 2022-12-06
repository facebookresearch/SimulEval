# Quick Start

To evaluate a text-to-text wait-3 system with random output

```
> simuleval --source source.txt --reference target.txt --agent dummy_waitk_text_agent_v1.py

2022-12-05 13:43:58 | INFO | simuleval.cli | Evaluate system: DummyWaitkTextAgent
2022-12-05 13:43:58 | INFO | simuleval.dataloader | Evaluating from text to text.
2022-12-05 13:43:58 | INFO | simuleval.sentence_level_evaluator | Results:
BLEU  AL    AP  DAL
1.541 3.0 0.688  3.0

```



More examples:
- `dummy_waitk_text_agent_v2.py`: customized agent argument
- `dummy_waitk_text_agent_v3.py`: agent pipeline

More details can be found [here](https://simuleval.readthedocs.io/en/v1.1.0/quick_start.html)
