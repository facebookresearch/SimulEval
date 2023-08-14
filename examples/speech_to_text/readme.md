## Simultaneous Speech-to-Text Translation

This tutorial provides a minimal example on how to evaluate a simultaneous speech-to-text translation system.

### Agent

The speech-to-text agent ([english_counter_agent.py](english_counter_agent.py)) in this example is a counter, which generates number of seconds in text, after waiting for `self.wait_seconds` seconds. The policy finishes when the source is finished.

```python
def policy(self):
    length_in_seconds = round(
        len(self.states.source) / self.states.source_sample_rate
    )
    if not self.states.source_finished and length_in_seconds < self.wait_seconds:
        return ReadAction()

    prediction = f"{length_in_seconds} second"

    return WriteAction(
        content=prediction,
        finished=self.states.source_finished,
    )
```

### Evaluation

The following command will start an evaluation

```bash
simuleval \
    --agent english_counter_agent.py \
    --source-segment-size 1000 \
    --source source.txt --target reference/en.txt \
    --output output
```

The results of the evaluation should be as following. The detailed results can be found in the `output` directory.

```
  BLEU     LAAL       AL     AP       DAL       ATD
 100.0  822.018  822.018  0.581  1061.271  2028.555
```

### Example Streaming ASR / S2T: Whipser Wait-K model

This section provide a more realistic model. [whisper_waitk.py](whisper_waitk.py) is a streaming ASR agent running [wait-k](https://aclanthology.org/P19-1289/) policy on the [Whisper](https://github.com/openai/whisper) ASR model

```bash
simuleval \
    --agent whisper_waitk.py \
    --source-segment-size 500 \
    --waitk-lagging 3 \
    --source source.txt --target reference/transcript.txt \
    --output output --quality-metrics WER
```

The results of the evaluation should be as following. The detailed results can be found in the `output` directory.

```
WER     LAAL    AL      AP      DAL     ATD
25.0    2353.772        2353.772        0.721   2491.04 2457.847
```

This agent can also perform S2T task, by adding `--task translate`.
