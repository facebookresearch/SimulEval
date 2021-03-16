# Evaluation

This document introduces how SimulEval runs the evaluation.

## Overview
SimulEval will run evaluation on two measurements: **Latency** and **Quality**.
While latency is a sentence level measurement,
quality (BLEU) should be calculated from the whole evaluation set.
The detailed information for evaluation, including the delays and latency metrics will be shown in `instances.log` in `--output` directory if `--output` is set.

## Quality
The quality is evaluated with BLEU score by [SacreBLEU](https://github.com/mjpost/sacrebleu).
The quality evaluation will be run after all the translations finish.
By default, the tokenizer for SacreBLEU is `13a`,
but you can choose other tokenizers by setting `--sacrebleu-tokenizer`.
Make sure you install all the dependencies for non-default tokenizers.
When evaluating the languages with no spaces between the words, such as Chinese, Japanese, Korean etc., `--no-space` options should be used.
`--no-space` will not introduce spaces when joining individual predicted words.

## Latency
SimulEval provides three metrics for latencies evaluation, along with their computation-aware versions (considering runtime for delays, useful for speech translation; more details can be found [here](https://www.aclweb.org/anthology/2020.aacl-main.58/))
- [Average Lagging (Ma et al., 2019)](https://www.aclweb.org/anthology/P19-1289.pdf)
- [Differentiable Average Lagging (Cherry and Foster, 2019)](https://arxiv.org/abs/1906.00048)
- [Average Proportion (Cho and Esipova, 2016)](https://arxiv.org/abs/1606.02012)

During the evaluation process,
sentence level latency is calculated when one instance (a text sentence or a utterance of speech) is fully translated.
The corpus level latency is the average of each sentence level latency.

By default, the latency is evaluated on the detokenized words splitted by spaces.
However, when evaluating languages with no spaces between words, we probably want to evaluate latency based on characters, which can be achieved by using `--eval-latency-unit char`.