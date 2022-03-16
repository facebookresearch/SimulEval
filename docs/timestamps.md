# Evaluate Text-to-Text Systems with Source Time Stamps

In real world applications, the inputs of a text-to-text simultaneous translation can come from an upstream automatic speech recognition (ASR) system.
Therefore, SimulEval provides a feature that computes latency metrics in time domain when given the time stamps of source input tokens.

For instance, a text system can be evaluated as follow:
```
simuleval \
	--agent examples/dummy_waitk_text_agent.py \
	--waitk 5 \
	--source examples/data/src.txt \
	--target examples/data/tgt.txt
```
In order to compute latency in time domain, an additional file contains time information (in this example, `examples/data/src_time.txt`) is needed
```
simuleval \
	--agent examples/dummy/dummy_waitk_text_agent.py \
	--waitk 5 \
	--source examples/data/src.txt \
	--target examples/data/tgt.txt \
	--source-timestamps examples/data/src_time.txt
```
The results will contain latency in time domain, the unit will be milliseconds.
```
{
    "Quality": {
        "BLEU": 0.06934086086827945
    },
    "Latency": {
        "AL": 5.0,
        "AL (Time in ms)": 1656.442724609375,
        "AP": 0.7560754895210267,
        "AP (Time in ms)": 0.7404875814914703,
        "DAL": 5.0,
        "DAL (Time in ms)": 2462.1722412109375
    }
}
```

The `--source-timestamps` option require a file, in which each line contains the time information for the corresponding sentence in milliseconds. For example, for certain line in `--source` is
```
I'm going to talk today about energy and climate.
```
The correspond line in `--source-timestamps` should look like
```
781 805 813 1194 1844 2370 3133 3291 3787
```
The time stamps is in milliseconds and separated by the space. Notice that the time stamps should have the same length of source input tokens.