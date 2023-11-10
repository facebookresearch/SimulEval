# TIMESTAMP=$(date +"%Y%m%d%H%M%S")
# OUTPUT_FILE="results_$TIMESTAMP.txt"

# simuleval \
#     --agent counter_in_tgt_lang_agent.py \
#     --source-segment-size 1000 \
#     --source source.txt --target reference/en.txt \
#     --tgt-lang reference/tgt_lang.txt \
#     --output output | tee results/$OUTPUT_FILE


#!/bin/bash

# for ((i = 1; i <= 200; i++)); do
#     TIMESTAMP=$(date +"%Y%m%d%H%M%S")
#     OUTPUT_FILE="results_$TIMESTAMP.txt"

#     echo "Running SimulEval iteration $i"
#     simuleval \
#         --agent counter_in_tgt_lang_agent.py \
#         --source-segment-size 1000 \
#         --source source.txt --target reference/en.txt \
#         --tgt-lang reference/tgt_lang.txt \
#         --output output | tee results/$OUTPUT_FILE

#     echo "Iteration $i completed. Output saved to $OUTPUT_FILE"
# done

TIMESTAMP=$(date +"%Y%m%d%H%M%S")
OUTPUT_FILE="results_$TIMESTAMP.txt"
simuleval \
    --agent whisper_waitk.py \
    --source-segment-size 500 \
    --waitk-lagging 3 \
    --source source.txt --target reference/transcript.txt \
    --output output --quality-metrics WER | tee results/$OUTPUT_FILE