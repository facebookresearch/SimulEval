simuleval \
    --agent whisper_waitk.py \
    --source-segment-size 500 \
    --waitk-lagging 3 \
    --source source.txt --target reference/transcript.txt \
    --output output --quality-metrics WER
