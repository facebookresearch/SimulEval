simuleval \
    --agent counter_in_tgt_lang_agent.py \
    --source-segment-size 1000 \
    --source source.txt --target reference/en.txt \
    --tgt-lang tgt-lang.txt \
    --output output 


# default="en", type=str, choices=["en", "es", "de"]
    # --tgt_lang tgt-lang.txt \
