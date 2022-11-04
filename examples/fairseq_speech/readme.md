# Fairseq Simultaneous Speech Translation Example
## To-Do List
- [X] Debug low quality on speech-to-speech model.
- [X] Use Hirofumi's checkpoint of offline xm_transformer.
- [ ] Use public version of asr-bleu from Iliad.

## Test Time Wait-K Speech-to-Text Translation
This example illustrate how to decode an offline speech-to-text model with test time [wait-k](https://aclanthology.org/P19-1289/) policy.

Dependencies required:
- `sentencepiece`: `pip install sentencepiece`

First of all, prepare an offline model with preprocessed data.
```bash
checkpoint="/private/home/hirofumii/large_experiments/checkpoints/s2ut_pt/es_en/s2t/s2t.pt_es_en.config_mbart.asr.rdrop10.0.ls0.2.maxtok2.0k.uf30.lr0.0005.wu1k.seed1.arch_xm_transformer.W2V_cfm_L.dr0.1.ld0.2.al1.dld0.2.mBART_spm.LND.ca.ngpu24/avg_best_10_checkpoint.pt"
data="/large_experiments/ust/hirofumii/datasets/s2ut_pt/es_en/s2t"
config="config_mbart.yaml"
subset="dev_mtedx_filt" # or subset=dev_epst
```

Then calling the following command.
```bash
agent="${simuleval_dir}/examples/fairseq_speech/fairseq_test_waitk_s2t_agent.py"

# step size means we run the policy every this number of encoder states
# each encoder state of the offline model we use has a span of 120ms
# Therefore we set the segment_size to 8 * 40 = 320ms
step=8
segment_size=`python -c "print(int(${step} * 40))"`

# The K value in waitk policy
waitk=4

# Kick off simuleval evaluation
# If you would like to make sure everything runs okay,
# use --end-index 10 to just evaluate the first 10 sentences.
# The output will be in the ${output_dir}
simuleval \
    --fairseq-data ${data} \
    --fairseq-config ${config} \
    --fairseq-gen-subset ${subset}\
    --checkpoint ${checkpoint} \
    --agent ${agent_file} \
    --output ${output_dir} \
    --device cuda:0 \
    --source-segment-size ${segment_size} \
    --max-len-a 0.125 \
    --max-len-b 10 \
    --waitk-lagging ${k} \
    --fixed-predicision-ratio ${step}
```
The output will be in the `${output_dir}`. There will be two files in output directory `instances.log` and `scores`. `instances.log` contains the details of decoding process for each sentence, in which each line is a json string. `scores` contains the final evaluation results. On EuraParl v1.0 dev set, the results under this setting is shown as follow. The suffix "_CA" indicates "computation aware", which we consider the computation time while calculating the latency.
```bash
cat ${output_dir}
{
    "Quality": {
        "BLEU": 35.08379067677246
    },
    "Latency": {
        "AL": 1850.5411219580699,
        "AL_CA": 2801.504919812908,
        "AP": 0.8234371600851501,
        "AP_CA": 1.1013422159825341,
        "DAL": 2195.3421246717207,
        "DAL_CA": 3330.618655136884
    }
}
```
## Test example for the transducer based speech to text translation
```bash
checkpoint="/private/home/yuntang/2022/streaming/tt_es_en_shared_code_fix_cfg/checkpoints/st.wal.freeze_3k.nk_1k_safe.optim_radam.lr_0.0003.cn_10.0.wu_12k.mt_10k.up_2.seed_12.arch_s2ttt_large_full.bsz_20.ngpu16/checkpoint_ave10.pt"
data="/private/home/yuntang/2022/streaming/data/es_en/v01/s2t/"
gcvmn="/private/home/yuntang/2022/streaming/data/es_en/v01/s2t/gcmn.npz"
config="config_1k_simuleval.yaml"
subset="dev_mtedx_filt"

agent="${simuleval_dir}/examples/fairseq_speech/fairseq_test_transducer_s2t_agent.py"
step=4
segment_size=`python -c "print(int(${step} * 40))"`

simuleval \
    --fairseq-data ${data} \
    --fairseq-config ${config} \
    --fairseq-gen-subset ${subset}\
    --checkpoint ${checkpoint} \
    --agent ${agent} \
    --output ${output_dir} \
    --device cuda:0 \
    --init-target-token "<s>" \
    --user-dir "$fairseq_root/examples/transformer_transducer" \
    --blankpen 1.6 \
    --global-cmvn $gcvmn \
    --source-segment-size ${segment_size}

```

The results are
```bash
{
    "Quality": {
        "BLEU": 23.85482604044558
    },
    "Latency": {
        "AL": 1245.6334870713097,
        "AL_CA": 1829.1715134959068,
        "AP": 0.7137666339835813,
        "AP_CA": 0.8493919421313545,
        "DAL": 1644.104458716608,
        "DAL_CA": 2216.199846153435
    }
}
```

## Test Time Wait-K Cascaded Speech-to-Speech Translation
First of all, install the following dependencies:
- `pip install huggingface_hub`
- `pip install g2p_en`
- `pip install git+ssh://git@github.com/fairinternal/ust_common.git`
- Montreal Force Aligner (`mfa`). Please see [here](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html) for installation.
Also make sure download `mfa` English dictionary and acoustic models.
```bash
mfa model download dictionary english_mfa
mfa model download acoustic english_mfa
```

The inference command for s2st system is
```bash
# Hyperparameters similar to the s2t inference
step=8
segment_size=`python -c "print(int(${step} * 40))"`
waitk=4

# Minimal number of phonemes everytime feed into the TTS module.
# Larger than 4 gives reasonable BLEU score.
min_ph=6

agent="${simuleval_dir}/examples/fairseq_speech/fairseq_test_waitk_s2s_tts_agent.py"

simuleval \
    --agent ${agent} \
    --fairseq-data ${data} \
    --fairseq-config ${config} \
    --fairseq-gen-subset ${subset}\
    --checkpoint ${checkpoint} \
    --output ${output} \
    --device cuda:0 \
    --source-segment-size ${segment_size}  \
    --waitk-lagging ${k} \
    --init-target-token "[en_XX]" \
    --fixed-predicision-ratio ${step} \
    --num-emit-phoneme ${min_ph} \
    --max-len-a 0.125 \
    --max-len-b 10
```
The first time the agent is run, a fastspeech2 TTS system will be downloaded.
The evaluation of speech-to-speech is more complicated than speech-to-text. Here are how things are done (The SimulEval will do everything for you)
- Quality
    - Run an ASR system on the speech output to get transcripts (TODO: use asr-bleu from Ilia).
    - Run sacrebleu on transcripts and reference.
- Latency
    - Run Montreal Force Aligner (MFA) to generate alignment between speech output and transcripts.
    - Use alignment to determine the time stamp of each word.
    - Compute the latency based on the time stamps.

In the final `${output_dir}`, there are following items
- `instances.log`: decoding details, similar to speech-to-text.
- `scores`: final results, similar to speech-to-text.
- `wavs`: generated wav files and transcript, following the pattern of `*_pred.wav` and `*_pred.txt`.
- `asr_prep_data`: prepared data for the ASR system.
- `asr_out`: transcripts from the ASR system
- `align`: the alignment between transcripts and speech out, following the pattern `*_pred.TextGrid`

The final scores of a speech-to-speech system looks like following. Only NCA (non-CA) latency is supported now.
`BOW`, `EOW`, `COW` indicates the time stamps that were used to compute the latency.
- `BOW`: beginning of the word
- `EOW`: end of the word
- `AOW`: center of the word (average of `BOW` and `EOW`)

```bash
{
    "Quality": {
        "BLEU": 20.176040705901404
    },
    "Latency": {
        "BOW": {
            "AL": 3010.4543390823624,
            "AP": 1.2425067150323403,
            "DAL": 3606.5989132293876
        },
        "EOW": {
            "AL": 3246.2921475340418,
            "AP": 1.362453529390238,
            "DAL": 3890.952848376258
        },
        "COW": {
            "AL": 3128.9276801675055,
            "AP": 1.3024801236087993,
            "DAL": 3734.43338009247
        }
    }
}
```
