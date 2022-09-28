# Fairseq Simultaneous Speech Translation Example
## To-Do List
- [ ] Debug low quality on speech-to-speech model.
- [ ] Use Hirofumi's checkpoint of offline xm_transformer.
- [ ] Use public version of asr-bleu from Iliad.

## Test Time Wait-K Speech-to-Text Translation
This example illustrate how to decode an offline speech-to-text model with test time [wait-k](https://aclanthology.org/P19-1289/) policy.

First of all, prepare an offline model with preprocessed data.
```bash
checkpoint="/large_experiments/ust/xutaima/expal/509/509-fairseq_train..ngpu64/checkpoint_average.pt"
data="/large_experiments/ust/xutaima/data/2022_h2_streaming/s2t_audio"
config="configs/all_w2v2_ccmtrx.yaml"
subset="epst-v1.0"
```

Then calling the following command.
```bash
agent="${simuleval_dir}/examples/fairseq_speech/fairseq_test_waitk_s2t_agent.py"

# step size means we run the policy every this number of encoder states
# each encoder state of the offline model we use has a span of 120ms
# Therefore we set the segment_size to 3 * 120 = 360ms
step=3
segment_size=`python -c "print(int(${step} * 120))"`

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
    --waitk-lagging ${k} \
    --fixed-predicision-ratio ${step}
```
The output will be in the `${output_dir}`. There will be two files in output directory `instances.log` and `scores`. `instances.log` contains the details of decoding process for each sentence, in which each line is a json string. `scores` contains the final evaluation results. On EuraParl v1.0 dev set, the results under this setting is shown as follow. The suffix "_CA" indicates "computation aware", which we consider the computation time while calculating the latency.
```bash
cat ${output_dir}
{
    "Quality": {
        "BLEU": 30.18841562122126
    },
    "Latency": {
        "AL": 3464.1245982859396,
        "AL_CA": 4435.5774802028545,
        "AP": 0.7880219553071673,
        "AP_CA": 0.9807443500745414,
        "DAL": 3791.4531129138304,
        "DAL_CA": 5463.9417615833845
    }
}
```

## Test Time Wait-K Cascaded Speech-to-Speech Translation (WIP)
The set to the previous section except that the agent is
```bash
agent="${simuleval_dir}/examples/fairseq_speech/fairseq_test_waitk_s2s_tts_agent.py"
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

(TODO, the quality is pretty bad, debugging.)
```bash
{
    "Quality": {
        "BLEU": 4.818878403927336
    },
    "Latency": {
        "BOW": {
            "AL": 3.8244810299117966,
            "AP": 0.9564174844486879,
            "DAL": 4.206126439099265
        },
        "EOW": {
            "AL": 4.2270508638703,
            "AP": 1.0218393994439947,
            "DAL": 4.689659468018182
        },
        "COW": {
            "AL": 4.006357623799013,
            "AP": 0.9891284302909775,
            "DAL": 4.415219217243761
        }
    }
}
```