#!/bin/bash
set -e

subset=$1 # choose from dev_epst, dev_mtedx_filt, dev_cv
step=$2
k=$3

checkpoint="/private/home/hirofumii/large_experiments/checkpoints/s2ut_pt/es_en/s2t/s2t.pt_es_en.config_mbart.asr.rdrop10.0.ls0.2.maxtok2.0k.uf30.lr0.0005.wu1k.seed1.arch_xm_transformer.W2V_cfm_L.dr0.1.ld0.2.al1.dld0.2.mBART_spm.LND.ca.ngpu24/avg_best_10_checkpoint.pt"
data="/large_experiments/seamless/ust/hirofumii/datasets/s2ut_pt/es_en/s2t"
config="config_mbart.yaml"

agent="./test_time_waitk_s2t_agent.py"

exp_dir="./experiments"

output=${exp_dir}/s2t-${subset}_output-${step}-${k}

mkdir -p ${output}

simuleval \
    --agent $agent \
    --fairseq-data ${data} \
    --fairseq-config ${config} \
    --fairseq-gen-subset ${subset}\
    --checkpoint ${checkpoint} \
    --output ${output} \
    --device cuda:0 \
    --source-segment-size `python -c "print(int(${step} * 40))"` \
    --waitk-lagging ${k} \
    --init-target-token "[en_XX]" \
    --fixed-predicision-ratio ${step} \
    --max-len-a 0.125 \
    --max-len-b 10 \
    --end-index 10
