#!/bin/bash

set -euo pipefail

stage=-10
nj=30
data_dir=data
train_set=train
test_set=test

dataset_yomdle=
dataset_slam=

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

if [ $stage -le -4 ]; then
    mkdir -p ${data_dir}/${train_set}
    mkdir -p ${data_dir}/${test_set}
    local/process_data.py ${dataset_yomdle} ${data_dir}/${train_set}
    local/process_data.py ${dataset_slam} ${data_dir}/${test_set}
    image/fix_data_dir.sh ${data_dir}/${train_set}
    image/fix_data_dir.sh ${data_dir}/${test_set}
fi

if [ $stage -le -3 ]; then
    for datasplit in ${train_set} ${test_set}; do
        local/extract_features.sh --nj $nj --cmd "$train_cmd" \
            --feat-dim 60 --num-channels 1 \
            ${data_dir}/${datasplit}
        steps/compute_cmvn_stats.sh ${data_dir}/${datasplit}
        utils/fix_data_dir.sh ${data_dir}/${datasplit}
    done

    local/augment_data.sh --nj $nj --cmd "$train_cmd" --feat-dim 60 ${data_dir}/${train_set} ${data_dir}/${train_set}_aug ${data_dir}
    steps/compute_cmvn_stats.sh ${data_dir}/${train_set}_aug
fi

if [ $stage -le -2 ]; then
    if [ ! -f ${data_dir}/${train_set}/bpe.out ]; then
        cut -d' ' -f2- ${data_dir}/${train_set}/text | \
            utils/lang/bpe/prepend_words.py | \
            python3 utils/lang/bpe/learn_bpe.py -s 700 > ${data_dir}/${train_set}/bpe.out
        for datasplit in ${train_set} ${train_set}_aug ${test_set}; do
            cut -d' ' -f1 ${data_dir}/${datasplit}/text > ${data_dir}/${datasplit}/ids
            cut -d' ' -f2- ${data_dir}/${datasplit}/text | \
                utils/lang/bpe/prepend_words.py | \
                python3 utils/lang/bpe/apply_bpe.py -c ${data_dir}/${train_set}/bpe.out | \
                sed 's/@@//g' > ${data_dir}/${datasplit}/bpe_text
            mv ${data_dir}/${datasplit}/text ${data_dir}/${datasplit}/text.old
            paste -d' ' ${data_dir}/${datasplit}/ids ${data_dir}/${datasplit}/bpe_text > ${data_dir}/${datasplit}/text
        done
    fi
    local/prepare_dict.sh --data-dir ${data_dir} --dir ${data_dir}/local/dict
    utils/prepare_lang.sh --num-sil-states 4 --num-nonsil-states 6 --sil-prob 0.95 --position-dependent-phones false \
        ${data_dir}/local/dict "<sil>" ${data_dir}/lang/temp ${data_dir}/lang
    silphonelist=`cat ${data_dir}/lang/phones/silence.csl`
    nonsilphonelist=`cat ${data_dir}/lang/phones/nonsilence.csl`
    local/gen_topo.py 8 4 10 ${nonsilphonelist} ${silphonelist} ${data_dir}/lang/phones.txt > $data_dir/lang/topo
    utils/lang/bpe/add_final_optional_silence.sh --final-sil-prob 0.5 ${data_dir}/lang

    local/train_lm.sh --data-dir ${data_dir} --dir ${data_dir}/local/local_lm
    utils/format_lm.sh ${data_dir}/lang ${data_dir}/local/local_lm/data/arpa/3gram_unpruned.arpa.gz \
        ${data_dir}/local/dict/lexicon.txt ${data_dir}/lang_test
fi
