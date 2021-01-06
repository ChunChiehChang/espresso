#!/bin/bash


set -e -o pipefail

stage=-10
ngpus=1 # num GPUs for multiple GPUs training within a single node; should match those in $free_gpu
free_qpu= # comma-separated available GPU ids, eg., "0" or "0,1"; automatically assigned if on CLSP grid
nj=30

# model and data related
affix=
kd_affix=
data_dir=data
exp_dir=exp
lang=lang_chain_e2e
tree_dir=e2e_tree  # it's actually just a trivial tree (no tree building)
whole_train_set=train_aug # will be split into train_set and valid_set
train_set=train_novalid
#valid_set=train_valid
valid_set=test
test_set=test
dumpdir=${data_dir}/dump   # directory to dump full features
checkpoint=checkpoint_best.pt

topk=10

#dataset_yomdle=/exp/scale18/ocr/data/derived/YOMDLE/final_chinese/
#dataset_yomdle=/exp/scale18/ocr/data/derived/YOMDLE/final_english/
dataset_yomdle=download/synth/
dataset_slam=/exp/scale18/ocr/data/derived/SLAM_2.0/Chinese/

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

dir=${exp_dir}/chain/tdnn_chain_e2e${affix:+_$affix}
dir_kd=${dir}_kd_top${topk}_${kd_affix}

local/common_data_prep.sh --data-dir ${data_dir} --stage $stage --dataset-yomdle $dataset_yomdle --dataset-slam $dataset_slam || exit 1;

if [ $stage -le 0 ]; then
    echo "Stage 0: Create the $lang Directory that Has a Specific HMM Topolopy"
    rm -rf $lang
    cp -r ${data_dir}/lang ${data_dir}/${lang}
    silphonelist=$(cat ${data_dir}/${lang}/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat ${data_dir}/${lang}/phones/nonsilence.csl) || exit 1;
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist > ${data_dir}/${lang}/topo
fi

if [ $stage -le 1 ]; then
    echo "Stage 1: Generate Denominator Graph and Numerator Fsts"
    echo "$0: Estimating a phone language model for the denominator graph..."
    mkdir -p ${exp_dir}/chain/${tree_dir}/log
    ${train_cmd} ${exp_dir}/chain/${tree_dir}/log/make_phone_lm.log \
        cat ${data_dir}/${whole_train_set}/text \| \
        steps/nnet3/chain/e2e/text_to_phones.py --between-silprob 0.1 \
        ${data_dir}/lang \| \
        utils/sym2int.pl -f 2- ${data_dir}/lang/phones.txt \| \
        chain-est-phone-lm --num-extra-lm-states=2000 \
        ark:- ${exp_dir}/chain/${tree_dir}/phone_lm.fst

    steps/nnet3/chain/e2e/prepare_e2e.sh --nj $nj --cmd "$train_cmd" \
        --shared-phones true ${data_dir}/${whole_train_set} ${data_dir}/${lang} ${exp_dir}/chain/${tree_dir}
    echo "$0: Making denominator fst..."
    $decode_cmd ${exp_dir}/chain/${tree_dir}/log/make_den_fst.log \
        chain-make-den-fst ${exp_dir}/chain/${tree_dir}/tree ${exp_dir}/chain/${tree_dir}/0.trans_mdl ${exp_dir}/chain/${tree_dir}/phone_lm.fst \
            ${exp_dir}/chain/${tree_dir}/den.fst ${exp_dir}/chain/${tree_dir}/normalization.fst || exit 1
    echo "$0: Making numerator fsts..."
    abs_treedir=`utils/make_absolute.sh ${exp_dir}/chain/${tree_dir}`
    $decode_cmd JOB=1:$nj ${exp_dir}/chain/${tree_dir}/log/make_num_fst_e2e.JOB.log \
        chain-make-num-fst-e2e ${exp_dir}/chain/${tree_dir}/0.trans_mdl ${exp_dir}/chain/${tree_dir}/normalization.fst \
            scp:${exp_dir}/chain/${tree_dir}/fst.JOB.scp ark,scp:${abs_treedir}/fst_nor.JOB.ark,${abs_treedir}/fst_nor.JOB.scp || exit 1
    for n in $(seq $nj); do
        cat ${exp_dir}/chain/${tree_dir}/fst_nor.${n}.scp || exit 1
    done > ${exp_dir}/chain/${tree_dir}/fst_nor.scp || exit 1
fi

if [ ${stage} -le 2 ]; then
    echo "Stage 2: Split the Whole Train Set into Train/Valid Set"
    # Get list of validation utterances.
    set +e
    awk '{print $1}' ${data_dir}/${whole_train_set}/utt2spk | utils/shuffle_list.pl 2>/dev/null | head -1000 > valid_uttlist
    set -e
    if [ -f ${data_dir}/${whole_train_set}/utt2uniq ]; then  # this matters if you use data augmentation.
        echo "File $data/utt2uniq exists, so augmenting valid_uttlist to"
        echo "include all perturbed versions of the same 'real' utterances."
        mv valid_uttlist valid_uttlist.tmp
        utils/utt2spk_to_spk2utt.pl ${data_dir}/${whole_train_set}/utt2uniq > uniq2utt
        cat valid_uttlist.tmp | utils/apply_map.pl ${data_dir}/${whole_train_set}/utt2uniq | \
            sort | uniq | utils/apply_map.pl utt2uniq | \
            awk '{for(n=1;n<=NF;n++) print $n;}' | sort  > valid_uttlist
        rm uniq2utt valid_uttlist.tmp 2>/dev/null
    fi

    # generate train/valid data dir
    utils/filter_scp.pl --exclude valid_uttlist ${data_dir}/${whole_train_set}/utt2spk | cut -d" " -f1 > novalid_uttlist || exit 1
    image/subset_data_dir.sh --utt-list novalid_uttlist ${data_dir}/${whole_train_set} ${data_dir}/${train_set} || exit 1
    image/subset_data_dir.sh --utt-list valid_uttlist ${data_dir}/${whole_train_set} ${data_dir}/${valid_set} || exit 1

    utils/filter_scp.pl novalid_uttlist ${exp_dir}/chain/${tree_dir}/fst_nor.scp > ${exp_dir}/chain/${tree_dir}/fst_novalid_nor.scp || exit 1
    utils/filter_scp.pl valid_uttlist ${exp_dir}/chain/${tree_dir}/fst_nor.scp > ${exp_dir}/chain/${tree_dir}/fst_valid_nor.scp || exit 1
    rm valid_uttlist novalid_uttlist 2>/dev/null

    # not all fsts can be generated successfully, just filter out those not having the fst
    for dataset in $train_set $valid_set; do
        tag=novalid && [[ "$dataset" == "$valid_set" ]] && tag=valid
        cp ${data_dir}/${dataset}/feats.scp ${data_dir}/${dataset}/feats.scp.tmp
        utils/filter_scp.pl ${exp_dir}/chain/${tree_dir}/fst_${tag}_nor.scp ${data_dir}/${dataset}/feats.scp.tmp \
            > ${data_dir}/${dataset}/feats.scp || exit 1
        rm ${data_dir}/${dataset}/feats.scp.tmp 2>/dev/null
        utils/fix_data_dir.sh ${data_dir}/${dataset} || exit 1
    done
fi

if [ ${stage} -le 3 ]; then
    echo "Stage 3: Dump Feature"
    for dataset in $train_set $valid_set $test_set; do
        nj=8
        utils/split_data.sh ${data_dir}/${dataset} $nj
        sdata=${data_dir}/${dataset}/split${nj}
        mkdir -p ${dumpdir}/${dataset}; abs_featdir=`utils/make_absolute.sh $dumpdir/${dataset}`
        ${train_cmd} JOB=1:$nj $abs_featdir/log/dump_feature.JOB.log \
            apply-cmvn --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp \
                scp:$sdata/JOB/feats.scp ark:- \| \
            copy-feats --compress=true --compression-method=2 ark:- \
                ark,scp:$abs_featdir/feats.JOB.ark,$abs_featdir/feats.JOB.scp || exit 1

        for n in $(seq $nj); do
            cat $abs_featdir/feats.$n.scp || exit 1
        done > $abs_featdir/feats.scp || exit 1
        
        rm $abs_featdir/feats.*.scp 2>/dev/null
        image/get_image2num_frames.py --feat-dim 60 --out-ark ${data_dir}/${dataset}/utt2num_frames ${data_dir}/${dataset}
        cat ${data_dir}/${dataset}/utt2num_frames > $abs_featdir/utt2num_frames || exit 1
        cat ${data_dir}/${dataset}/utt2spk > $abs_featdir/utt2spk || exit 1
    done
fi

if [ ${stage} -le 4 ]; then
    echo "Stage 4: Dump Json Files"
    train_feat=${dumpdir}/${train_set}/feats.scp
    train_fst=${exp_dir}/chain/${tree_dir}/fst_novalid_nor.scp
    train_text=${data_dir}/${train_set}/text
    train_utt2num_frames=${data_dir}/${train_set}/utt2num_frames
    valid_feat=$dumpdir/${valid_set}/feats.scp
    valid_fst=${exp_dir}/chain/${tree_dir}/fst_valid_nor.scp
    valid_text=${data_dir}/${valid_set}/text
    valid_utt2num_frames=${data_dir}/${valid_set}/utt2num_frames
    mkdir -p ${data_dir}/chain_e2e
    asr_prep_json.py --feat-files $train_feat --numerator-fst-files $train_fst --text-files $train_text \
        --utt2num-frames-files $train_utt2num_frames --output ${data_dir}/chain_e2e/train.json
    asr_prep_json.py --feat-files $valid_feat --numerator-fst-files $valid_fst --text-files $valid_text \
        --utt2num-frames-files $valid_utt2num_frames --output ${data_dir}/chain_e2e/valid.json

    for dataset in ${test_set} ${valid_set}; do
        nj=$(wc -l <${data_dir}/${dataset}/spk2utt)
        utils/split_data.sh ${data_dir}/${dataset} $nj
        utils/split_data.sh ${dumpdir}/${dataset} $nj
        for n in $(seq $nj); do
            feat=${dumpdir}/${dataset}/split$nj/$n/feats.scp
            text=${data_dir}/${dataset}/split$nj/$n/text
            utt2num_frames=${data_dir}/${dataset}/split$nj/$n/utt2num_frames
            asr_prep_json.py --feat-files $feat --text-files $text --utt2num-frames-files $utt2num_frames \
                --output ${data_dir}/chain_e2e/${dataset}.${n}.json
        done
    done
fi

num_targets=$(tree-info ${exp_dir}/chain/${tree_dir}/tree | grep num-pdfs | awk '{print $2}')

if [ ${stage} -le 5 ]; then
    echo "Stage 5: Model Training"
    valid_subset=valid
    mkdir -p $dir/log
    log_file=$dir/log/train.log
    [ -f $dir/checkpoint_last.pt ] && log_file="-a $log_file"
    update_freq=1
    qsub -v PATH -S /bin/bash -b y -q gpu.q@@rtx -cwd -j y -N ocr_train -l gpu=$ngpus,hostname='!r6n02*',mem_free=80G,h_rt=600:00:00 -o $dir/log/log_train -sync y ./limit_num_gpus.sh \
    python3 $ESPRESSO_ROOT/espresso/speech_train.py ${data_dir}/chain_e2e --task speech_recognition_hybrid --seed 1 --user-dir $ESPRESSO_ROOT/espresso \
        --log-interval $((200/ngpus/update_freq)) --log-format simple \
        --num-workers 0 --data-buffer-size 0 --max-tokens 12000 --max-sentences 32 --curriculum 1 --empty-cache-freq 50 \
        --valid-subset $valid_subset --max-sentences-valid 32 --ddp-backend no_c10d --update-freq $update_freq \
        --distributed-world-size $ngpus \
        --max-epoch 15 --optimizer adam --lr 0.001 --weight-decay 0.0 --start-reduce-lr-epoch 5 \
        --lr-scheduler reduce_lr_on_plateau_v2 --lr-shrink 0.5 \
        --save-dir $dir --restore-file checkpoint_last.pt --save-interval-updates $((400/ngpus/update_freq)) \
        --keep-interval-updates 5 --keep-last-epochs 10 --validate-interval 1 \
        --arch speech_yomdle --criterion lattice_free_mmi --num-targets $num_targets \
        --denominator-fst-path ${exp_dir}/chain/${tree_dir}/den.fst --leaky-hmm-coefficient 1e-03 \
        --max-source-positions 9999 --max-target-positions 9999 2>&1 | tee $log_file
fi

if [ ${stage} -le 6 ]; then
    graph_dir=${exp_dir}/chain/${tree_dir}/graph
    utils/mkgraph.sh --self-loop-scale 1.0 ${data_dir}/lang_test  ${exp_dir}/chain/${tree_dir} ${graph_dir}
    
    echo "Stage 6: Decoding"
    rm $dir/.error 2>/dev/null || true
    queue_opt="--num-threads 4"
    path=$dir/$checkpoint
    for dataset in $test_set $valid_set; do
        (
            data_affix=$(echo $dataset | sed s/test_//)
            decode_dir=decode_${data_affix}
            nj=$(wc -l <${data_dir}/${dataset}/spk2utt)

            $decode_cmd $queue_opt JOB=1:$nj $dir/${decode_dir}/log/decode.JOB.log \
                dump_posteriors.py ${data_dir}/chain_e2e --cpu --task speech_recognition_hybrid --user-dir espresso \
                    --max-tokens 12000 --max-sentences 32 --num-shards 1 --shard-id 0 --num-targets $num_targets \
                    --gen-subset $dataset.JOB \
                    --max-source-positions 9999 --path $path \| \
                latgen-faster-mapped --max-active=7000 --min-active=20 --beam=15 --lattice-beam=8 --acoustic-scale=1.0 \
                    --allow-partial=true --word-symbol-table="$graph_dir/words.txt" \
                    ${exp_dir}/chain/$tree_dir/0.trans_mdl $graph_dir/HCLG.fst ark:- \
                    "ark:| lattice-scale --acoustic-scale=10.0 ark:- ark:- | gzip -c >$dir/${decode_dir}/lat.JOB.gz" || exit 1
            local/score.sh --cmd "$decode_cmd" ${data_dir}/${dataset} ${graph_dir} $dir/${decode_dir} || exit 1
            echo $nj > $dir/${decode_dir}/num_jobs

    ) || touch $dir/.error &
    done
    wait
fi

if [ ${stage} -le 7 ]; then
    echo "Stage 7: Model Training KD"
    valid_subset=valid
    mkdir -p $dir_kd/log
    log_file=$dir_kd/log/train.log
    [ -f $dir_kd/checkpoint_last.pt ] && log_file="-a $log_file"
    update_freq=1
    qsub -v PATH -S /bin/bash -b y -q gpu.q@@rtx -cwd -j y -N ocr_train -l gpu=$ngpus,hostname='!r6n02*',mem_free=80G,h_rt=600:00:00 -o $dir_kd/log/log_train -sync y ./limit_num_gpus.sh \
    python3 $ESPRESSO_ROOT/espresso/speech_train.py ${data_dir}/chain_e2e --task speech_recognition_hybrid --seed 1 --user-dir $ESPRESSO_ROOT/espresso \
        --log-interval $((200/ngpus/update_freq)) --log-format simple \
        --num-workers 0 --data-buffer-size 0 --max-tokens 12000 --max-sentences 32 --curriculum 1 --empty-cache-freq 50 \
        --valid-subset $valid_subset --max-sentences-valid 32 --ddp-backend no_c10d --update-freq $update_freq \
        --distributed-world-size $ngpus \
        --max-epoch 15 --optimizer adam --lr 0.001 --weight-decay 0.0 --start-reduce-lr-epoch 5 \
        --lr-scheduler reduce_lr_on_plateau_v2 --lr-shrink 0.5 \
        --save-dir $dir_kd --restore-file checkpoint_last.pt --save-interval-updates $((400/ngpus/update_freq)) \
        --keep-interval-updates 5 --keep-last-epochs 10 --validate-interval 1 \
        --arch speech_yomdle --criterion lattice_free_mmi_kd_xent --num-targets $num_targets \
        --denominator-fst-path ${exp_dir}/chain/${tree_dir}/den.fst --leaky-hmm-coefficient 1e-03 \
        --teacher-model-path $dir/$checkpoint --kd-topk $topk \
        --max-source-positions 9999 --max-target-positions 9999 2>&1 | tee $log_file
fi

if [ ${stage} -le 8 ]; then
    graph_dir=${exp_dir}/chain/${tree_dir}/graph
    utils/mkgraph.sh --self-loop-scale 1.0 ${data_dir}/lang_test  ${exp_dir}/chain/${tree_dir} ${graph_dir}

    echo "Stage 8: Decoding KD"
    rm $dir_kd/.error 2>/dev/null || true
    queue_opt="--num-threads 4"
    path=$dir_kd/$checkpoint
    #for dataset in $test_set $valid_set; do
    for dataset in $valid_set; do
        (
            data_affix=$(echo $dataset | sed s/test_//)
            decode_dir=decode_${data_affix}
            nj=$(wc -l <${data_dir}/${dataset}/spk2utt)

            $decode_cmd $queue_opt JOB=1:$nj $dir_kd/${decode_dir}/log/decode.JOB.log \
                dump_posteriors.py ${data_dir}/chain_e2e --cpu --task speech_recognition_hybrid --user-dir espresso \
                    --max-tokens 12000 --max-sentences 32 --num-shards 1 --shard-id 0 --num-targets $num_targets \
                    --gen-subset $dataset.JOB \
                    --max-source-positions 9999 --path $path \| \
                latgen-faster-mapped --max-active=7000 --min-active=20 --beam=15 --lattice-beam=8 --acoustic-scale=1.0 \
                    --allow-partial=true --word-symbol-table="$graph_dir/words.txt" \
                    ${exp_dir}/chain/$tree_dir/0.trans_mdl $graph_dir/HCLG.fst ark:- \
                    "ark:| lattice-scale --acoustic-scale=10.0 ark:- ark:- | gzip -c >$dir_kd/${decode_dir}/lat.JOB.gz" || exit 1
            local/score.sh --cmd "$decode_cmd" ${data_dir}/${dataset} ${graph_dir} $dir_kd/${decode_dir} || exit 1
            echo $nj > $dir_kd/${decode_dir}/num_jobs

    ) || touch $dir_kd/.error &
    done
    wait
fi

#if [ ${stage} -le 9 ]; then
#    graph_dir=${exp_dir}/chain/${tree_dir}/graph_large
#    utils/mkgraph.sh --self-loop-scale 1.0 ${data_dir}/lang_test_large  ${exp_dir}/chain/${tree_dir} ${graph_dir}
#
#    echo "Stage 8: Decoding KD"
#    rm $dir_kd/.error 2>/dev/null || true
#    queue_opt="--num-threads 4"
#    path=$dir_kd/$checkpoint
#    #path=baseline_experiments_with_topk/exp/chain/tdnn_chain_e2e_kd_top10/checkpoint_best.pt
#    for dataset in $test_set $valid_set; do
#        (
#            data_affix=$(echo $dataset | sed s/test_//)
#            decode_dir=decode_${data_affix}_kd_large
#            nj=$(wc -l <${data_dir}/${dataset}/spk2utt)
#
#            $decode_cmd $queue_opt JOB=1:$nj $dir_kd/${decode_dir}/log/decode.JOB.log \
#                dump_posteriors.py ${data_dir}/chain_e2e --cpu --task speech_recognition_hybrid --user-dir espresso \
#                    --max-tokens 12000 --max-sentences 32 --num-shards 1 --shard-id 0 --num-targets $num_targets \
#                    --gen-subset $dataset.JOB \
#                    --max-source-positions 9999 --path $path \| \
#                latgen-faster-mapped --max-active=7000 --min-active=20 --beam=15 --lattice-beam=8 --acoustic-scale=1.0 \
#                    --allow-partial=true --word-symbol-table="$graph_dir/words.txt" \
#                    ${exp_dir}/chain/$tree_dir/0.trans_mdl $graph_dir/HCLG.fst ark:- \
#                    "ark:| lattice-scale --acoustic-scale=10.0 ark:- ark:- | gzip -c >$dir_kd/${decode_dir}/lat.JOB.gz" || exit 1
#            local/score.sh --cmd "$decode_cmd" ${data_dir}/${dataset} ${graph_dir} $dir_kd/${decode_dir} || exit 1
#            echo $nj > $dir_kd/${decode_dir}/num_jobs
#
#    ) || touch $dir_kd/.error &
#    done
#    wait
#fi
