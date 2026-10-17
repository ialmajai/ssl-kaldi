#!/usr/bin/env bash
# Choose the GMM ladder's AV-HuBERT layer on the 14 Lombard dev speakers rather
# than on GRID test (2026-10-17, review of the paper: the published taps, base L4
# and large L13, came from a tri3b sweep scored on test). Same pipeline as the
# published -rt systems: one multi-layer extraction pass over the training set
# (local/compute_avhubert_layers.py, mean-face), PCA-30 fitted on train, x2
# nearest upsampling to 50 fps, mono -> tri1 -> tri2b -> tri3b with the sweep's
# settings, then decode_fmllr on the dev speakers' Lombard and plain clips,
# extracted per layer with local/make_avhubert.sh as the shared Lombard set is.
# The published layer reuses its -rt model and existing Lombard decodes.
# Prints dev WER per layer (pooled Lombard + plain, best setting on dev).
# Run from egs/grid/s5 in the ssl-kaldi environment:
#   ./run_gmm_dev_layer_sweep.sh --model base  --layers 2,3,4,5,6,7,8  [--store DIR]
#   ./run_gmm_dev_layer_sweep.sh --model large --layers 9,11,13,15,17 [--store DIR]
set -uo pipefail
model=base
layers=2,3,4,5,6,7,8
store=          # raw training features (deleted after PCA); default data/gmm_sweep_store
nj=2            # extraction jobs, each holding one encoder on the GPU
. ./cmd.sh; . ./path.sh
. utils/parse_options.sh
store=${store:-$PWD/data/gmm_sweep_store}

case $model in
  base)  ckpt=input/base_vox_iter5.pt;  pub=4;  pubE=exp-sweep-mf50-rt/l4;   pubtag=l4 ;;
  large) ckpt=input/large_vox_iter5.pt; pub=13; pubE=exp-sweep-mfL50-rt/l13; pubtag=l13 ;;
  *) echo "$0: --model base|large"; exit 1 ;;
esac
step() { echo "######## $(date +%H:%M:%S) [$model] $*"; }
D=gmm_dev_sweep/$model; mkdir -p $D $store/$model
LL=""; for L in ${layers//,/ }; do [ $L = $pub ] || LL="$LL $L"; done

# Dev speakers' Lombard and plain clips from the shared set.
for set in lomb_l lomb_p; do
  d=data/dvs_${set}_dev
  [ -s $d/utt2spk ] || utils/subset_data_dir.sh --spk-list local/lombard_dev_speakers.txt data/s_$set $d
done

# 1. One multi-layer pass over the training set for the layers without a model.
todo=""; for L in $LL; do [ -s data/dvs_${model}${L}_train_up/feats.scp ] || todo="$todo,$L"; done
todo=${todo#,}
if [ -n "$todo" ]; then
  step "extracting training layers $todo"
  log=$D/log_extract; mkdir -p $log
  split=; for j in $(seq $nj); do split="$split $log/v.$j.scp"; done
  utils/split_scp.pl data/train/video.scp $split
  $train_cmd JOB=1:$nj $log/extract.JOB.log \
    python local/compute_avhubert_layers.py \
      --out-template "ark,scp:$store/$model/raw_l{layer}.JOB.ark,$store/$model/raw_l{layer}.JOB.scp" \
      --layers $todo --ckpt $ckpt --path $PWD/av_hubert --crop-mode meanface scp:$log/v.JOB.scp \
    || { echo "FAILED extraction"; exit 1; }
fi

for L in $LL; do
  T=data/dvs_${model}${L}_train; E=exp-gmm-dev-sweep/$model/l$L; pca=$D/pca-30d-l$L.pt
  # 2. PCA-30 on train, x2 upsample, drop the raw layer.
  if [ ! -s ${T}_up/feats.scp ]; then
    step "L$L: PCA and upsample"
    raw=${T}_raw; rm -rf $raw $T ${T}_up; utils/copy_data_dir.sh data/train $raw
    for j in $(seq $nj); do cat $store/$model/raw_l$L.$j.scp; done | sort > $raw/feats.scp
    utils/fix_data_dir.sh $raw
    [ -f $pca ] || python shared/pca.py --pca_dim=30 --mode=train \
      --feats_scp=$raw/feats.scp --pca_model=$pca --max_utts=28000 || { echo "FAILED PCA L$L"; exit 1; }
    utils/copy_data_dir.sh data/train $T
    shared/make_pca_features.sh --cmd "$decode_cmd" --nj 8 --pca-model $pca $raw $T
    steps/compute_cmvn_stats.sh $T; utils/fix_data_dir.sh $T
    utils/copy_data_dir.sh $T ${T}_up
    shared/upsample_features.sh --cmd "$decode_cmd" --nj 8 --interp-mode nearest --upsample-factor 2 $T ${T}_up
    steps/compute_cmvn_stats.sh ${T}_up; utils/fix_data_dir.sh ${T}_up
    rm -rf $raw $store/$model/raw_l$L.*
  fi
  # 3. GMM ladder with the published sweep's settings.
  [ -f $E/mono/final.mdl ] || { step "L$L: mono"; steps/train_mono.sh --nj 8 --cmd "$train_cmd" --boost-silence 1.5 ${T}_up data/lang $E/mono; }
  [ -f $E/mono_ali/ali.1.gz ] || steps/align_si.sh --nj 8 --cmd "$train_cmd" ${T}_up data/lang $E/mono $E/mono_ali
  [ -f $E/tri1/final.mdl ] || steps/train_deltas.sh --cmd "$train_cmd" --boost-silence 1.5 2000 10000 ${T}_up data/lang $E/mono_ali $E/tri1
  [ -f $E/tri1_ali/ali.1.gz ] || steps/align_si.sh --nj 8 --cmd "$train_cmd" ${T}_up data/lang $E/tri1 $E/tri1_ali
  [ -f $E/tri2b/final.mdl ] || steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 3500 20000 ${T}_up data/lang $E/tri1_ali $E/tri2b
  [ -f $E/tri2b_ali/ali.1.gz ] || steps/align_si.sh --nj 8 --cmd "$train_cmd" ${T}_up data/lang $E/tri2b $E/tri2b_ali
  [ -f $E/tri3b/final.mdl ] || { step "L$L: tri3b"; steps/train_sat.sh --cmd "$train_cmd" 5000 40000 ${T}_up data/lang $E/tri2b_ali $E/tri3b; }
  [ -f $E/tri3b/graph/HCLG.fst ] || utils/mkgraph.sh data/lang $E/tri3b $E/tri3b/graph
  # 4. Dev speakers: extract this layer, project, upsample, decode.
  for set in lomb_l lomb_p; do
    v=data/dvs_${set}_dev_${model}${L}; out=$E/tri3b/decode_${set}_dev
    [ -f $out/.done ] && continue
    step "L$L: decode $set dev"
    rm -rf ${v}_raw $v ${v}_up; utils/copy_data_dir.sh data/dvs_${set}_dev ${v}_raw
    local/make_avhubert.sh --cmd run.pl --nj 2 --ckpt $ckpt --avhubert-path $PWD/av_hubert \
      --layer $L --crop-mode meanface --min-frames 30 ${v}_raw || { echo "FAILED dev extraction L$L"; exit 1; }
    utils/fix_data_dir.sh ${v}_raw
    utils/copy_data_dir.sh ${v}_raw $v
    shared/make_pca_features.sh --cmd "$decode_cmd" --nj 4 --pca-model $pca ${v}_raw $v
    steps/compute_cmvn_stats.sh $v; utils/fix_data_dir.sh $v
    utils/copy_data_dir.sh $v ${v}_up
    shared/upsample_features.sh --cmd "$decode_cmd" --nj 4 --interp-mode nearest --upsample-factor 2 $v ${v}_up
    steps/compute_cmvn_stats.sh ${v}_up; utils/fix_data_dir.sh ${v}_up
    steps/decode_fmllr.sh --nj 4 --cmd "$decode_cmd" $E/tri3b/graph ${v}_up $out || { echo "FAILED decode L$L"; exit 1; }
    touch $out/.done
  done
done

echo "==== $model GMM tri3b by layer, Lombard dev-14 (pooled Lombard + plain) ===="
for L in ${layers//,/ }; do
  if [ $L = $pub ]; then l=$pubE/tri3b/decode_s_lomb_l; p=$pubE/tri3b/decode_s_lomb_p; note=" (published, -rt)"
  else l=exp-gmm-dev-sweep/$model/l$L/tri3b/decode_lomb_l_dev; p=exp-gmm-dev-sweep/$model/l$L/tri3b/decode_lomb_p_dev; note=""; fi
  printf "  L%-3s %s%s\n" $L "$(local/score_lombard_pooled.sh $l $p 2>/dev/null | grep -oE 'dev=[0-9.]+')" "$note"
done
echo "GMM-DEV-SWEEP-$model-DONE"
