#!/usr/bin/env bash
# Frozen AV-HuBERT large fMLLR stack on layer 11, the GMM tap chosen on the 14
# Lombard dev speakers by run_gmm_dev_layer_sweep.sh (2026-10-19; the published
# layer 13 was chosen on GRID test). Reuses the sweep's L11 GMM ladder and
# training features, extracts L11 for the 4,000-clip GRID test and the shared
# Lombard set, decodes the GMM ladder, builds fMLLR features, and trains the
# fMLLR TDNN-F chain (as tdnntdnnf_l13) with three seeds. Seeds 1 and 2 reuse
# seed 0's tree, lattices and egs. Prints every system at the LM setting chosen
# on Lombard dev.
# Run from egs/grid/s5 in the ssl-kaldi environment:
#   ./run_large_l11_fmllr.sh [--store DIR]
set -uo pipefail
store=   # raw L11 test and Lombard features; default data/ft_store
. ./cmd.sh; . ./path.sh
. utils/parse_options.sh
store=${store:-$PWD/data/ft_store}/ssl-kaldi-grid-Ld11
tag=Ld11; L=11; ckpt=input/large_vox_iter5.pt
E=exp-gmm-dev-sweep/large/l11; F=data-fmllr-tri3b; pca=gmm_dev_sweep/large/pca-30d-l11.pt
step() { echo "######## $(date +%H:%M:%S) [$tag] $*"; }
nj_for() { local n; n=$(wc -l < $1/spk2utt); echo $(( n < 8 ? n : 8 )); }

# 1. Training features: the sweep's L11 PCA-30, 50 fps set, under this stack's name.
[ -s data/${tag}_train_up/feats.scp ] || local/copy_data_dir.sh data/dvs_large11_train_up data/${tag}_train_up

# 2. L11 for GRID test and both Lombard sets: extract, PCA-30, x2 upsample.
for set in r_test s_lomb_l s_lomb_p; do
  raw=data/${tag}_${set}_raw; p=data/${tag}_${set}; up=${p}_up
  [ -s $up/feats.scp ] && continue
  step "extract, PCA and upsample $set"
  min=$([ $set = r_test ] && echo 74 || echo 30)
  rm -rf $raw $p $up $store/$set; mkdir -p $store/$set
  local/copy_data_dir.sh data/$set $raw
  rm -f $raw/feats.scp $raw/cmvn.scp; ln -sfn $store/$set $raw/data
  local/make_avhubert.sh --cmd run.pl --nj 2 --ckpt $ckpt --avhubert-path $PWD/av_hubert \
    --layer $L --crop-mode meanface --min-frames $min $raw || { echo "FAILED extraction $set"; exit 1; }
  utils/fix_data_dir.sh $raw
  local/copy_data_dir.sh $raw $p; rm -f $p/feats.scp $p/cmvn.scp
  shared/make_pca_features.sh --cmd "$decode_cmd" --nj 4 --pca-model $pca $raw $p
  steps/compute_cmvn_stats.sh $p; utils/fix_data_dir.sh $p
  local/copy_data_dir.sh $p $up
  shared/upsample_features.sh --cmd "$decode_cmd" --nj 4 --interp-mode nearest --upsample-factor 2 $p $up
  steps/compute_cmvn_stats.sh $up; utils/fix_data_dir.sh $up
done

# 3. GMM ladder decodes (the tri3b with fMLLR, as the published rows).
for m in mono tri1 tri2b tri3b; do
  [ -f $E/$m/graph/HCLG.fst ] || utils/mkgraph.sh data/lang $E/$m $E/$m/graph
  for set in r_test s_lomb_l s_lomb_p; do
    d=$E/$m/decode_$set; [ -f $d/.done ] && continue
    step "decode $m on $set"
    if [ $m = tri3b ]; then
      steps/decode_fmllr.sh --nj $(nj_for data/${tag}_${set}_up) --cmd "$decode_cmd" $E/$m/graph data/${tag}_${set}_up $d
    else
      steps/decode.sh --nj $(nj_for data/${tag}_${set}_up) --cmd "$decode_cmd" $E/$m/graph data/${tag}_${set}_up $d
    fi || { echo "FAILED decode $m $set"; exit 1; }
    touch $d/.done
  done
done

# 4. fMLLR alignments and features (train from alignments, test sets from decodes).
[ -f ${E}/tri3b_ali_$tag/ali.1.gz ] || steps/align_fmllr.sh --nj 8 --cmd "$train_cmd" \
  data/${tag}_train_up data/lang $E/tri3b $E/tri3b_ali_$tag || { echo "FAILED align"; exit 1; }
for set in train r_test s_lomb_l s_lomb_p; do
  dirc=$F/${tag}_${set}_up; [ -s $dirc/feats.scp ] && continue
  case $set in train) tr=$E/tri3b_ali_$tag;; *) tr=$E/tri3b/decode_$set;; esac
  step "fMLLR features $set"
  steps/nnet/make_fmllr_feats.sh --nj 4 --cmd "$train_cmd" --transform-dir $tr \
    $dirc data/${tag}_${set}_up $E/tri3b $dirc/log $dirc/data || { echo "FAILED fMLLR $set"; exit 1; }
  steps/compute_cmvn_stats.sh $dirc $dirc/log $dirc/data
done

# 5. fMLLR TDNN-F chain, three seeds; seed 0 builds the lattices, tree and egs.
for seed in 0 1 2; do
  affix=tdnnf_uc$([ $seed = 0 ] || echo _s$seed); dir=exp/chain/tdnn${affix}_$tag
  if [ ! -f $dir/final.mdl ]; then
    step "chain seed $seed"
    if [ $seed = 0 ]; then opts="--stage 9"; else opts="--stage 12 --common-egs-dir exp/chain/tdnntdnnf_uc_$tag/egs"; fi
    local/chain/run_tdnn_fmllr_tdnnf.sh --affix $affix --suffix _$tag --srand $seed $opts \
      --gmm $E/tri3b --train-set ${tag}_train_up --test-set ${tag}_r_test_up \
      || { echo "FAILED chain seed $seed"; exit 1; }
  fi
  for set in s_lomb_l s_lomb_p; do
    d=$dir/decode_$set; [ -f $d/.done ] && continue
    step "decode seed $seed on $set"
    steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --nj 8 --cmd "$decode_cmd" \
      $dir/graph $F/${tag}_${set}_up $d || { echo "FAILED chain decode $set"; exit 1; }
    touch $d/.done
  done
done

# 6. Results at the LM setting chosen on the Lombard dev speakers.
row() {  # row NAME GRID_DECODE LOMBARD_DECODE PLAIN_DECODE
  local pooled; pooled=$(local/score_lombard_pooled.sh $3 $4)
  echo "$1: GRID $(grep WER $2/wer_${pooled##*setting=} | awk '{print $2}')" \
    "| Lombard $(local/score_lombard_split.sh $3 data/lang/words.txt data/${tag}_s_lomb_l_raw)" \
    "| plain $(local/score_lombard_split.sh $4 data/lang/words.txt data/${tag}_s_lomb_p_raw) | $pooled"
}
for m in mono tri1 tri2b tri3b; do row "L11 $m" $E/$m/decode_r_test $E/$m/decode_s_lomb_l $E/$m/decode_s_lomb_p; done
for seed in 0 1 2; do
  dir=exp/chain/tdnntdnnf_uc$([ $seed = 0 ] || echo _s$seed)_$tag
  row "L11 fMLLR chain seed $seed" $dir/decode_test $dir/decode_s_lomb_l $dir/decode_s_lomb_p
done
echo LARGE-L11-FMLLR-DONE
