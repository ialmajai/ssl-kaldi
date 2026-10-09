#!/bin/bash
# Copyright   2025  Ibrahim Almajai
# License: Apache 2.0.

set -euo pipefail

stage=0
stop_stage=100   # run stages [stage, stop_stage]; without this
                 # --stage N ran to the end and could overwrite exp/
nj=8
njtest=4

pca_dim=30
pca_train_utts=28000
interp_mode=nearest
upsample_factor=2

#path to grid corpus dataset
datasrc=grid-corpus
# Lombard GRID front-view videos, for stages 14-15: unzip lombardgrid_front.zip into lombard-grid/
lombard_dir=lombard-grid/lombardgrid/front

model_size=base
crop_mode=meanface       # canonical AV-HuBERT mean-face alignment (main method);
                         # resolution-invariant, marginally best. Alternatives:
                         # legacy 'centroid' (fixed 64px) or centroid + --crop-scale (dynamic IOD).

# Empty values take the model_size defaults below: the re-validated best layers
# (base 4, large 13; see README layer sweeps).
avhubert_model=
encoder_layer=
njfeats=

#path to avhubert codebase (see README: cloned into this directory)
avhubert_path=$PWD/av_hubert

lang=data/lang
dict=data/local/dict
langtmp=data/local/lang
data_affix=_upsampled
njpca=15

monogauss=1000
numLeavesTri1=2000
numGaussTri1=10000
numLeavesMLLT=3500
numGaussMLLT=20000
numLeavesSAT=5000
numGaussSAT=40000

. cmd.sh
. path.sh
. ./utils/parse_options.sh

# Resolve model_size defaults after parsing so --model-size takes effect.
case $model_size in
  base)  : ${avhubert_model:=base_vox_iter5.pt}  ${encoder_layer:=4}  ${njfeats:=8} ;;
  large) : ${avhubert_model:=large_vox_iter5.pt} ${encoder_layer:=13} ${njfeats:=4} ;;
  *) echo "$0: --model-size must be base or large, got '$model_size'"; exit 1 ;;
esac

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  python local/download_grid_corpus.py --dir $datasrc
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then 	
  local/grid_data_prep.sh $datasrc
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "preparing dictionary and lang"
  mkdir -p $langtmp $dict
  local/chime1_prepare_dict.sh $dict || exit 1
   
  utils/prepare_lang.sh --num-sil-states 5 \
     --num-nonsil-states 3 \
     --position-dependent-phones false \
     --share-silence-phones true \
     $dict "a"  $langtmp $lang || exit 1

  local/grid_prepare_grammar.sh || exit 1
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then

  if [ ! -f "input/$avhubert_model" ]; then
    echo "Downloading AvHubert checkpoint"
	# https://facebookresearch.github.io/av_hubert: AV-HuBERT Base | LRS3 + VoxCeleb2 (En) | No finetuning
	model_link=https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/$avhubert_model
    wget $model_link -O input/$avhubert_model.tmp || { rm -f input/$avhubert_model.tmp; exit 1; }
    mv input/$avhubert_model.tmp input/$avhubert_model
  fi
  if [ ! -f input/shape_predictor_68_face_landmarks.dat ]; then
    echo "Downloading dlib 68-point landmark model"
    wget https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O input/sp68.bz2.tmp \
      || { rm -f input/sp68.bz2.tmp; exit 1; }
    bunzip2 -c input/sp68.bz2.tmp > input/sp68.dat.tmp \
      && mv input/sp68.dat.tmp input/shape_predictor_68_face_landmarks.dat && rm input/sp68.bz2.tmp
  fi
  # Mean-face reference for --crop-mode meanface; not shipped with the av_hubert checkout.
  mf=$avhubert_path/avhubert/20words_mean_face.npy
  if [ "$crop_mode" = meanface ] && [ ! -f $mf ]; then
    wget https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy \
      -O $mf.tmp || { rm -f $mf.tmp; exit 1; }
    mv $mf.tmp $mf
  fi

  for x in train test; do
    echo "preparing features"
    rm -rf data/${x}_raw
    local/copy_data_dir.sh data/$x data/${x}_raw
	
    local/make_avhubert.sh --cmd "$feats_cmd"  --nj $njfeats  --ckpt input/$avhubert_model \
     --avhubert-path $avhubert_path --layer $encoder_layer --crop-mode $crop_mode data/${x}_raw  || exit 1;
    
    steps/compute_cmvn_stats.sh data/${x}_raw   || exit 1;
    utils/fix_data_dir.sh data/${x}_raw
  done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  pca_model=pca-${pca_dim}d.pt    
  pca_dir=pca
  mkdir -p $pca_dir
 
  if [[ ! -f $pca_dir/$pca_model  ||  $pca_dir/$pca_model \
	  -ot data/train_raw/feats.scp ]] ; then
    echo "Training PCA model"
    python shared/pca.py  --pca_dim=$pca_dim --mode=train \
      --feats_scp=data/train_raw/feats.scp \
      --pca_model=$pca_dir/$pca_model \
      --max_utts=$pca_train_utts
  fi

  for x in train test; do
    echo "preparing pca features"    
    shared/make_pca_features.sh --cmd "$decode_cmd" --nj $njpca \
	  --pca-model $pca_dir/$pca_model \
          data/${x}_raw data/${x}   || exit 1;
    steps/compute_cmvn_stats.sh data/$x   || exit 1;
    utils/fix_data_dir.sh data/$x
  done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then

  for x in train test; do
    echo "upsampling features"
    src=data/${x}
    dst=data/${x}_upsampled
    local/copy_data_dir.sh $src $dst
    shared/upsample_features.sh --cmd "$decode_cmd" --nj $njpca \
      --interp-mode $interp_mode --upsample-factor $upsample_factor \
       $src $dst   || exit 1;

    steps/compute_cmvn_stats.sh $dst   || exit 1;
    utils/fix_data_dir.sh $dst
  done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  steps/train_mono.sh --nj 8 --cmd "$train_cmd" --boost-silence 1.5 \
    data/train$data_affix data/lang exp/mono
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  utils/mkgraph.sh  data/lang  exp/mono exp/mono/graph
  steps/decode.sh  --nj $njtest --cmd "$decode_cmd" \
    exp/mono/graph data/test$data_affix exp/mono/decode  
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  # Get alignments from monophone system.
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train$data_affix data/lang exp/mono exp/mono_ali

  steps/train_deltas.sh --cmd "$train_cmd" --boost-silence 1.5 \
    $numLeavesTri1  $numGaussTri1 data/train$data_affix data/lang \
    exp/mono_ali exp/tri1

  utils/mkgraph.sh data/lang exp/tri1 exp/tri1/graph
  steps/decode.sh  --nj $njtest --cmd "$decode_cmd" \
    exp/tri1/graph data/test$data_affix exp/tri1/decode
  
  # align tri1
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train$data_affix data/lang exp/tri1 exp/tri1_ali
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  # train and decode tri2b [LDA+MLLT]
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" $numLeavesMLLT \
    $numGaussMLLT data/train$data_affix data/lang exp/tri1_ali exp/tri2b

  utils/mkgraph.sh data/lang exp/tri2b exp/tri2b/graph
  steps/decode.sh --nj $njtest --cmd "$decode_cmd" \
    exp/tri2b/graph data/test$data_affix exp/tri2b/decode 
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
# Align all data with LDA+MLLT system (tri2b)
steps/align_si.sh --nj $nj --cmd "$train_cmd" \
   data/train$data_affix data/lang exp/tri2b exp/tri2b_ali
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
  # Do LDA+MLLT+SAT, and decode.
  steps/train_sat.sh --cmd "$train_cmd"  $numLeavesSAT  $numGaussSAT \
    data/train$data_affix data/lang exp/tri2b_ali exp/tri3b
    
  utils/mkgraph.sh data/lang exp/tri3b exp/tri3b/graph
  steps/decode_fmllr.sh --nj $njtest --cmd "$decode_cmd" \
      exp/tri3b/graph data/test$data_affix exp/tri3b/decode
fi

if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
  # Align all data with LDA+MLLT+SAT
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
      data/train$data_affix data/lang exp/tri3b exp/tri3b_ali
fi

data_fmllr=data-fmllr-tri3b
gmm=exp/tri3b
if [ $stage -le 12 ] && [ $stop_stage -ge 12 ]; then

  dirc=${data_fmllr}/test$data_affix
  mkdir -p $dirc
  steps/nnet/make_fmllr_feats.sh --nj 4 --cmd "$train_cmd" \
     --transform-dir $gmm/decode \
     $dirc data/test$data_affix $gmm $dirc/log $dirc/data
  steps/compute_cmvn_stats.sh $dirc  $dirc/log $dirc/data || exit 1;

  dirc=${data_fmllr}/train$data_affix
  mkdir -p $dirc
  steps/nnet/make_fmllr_feats.sh --nj 8 --cmd "$train_cmd" \
     --transform-dir ${gmm}_ali \
     $dirc data/train$data_affix $gmm $dirc/log $dirc/data
  steps/compute_cmvn_stats.sh $dirc  $dirc/log $dirc/data || exit 1;
fi

if [ $stage -le 13 ] && [ $stop_stage -ge 13 ]; then
  # Train the TDNN-F chain on the fMLLR features (5.83% with --model-size large).
  local/chain/run_tdnn_fmllr_tdnnf.sh --stage 9 --gmm $gmm --data-fmllr $data_fmllr \
    --train-set train$data_affix --test-set test$data_affix
fi

if [ $stage -le 14 ] && [ $stop_stage -ge 14 ]; then
  # Build Lombard features as the training data were, then GMM-SAT decode for fMLLR.
  [ -d "$lombard_dir" ] || { echo "$0: no Lombard videos in $lombard_dir (unzip lombardgrid_front.zip into lombard-grid/)"; exit 1; }
  rm -rf data/lomb_raw data/lomb data/lomb$data_affix
  python local/lombard_data_prep.py --video-dir $lombard_dir --out data/lomb_raw
  # Lower --min-frames: Lombard clips run ~60 frames.
  local/make_avhubert.sh --cmd "$feats_cmd" --nj $njfeats --ckpt input/$avhubert_model \
    --avhubert-path $avhubert_path --layer $encoder_layer --crop-mode $crop_mode \
    --min-frames 30 data/lomb_raw || exit 1;
  steps/compute_cmvn_stats.sh data/lomb_raw && utils/fix_data_dir.sh data/lomb_raw

  local/copy_data_dir.sh data/lomb_raw data/lomb
  shared/make_pca_features.sh --cmd "$decode_cmd" --nj $njpca --pca-model pca/pca-${pca_dim}d.pt \
    data/lomb_raw data/lomb || exit 1;
  steps/compute_cmvn_stats.sh data/lomb && utils/fix_data_dir.sh data/lomb
  local/copy_data_dir.sh data/lomb data/lomb$data_affix
  shared/upsample_features.sh --cmd "$decode_cmd" --nj $njpca \
    --interp-mode $interp_mode --upsample-factor $upsample_factor \
    data/lomb data/lomb$data_affix || exit 1;
  steps/compute_cmvn_stats.sh data/lomb$data_affix && utils/fix_data_dir.sh data/lomb$data_affix

  steps/decode_fmllr.sh --nj $njtest --cmd "$decode_cmd" \
    $gmm/graph data/lomb$data_affix $gmm/decode_lombard
  dirc=${data_fmllr}/lomb$data_affix
  steps/nnet/make_fmllr_feats.sh --nj $njtest --cmd "$train_cmd" \
    --transform-dir $gmm/decode_lombard \
    $dirc data/lomb$data_affix $gmm $dirc/log $dirc/data
  steps/compute_cmvn_stats.sh $dirc $dirc/log $dirc/data || exit 1;
fi

if [ $stage -le 15 ] && [ $stop_stage -ge 15 ]; then
  # Decode Lombard with the fMLLR TDNN-F chain; score both systems on the dev/eval split.
  chain=exp/chain/tdnntdnnf
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --nj $njtest --cmd "$decode_cmd" \
    $chain/graph ${data_fmllr}/lomb$data_affix $chain/decode_lombard || exit 1;
  for d in $gmm/decode_lombard $chain/decode_lombard; do
    split=$(local/score_lombard_split.sh $d data/lang/words.txt data/lomb_raw)
    echo "$d: $split"
  done
fi

exit 0;
