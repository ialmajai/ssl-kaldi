#!/usr/bin/env bash
# Audio-visual fine-tune, video-only features (2026-10-17, NOTES.md "Idea:
# audio-visual fine-tuning"): base LoRA r32 fine-tuned on GRID audio + video
# with modality dropout (base_ctc_lora_av.yaml), then read at L10 from video
# alone through the same e2e chain (chain seed 0) as the three video-only LoRA
# fine-tunes it is compared with. Copy of run_ft_seeds_base.sh for one config;
# runs on fierce.
set -uo pipefail
cd /data/learn/ssl-kaldi/egs/grid/s5 || exit 1
export PATH=/data/miniconda3/envs/ssl-kaldi/bin:$PATH
export STORE=/srv/nfs/data/ssl-kaldi-feats-xpaper
. ./cmd.sh; . ./path.sh
need_space() {
  local free=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
  [ $free -ge 10 ] || { echo "$(date +%H:%M) only ${free}G free on /data, stopping before $1"; exit 1; }
  free=$(df --output=avail -BG /srv/nfs/data | tail -1 | tr -dc 0-9)
  [ $free -ge 10 ] || { echo "$(date +%H:%M) only ${free}G free on /srv/nfs/data, stopping before $1"; exit 1; }
}

cfg=base_ctc_lora_av; tag=lor10av; ckpt=input/avhubert_$cfg.pt; ft=finetune_av
# Same clips and validation split as finetune_lora/manifest, plus the audio.
if [ ! -s $ft/manifest/train.tsv ]; then
  python local/finetune/prepare_ctc_manifests.py --data data/train_lor10_raw \
    --roi-dir finetune_lora/rois96 --out $ft/manifest --audio-dir /srv/nfs/data/grid16k_true \
    || { echo "FAILED manifests"; exit 1; }
  cmp -s <(cut -f1,2,4 finetune_lora/manifest/train.tsv) <(cut -f1,2,4 $ft/manifest/train.tsv) \
    || { echo "FAILED: manifest clips differ from finetune_lora"; exit 1; }
fi

if [ ! -f $ckpt ]; then
  need_space $cfg
  echo "$(date +%H:%M) fine-tuning $cfg"
  ./run_ctc_finetune.sh --config $cfg --ft-dir $ft --stage 2 --stop-stage 3 \
    > logs_ftav_$cfg.log 2>&1 || { echo "FAILED fine-tune $cfg"; exit 1; }
  [ -f $ckpt ] || { echo "FAILED export $cfg"; exit 1; }
  grep -o '"valid_wer": "[0-9.]*"' $ft/exp_$cfg/hydra_train.log | sort -t'"' -k4 -g | head -1
  rm -rf $ft/exp_$cfg/checkpoints
fi
c=exp/chain/e2e_tdnnf_${tag}norelu_cmvn
if [ ! -f $c/final.mdl ]; then
  need_space $tag
  hostname > exp/chain/.claim_$tag
  FEATDIM=768 ./run_ft_feats_arm.sh 10 $ckpt $tag > logs_ftav_$tag.log 2>&1 \
    || { echo "FAILED chain $tag"; exit 1; }
  echo "$(date +%H:%M) done $tag: $(grep SUMMARY logs_ftav_$tag.log)"
fi
export EXTRA_CONFIGS="$tag:$ckpt:10"
export EXTRA_CHAINS="e2e_tdnnf_${tag}norelu_cmvn:$tag:exp/chain/e2e_tree_ft/graph"
ONLY_TAGS=$tag STAGES="1 2" SETS=test ./run_rescore_r4000_lomb25.sh >> logs_ftav_decode.log 2>&1 \
  || { echo "FAILED r4000 decode $tag"; exit 1; }
ONLY_TAGS=$tag STAGES="1 2" ./run_shared_lombard.sh >> logs_ftav_decode.log 2>&1 \
  || { echo "FAILED Lombard decode $tag"; exit 1; }
rm -rf $STORE/ssl-kaldi-grid-$tag/train && echo "$(date +%H:%M) dropped train feats of $tag"
echo "Lombard speech: $(local/score_lombard_split.sh $c/decode_s_lomb_l data/lang/words.txt data/s_lomb_l_${tag}_raw)"
echo "plain speech:   $(local/score_lombard_split.sh $c/decode_s_lomb_p data/lang/words.txt data/s_lomb_p_${tag}_raw)"
pooled=$(local/score_lombard_pooled.sh $c/decode_s_lomb_l $c/decode_s_lomb_p)
echo "pooled 5,142:   $pooled"
# Score GRID at the LM setting chosen on the Lombard dev speakers.
echo "GRID 4,000:     $(grep WER $c/decode_r_test/wer_${pooled##*setting=})"
echo FT-AV-DONE
