#!/usr/bin/env python3
# Copyright  2026  Ibrahim Almajai
# Apache 2.0
"""Fine-tune an SSL encoder with a character CTC head, then save it for feature
extraction.

Port of egs/timit/s5/local/finetune/finetune_ctc.py to word-level data: the
Kaldi `text` holds words, so transcripts are tokenised to characters with `|`
as the word delimiter (the wav2vec 2.0 convention) and validation reports
greedy word error rate instead of phone error rate. Everything else -- LoRA,
layer freezing, the LR schedule, checkpoint layout -- matches the timit
version, so the two files stay diffable.

The output directory is a plain HuggingFace model directory, so the rest of
the recipe consumes it unchanged:

    shared/make_ssl.sh --ssl-model exp_ft/mms300m_lora32 --layer 14 data/train_raw

Usage:
    python local/finetune/finetune_ctc.py --ssl-model facebook/mms-300m \
        --lora-rank 32 --lora-alpha 64 \
        --train-dir data/train_clean_5 --valid-dir data/dev_clean_2 \
        --out-dir exp_ft/mms300m_lora32
"""
import argparse
import json
import logging
import os
import random
import re

import numpy as np
import torch
from kaldiio import ReadHelper
from transformers import AutoFeatureExtractor, AutoModelForCTC

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
# transformers' from_pretrained raises the ROOT logger level to WARNING, which
# would silence every info() below it (including the per-epoch lines). Setting
# this logger's own level makes it immune: getEffectiveLevel stops here instead
# of inheriting root's.
logger.setLevel(logging.INFO)

EXPECTED_SAMPLE_RATE = 16000
BLANK = 0
WORD_DELIM = "|"


def parse_args():
    p = argparse.ArgumentParser(
        description="CTC fine-tuning of an SSL encoder for Kaldi feature extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ssl-model", default="facebook/mms-300m")
    p.add_argument("--head-layer", type=int, default=0,
                   help="layer the CTC head reads: the encoder is truncated to "
                        "this many layers before training. 0 keeps every layer, "
                        "which lets you tap a middle layer at extraction time "
                        "and keep distance from the CTC-specialised top")
    p.add_argument("--freeze-layers", type=int, default=0,
                   help="freeze the bottom N transformer layers, along with the "
                        "feature projection and positional embedding, so the "
                        "early acoustic pathway stays pretrained. Must be BELOW "
                        "the layer you intend to extract, or fine-tuning cannot "
                        "change the extracted features at all")
    p.add_argument("--train-dir", default="data/train_clean_5")
    p.add_argument("--valid-dir", default="data/dev_clean_2")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=0,
                   help="stop early if validation WER has not improved for this "
                        "many consecutive epochs. 0 disables it. The saved "
                        "checkpoint is always the best epoch, so stopping early "
                        "never discards a better model")
    p.add_argument("--batch-utts", type=int, default=2,
                   help="LibriSpeech utterances run to 17s against TIMIT's 8, "
                        "so the default batch is smaller and --grad-accum "
                        "correspondingly larger than the timit version's 8/2")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--hold-frac", type=float, default=0.0,
                   help="fraction of updates to hold the LR at its peak after "
                        "warmup, before linear decay. 0.4 with --warmup-frac 0.1 "
                        "gives the tri-state schedule of the wav2vec 2.0 paper")
    p.add_argument("--freeze-updates", type=int, default=0,
                   help="train only the CTC output layer for this many optimizer "
                        "steps before unfreezing the encoder (the paper uses "
                        "10k of 20k updates)")
    p.add_argument("--mask-time-prob", type=float, default=None,
                   help="SpecAugment time-mask probability; None keeps the "
                        "checkpoint default (usually far milder than the value "
                        "used for fine-tuning)")
    p.add_argument("--mask-time-length", type=int, default=10)
    p.add_argument("--mask-feature-prob", type=float, default=None,
                   help="SpecAugment channel-mask probability")
    p.add_argument("--mask-feature-length", type=int, default=64)
    p.add_argument("--layerdrop", type=float, default=None)
    p.add_argument("--normalize-waveform", choices=["true", "false"], default=None,
                   help="override the checkpoint's feature-extractor "
                        "do_normalize. Checkpoints disagree: hubert-base-ls960 "
                        "and mms-300m ship True, wavlm-base ships False, so a "
                        "cross-model comparison has to set this explicitly")
    p.add_argument("--attention-mask", choices=["true", "false"], default=None,
                   help="override return_attention_mask. Models with "
                        "feat_extract_norm='group' were pretrained on "
                        "zero-padded batches without a mask, but wavlm-base "
                        "ships True anyway")
    p.add_argument("--lora-rank", type=int, default=0,
                   help="if >0, adapt every layer with LoRA of this rank instead "
                        "of full-rank updates on a few layers. Needs peft. The "
                        "CTC head still trains normally, and the adapters are "
                        "merged into the encoder before saving so the output is "
                        "still a plain HuggingFace directory")
    p.add_argument("--lora-ffn", action="store_true",
                   help="also adapt the feed-forward projections, not just the "
                        "attention ones")
    p.add_argument("--train-layer-norms", action="store_true",
                   help="train the encoder-block layer norms at full rank "
                        "alongside the adapters; LoRA freezes them by default")
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--curriculum-epochs", type=int, default=0,
                   help="iterate batches shortest-first (no shuffle) for this "
                        "many initial epochs. Mini has no short-target "
                        "utterances to bootstrap CTC alignment on (torgo's "
                        "single-word utts serve that role), so start from the "
                        "easiest examples instead of a random order")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def read_data(data_dir):
    """Returns (utt_ids, waveforms, char transcripts) for a Kaldi data dir.

    Transcripts come back as character-token lists with WORD_DELIM between
    words: "THE CAT" -> ['T','H','E','|','C','A','T'].
    """
    with open(os.path.join(data_dir, "text")) as f:
        text = dict(line.strip().split(None, 1) for line in f if line.strip())

    utts, waves, trans = [], [], []
    with ReadHelper(f"scp:{os.path.join(data_dir, 'wav.scp')}") as reader:
        for utt, (rate, wav) in reader:
            if rate != EXPECTED_SAMPLE_RATE:
                raise ValueError(f"{utt}: {rate} != {EXPECTED_SAMPLE_RATE}")
            if utt not in text:
                logger.warning("no transcript for %s, skipping", utt)
                continue
            if np.issubdtype(wav.dtype, np.integer):
                wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
            utts.append(utt)
            waves.append(wav.astype(np.float32))
            trans.append(list(text[utt].replace(" ", WORD_DELIM)))
    logger.info("%s: %d utterances", data_dir, len(utts))
    return utts, waves, trans


def build_vocab(transcripts):
    """Character -> id. Index 0 is reserved for the CTC blank."""
    chars = sorted({c for t in transcripts for c in t})
    vocab = {c: i + 1 for i, c in enumerate(chars)}
    logger.info("vocab: %d characters + blank", len(chars))
    return vocab


def _in_frozen_layer(name, freeze_layers):
    """True if a parameter sits in one of the first freeze_layers encoder layers."""
    if not freeze_layers:
        return False
    m = re.search(r"\.layers\.(\d+)\.", name)
    return m is not None and int(m.group(1)) < freeze_layers


def make_batches(waves, batch_utts):
    """Length-sorted batches of indices, so padding stays small."""
    order = sorted(range(len(waves)), key=lambda i: len(waves[i]))
    return [order[i:i + batch_utts] for i in range(0, len(order), batch_utts)]


def collate(idx, waves, transcripts, vocab, extractor, device):
    batch = extractor(
        [waves[i] for i in idx],
        sampling_rate=EXPECTED_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    labels = [torch.tensor([vocab[c] for c in transcripts[i]]) for i in idx]
    maxlen = max(len(x) for x in labels)
    # -100 is ignored by the CTC loss when computing target lengths.
    padded = torch.full((len(labels), maxlen), -100, dtype=torch.long)
    for j, x in enumerate(labels):
        padded[j, :len(x)] = x
    batch = {k: v.to(device) for k, v in batch.items()}
    batch["labels"] = padded.to(device)
    return batch


def levenshtein(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def chars_to_words(chars):
    """['T','H','E','|','C','A','T'] -> ['THE', 'CAT']"""
    return [w for w in "".join(chars).split(WORD_DELIM) if w]


@torch.no_grad()
def validate(model, batches, waves, transcripts, vocab, extractor, device):
    """Greedy CTC word error rate, for model selection only."""
    model.eval()
    inv = {i: c for c, i in vocab.items()}
    errs = length = 0
    for idx in batches:
        batch = collate(idx, waves, transcripts, vocab, extractor, device)
        labels = batch.pop("labels")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**batch).logits
        pred = logits.argmax(-1).cpu()
        for row, i in enumerate(idx):
            ids, prev = [], None
            for t in pred[row].tolist():
                if t != prev and t != BLANK:
                    ids.append(t)
                prev = t
            hyp = chars_to_words([inv[t] for t in ids if t in inv])
            ref = chars_to_words(transcripts[i])
            errs += levenshtein(ref, hyp)
            length += len(ref)
        del labels
    model.train()
    return 100.0 * errs / max(length, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("fine-tuning needs a GPU")

    tr_utts, tr_waves, tr_trans = read_data(args.train_dir)
    va_utts, va_waves, va_trans = read_data(args.valid_dir)
    vocab = build_vocab(tr_trans)
    unseen = {c for t in va_trans for c in t} - set(vocab)
    if unseen:
        logger.warning("characters in valid but not train (ignored in WER): %s",
                       unseen)

    extractor = AutoFeatureExtractor.from_pretrained(args.ssl_model)
    # The extractor is saved alongside the encoder, so these carry through to
    # score_ctc.py and to Kaldi feature extraction.
    for opt, attr in (("normalize_waveform", "do_normalize"),
                      ("attention_mask", "return_attention_mask")):
        val = getattr(args, opt)
        if val is not None:
            setattr(extractor, attr, val == "true")
            logger.info("feature extractor override: %s=%s", attr, val)
    overrides = {}
    if args.mask_time_prob is not None:
        overrides["mask_time_prob"] = args.mask_time_prob
        overrides["mask_time_length"] = args.mask_time_length
    if args.mask_feature_prob is not None:
        overrides["mask_feature_prob"] = args.mask_feature_prob
        overrides["mask_feature_length"] = args.mask_feature_length
    if args.layerdrop is not None:
        overrides["layerdrop"] = args.layerdrop
    model = AutoModelForCTC.from_pretrained(
        args.ssl_model,
        vocab_size=len(vocab) + 1,
        pad_token_id=BLANK,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        **overrides,
    )
    if overrides:
        logger.info("config overrides: %s", overrides)

    # AutoModelForCTC gives Wav2Vec2ForCTC / HubertForCTC / etc; the base
    # encoder hangs off an architecture-specific attribute name.
    base = getattr(model, model.base_model_prefix)
    total_layers = len(base.encoder.layers)
    if args.head_layer:
        if not 1 <= args.head_layer <= total_layers:
            raise SystemExit(
                f"--head-layer {args.head_layer} out of range (1..{total_layers})")
        # Truncating puts the CTC head on the extracted layer. Keep the config in
        # step or the saved directory would reload with random top layers.
        base.encoder.layers = base.encoder.layers[:args.head_layer]
        model.config.num_hidden_layers = args.head_layer
        base.config.num_hidden_layers = args.head_layer
        logger.info("truncated encoder from %d to %d layers; CTC head reads "
                    "layer %d", total_layers, args.head_layer, args.head_layer)
    else:
        logger.info("keeping all %d layers; CTC head reads the top layer, so "
                    "extraction can tap a middle layer", total_layers)

    model.freeze_feature_encoder()

    if args.freeze_layers:
        kept = len(base.encoder.layers)
        if args.freeze_layers >= kept:
            raise SystemExit(
                f"--freeze-layers {args.freeze_layers} leaves nothing trainable "
                f"below the head ({kept} layers kept)")
        # Freeze the projection and positional embedding too: they sit below
        # layer 1, so leaving them trainable would change the input to the
        # "frozen" layers and defeat the point.
        for mod in (base.feature_projection, base.encoder.pos_conv_embed):
            for p in mod.parameters():
                p.requires_grad = False
        for layer in base.encoder.layers[:args.freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False
        logger.info("froze feature projection, positional embedding and encoder "
                    "layers 1-%d; layers %d-%d trainable",
                    args.freeze_layers, args.freeze_layers + 1, kept)

    if args.lora_rank:
        from peft import LoraConfig, get_peft_model
        # Adapt the attention projections with low-rank deltas. peft freezes
        # everything else, so re-enable the CTC head: it is randomly initialised
        # and has to train. --freeze-layers restricts which layers get adapters
        # at all, otherwise peft would adapt the "frozen" ones too.
        kept = len(base.encoder.layers)
        lora_kwargs = {}
        if args.freeze_layers:
            lora_kwargs["layers_to_transform"] = list(range(args.freeze_layers,
                                                            kept))
            lora_kwargs["layers_pattern"] = "layers"
        targets = ["q_proj", "k_proj", "v_proj", "out_proj"]
        # WavLM conditions its attention on a gated relative position bias that
        # HuBERT and wav2vec2 do not have. Leaving it frozen while the
        # projections move costs ~10 PER.
        gated = any(n.endswith("gru_rel_pos_linear")
                    for n, _ in base.named_modules())
        if gated:
            targets.append("gru_rel_pos_linear")
        if args.lora_ffn:
            targets += ["intermediate_dense", "output_dense"]
        model = get_peft_model(model, LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, bias="none",
            target_modules=targets,
            **lora_kwargs,
        ))
        for p in model.base_model.model.lm_head.parameters():
            p.requires_grad = True
        if gated:
            # The other two halves of that bias are an Embedding and a bare
            # Parameter, neither of which LoRA can wrap. They live only in the
            # first layer and total a few thousand values, so train them at
            # full rank, and only when that layer is not frozen.
            n_gate = 0
            for name, p in model.named_parameters():
                if ("rel_attn_embed" in name or "gru_rel_pos_const" in name) \
                        and not _in_frozen_layer(name, args.freeze_layers):
                    p.requires_grad = True
                    n_gate += p.numel()
            logger.info("training rel_attn_embed and gru_rel_pos_const at full "
                        "rank (%d values)", n_gate)
            # WavLM hands the gated bias to attention as attn_mask. Once it
            # requires grad the fused SDPA backward fails with "LSE is not
            # correctly aligned (strideH)", so fall back to the math kernel.
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            logger.info("disabled fused SDPA kernels (trainable attention bias)")
        if args.train_layer_norms:
            n_ln = 0
            for name, p in model.named_parameters():
                if "encoder.layers." in name and "layer_norm" in name \
                        and not _in_frozen_layer(name, args.freeze_layers):
                    p.requires_grad = True
                    n_ln += p.numel()
            logger.info("training encoder-block layer norms at full rank "
                        "(%d values)", n_ln)
        logger.info("LoRA rank %d (alpha %d, dropout %.2f) on %s of "
                    "layers %s", args.lora_rank, args.lora_alpha,
                    args.lora_dropout, "/".join(targets),
                    f"{args.freeze_layers + 1}-{kept}" if args.freeze_layers
                    else f"1-{kept} (all)")

    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_fr = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info("trainable %.1fM / frozen %.1fM (%.1f%% frozen)",
                n_tr / 1e6, n_fr / 1e6, 100.0 * n_fr / (n_tr + n_fr))

    model = model.to(device)
    model.train()

    train_batches = make_batches(tr_waves, args.batch_utts)
    valid_batches = make_batches(va_waves, args.batch_utts)

    steps_per_epoch = (len(train_batches) + args.grad_accum - 1) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs

    # Optimizer must own the encoder params even while they are temporarily
    # frozen for --freeze-updates, or they would never be optimized after the
    # unfreeze. Params with requires_grad False simply get no grad and are
    # skipped until then.
    held_out = []
    if args.freeze_updates:
        held_out = [p for p in base.parameters() if p.requires_grad]
        for p in held_out:
            p.requires_grad = False
        logger.info("training only the CTC output layer for the first %d "
                    "updates (%.0f%% of %d)", args.freeze_updates,
                    100.0 * args.freeze_updates / max(1, total_steps), total_steps)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + held_out,
        lr=args.lr, weight_decay=0.005,
    )
    warmup = max(1, int(args.warmup_frac * total_steps))
    hold = int(args.hold_frac * total_steps)

    def lr_at(step):
        """Tri-state when --hold-frac > 0: warmup, hold, linear decay."""
        if step < warmup:
            return step / warmup
        if step < warmup + hold:
            return 1.0
        decay_steps = max(1, total_steps - warmup - hold)
        return max(0.0, (total_steps - step) / decay_steps)

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_at)
    scaler = torch.amp.GradScaler("cuda")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "chars.json"), "w") as f:
        json.dump(vocab, f, indent=2, sort_keys=True)

    best, stale = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        if epoch > args.curriculum_epochs:
            random.shuffle(train_batches)
        else:
            train_batches.sort(
                key=lambda idx: min(len(tr_waves[i]) for i in idx))
        running, nb = 0.0, 0
        optim.zero_grad(set_to_none=True)
        for bi, idx in enumerate(train_batches):
            batch = collate(idx, tr_waves, tr_trans, vocab, extractor, device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / args.grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * args.grad_accum
            nb += 1
            if (bi + 1) % args.grad_accum == 0 or bi + 1 == len(train_batches):
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()
                if held_out and sched.last_epoch == args.freeze_updates:
                    for p in held_out:
                        p.requires_grad = True
                    logger.info("unfroze the encoder at update %d",
                                sched.last_epoch)

        wer = validate(model, valid_batches, va_waves, va_trans, vocab,
                       extractor, device)
        logger.info("epoch %d/%d  loss %.4f  valid WER %.2f%%  lr %.2e",
                    epoch, args.epochs, running / max(nb, 1), wer,
                    sched.get_last_lr()[0])

        if wer < best:
            best, stale = wer, 0
            # Save the encoder only: AutoModel/AutoFeatureExtractor read this
            # directory directly in shared/compute_ssl_feats.py.
            if args.lora_rank:
                # merge_and_unload() folds the deltas back into plain Linear
                # layers, which is what AutoModel can reload. It destroys the
                # adapter structure, so do it on a throwaway copy and keep
                # training the original.
                import copy
                merged = copy.deepcopy(model).merge_and_unload()
                save_enc = getattr(merged, merged.base_model_prefix)
                save_ctc = merged
            else:
                save_enc, save_ctc = base, model
            save_enc.save_pretrained(args.out_dir)
            extractor.save_pretrained(args.out_dir)
            # The encoder alone is what feature extraction needs, but keep the
            # full CTC model too so the fine-tuned system can be decoded
            # directly (local/finetune/score_ctc.py) without retraining.
            save_ctc.save_pretrained(os.path.join(args.out_dir, "ctc"))
            extractor.save_pretrained(os.path.join(args.out_dir, "ctc"))
            if args.lora_rank:
                del merged, save_enc, save_ctc
                torch.cuda.empty_cache()
            logger.info("saved %s (valid WER %.2f%%)", args.out_dir, wer)
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                logger.info("no improvement for %d epoch(s) (best %.2f%%); "
                            "stopping early at epoch %d/%d",
                            stale, best, epoch, args.epochs)
                break

    logger.info("done; best valid WER %.2f%%, encoder in %s", best, args.out_dir)


if __name__ == "__main__":
    main()
