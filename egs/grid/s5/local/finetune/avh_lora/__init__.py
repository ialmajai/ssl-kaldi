"""fairseq --user-dir plugin: LoRA CTC fine-tuning of AV-HuBERT.

Registers av_hubert_ctc_lora: the pretrained AV-HuBERT encoder (built and loaded
as AVHubertSeq2Seq does, frozen) with a CTC projection over
task.target_dictionary, plus LoRA on the listed Linear weights of every
transformer layer. LoRA is a weight parametrisation (W + B A * alpha/r), not a
module wrapper, because fairseq's MultiheadAttention fast path reads
q_proj.weight etc. directly. B starts at zero, so training starts exactly at
the pretrained encoder.

freeze_finetune_updates keeps the encoder (and so LoRA) under no_grad for the
first N updates, so the CTC head learns before the encoder moves.

full_finetune=true is the matched control: no LoRA, every encoder weight
(front end included) trainable, everything else identical.

Env: AVHUBERT_ROOT, the directory holding the `avhubert` package.
"""
import contextlib
import os
import sys
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn.utils import parametrize

_root = os.environ.get("AVHUBERT_ROOT")
if _root and _root not in sys.path:
    sys.path.insert(0, _root)

import avhubert  # noqa: F401,E402  registers stock models/criterions/task
from avhubert.hubert_asr import AVHubertCtc, AVHubertCtcConfig  # noqa: E402
from fairseq import checkpoint_utils, tasks  # noqa: E402
from fairseq.models import FairseqEncoder, register_model  # noqa: E402


class LoRA(nn.Module):
    """Parametrisation W -> W + (B @ A) * scale."""

    def __init__(self, out_dim, in_dim, rank, alpha):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.scale = alpha / rank

    def forward(self, W):
        return W + (self.B @ self.A).to(W.dtype) * self.scale


def add_lora(w2v_model, targets, rank, alpha):
    n = 0
    for layer in w2v_model.encoder.layers:
        mods = {"q_proj": layer.self_attn.q_proj, "k_proj": layer.self_attn.k_proj,
                "v_proj": layer.self_attn.v_proj, "out_proj": layer.self_attn.out_proj,
                "fc1": layer.fc1, "fc2": layer.fc2}
        for name in targets:
            lin = mods[name]
            out_dim, in_dim = lin.weight.shape
            parametrize.register_parametrization(lin, "weight", LoRA(out_dim, in_dim, rank, alpha))
            n += 1
    return n


def load_pretrained(cfg):
    """Build and load the pretrained AV-HuBERT, as AVHubertSeq2Seq.build_model does."""
    overrides = {"dropout": cfg.dropout, "activation_dropout": cfg.activation_dropout,
                 "dropout_input": cfg.dropout_input, "attention_dropout": cfg.attention_dropout,
                 "encoder_layerdrop": cfg.layerdrop, "feature_grad_mult": cfg.feature_grad_mult,
                 "mask_length": cfg.mask_length, "mask_prob": cfg.mask_prob,
                 "mask_selection": cfg.mask_selection, "mask_other": cfg.mask_other,
                 "no_mask_overlap": cfg.no_mask_overlap,
                 "mask_channel_length": cfg.mask_channel_length,
                 "mask_channel_prob": cfg.mask_channel_prob,
                 "mask_channel_selection": cfg.mask_channel_selection,
                 "mask_channel_other": cfg.mask_channel_other,
                 "no_mask_channel_overlap": cfg.no_mask_channel_overlap}
    state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, overrides)
    w2v_args = state["cfg"]
    cfg.w2v_args = w2v_args
    assert cfg.normalize == w2v_args.task.normalize
    w2v_args.task.data = cfg.data
    task_pretrain = tasks.setup_task(w2v_args.task)
    task_pretrain.load_state_dict(state["task_state"])
    model = task_pretrain.build_model(w2v_args.model)
    del state["model"]["mask_emb"]
    model.load_state_dict(state["model"], strict=False)
    model.remove_pretraining_modules()
    return model


class LoraCtcEncoder(FairseqEncoder):
    def __init__(self, cfg, tgt_dict):
        super().__init__(None)
        self.w2v_model = load_pretrained(cfg)
        for p in self.w2v_model.parameters():
            p.requires_grad = cfg.full_finetune
        self.n_lora = 0 if cfg.full_finetune else add_lora(
            self.w2v_model, cfg.lora_targets.split(","), cfg.lora_rank, cfg.lora_alpha)
        self.apply_mask = cfg.apply_mask
        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0
        self.proj = nn.Linear(self.w2v_model.encoder.embedding_dim, len(tgt_dict))
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def train(self, mode=True):
        # Keep the frozen ResNet front end in eval mode: BatchNorm running stats
        # update in train mode even with requires_grad=False, which would move
        # the "frozen" encoder away from the pretrained one.
        super().train(mode)
        for name in ("feature_extractor_video", "feature_extractor_audio"):
            fe = getattr(self.w2v_model, name, None)
            if fe is not None:
                fe.eval()
        return self

    def forward(self, source, padding_mask, tbc=True, **kwargs):
        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_finetune(
                source=source, padding_mask=padding_mask, mask=self.apply_mask and self.training)
            if tbc:
                x = x.transpose(0, 1)
        x = self.proj(self.final_dropout(x))
        return {"encoder_out": x, "encoder_padding_mask": padding_mask,
                "padding_mask": padding_mask}

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = \
                encoder_out["encoder_padding_mask"].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        return None


@dataclass
class AVHubertCtcLoraConfig(AVHubertCtcConfig):
    lora_rank: int = field(default=32, metadata={"help": "LoRA rank"})
    lora_alpha: float = field(default=32.0, metadata={"help": "LoRA alpha (scale = alpha/rank)"})
    lora_targets: str = field(default="q_proj,k_proj,v_proj,out_proj,fc1,fc2",
                              metadata={"help": "comma-separated Linear names per layer"})
    full_finetune: bool = field(default=False,
                                metadata={"help": "no LoRA; train every encoder weight (control)"})


@register_model("av_hubert_ctc_lora", dataclass=AVHubertCtcLoraConfig)
class AVHubertCtcLora(AVHubertCtc):
    @classmethod
    def build_model(cls, cfg, task):
        enc = LoraCtcEncoder(cfg, task.target_dictionary)
        model = cls(cfg, enc)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        lora = sum(p.numel() for n, p in model.named_parameters() if n.endswith((".A", ".B")))
        print(f"[avh_lora] {enc.n_lora} LoRA weights, rank {cfg.lora_rank}, alpha "
              f"{cfg.lora_alpha}; {trainable:,} trainable params ({lora:,} LoRA)", flush=True)
        return model
