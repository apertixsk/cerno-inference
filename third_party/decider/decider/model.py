"""Backbone -> slot hidden states -> restricted logits over option letters."""
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from decider.prompt import letter_ids, MAX_OPTIONS

ATTENTION = ("causal", "block")   # "block" (v1.2) = block-bidirectional full-attention layers (the context block, one block per question)


class DecisionModel(nn.Module):
    def __init__(self, name, dtype=torch.bfloat16, grad_ckpt=True, gc_reentrant=True, fsdp=False, attention="causal"):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(name)
        self.lm = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        self.fsdp = fsdp
        assert attention in ATTENTION, attention
        self.attention, self.dtype = attention, dtype
        if attention == "block":   # the per-layer-type mask dict below covers exactly these two layer types
            cfg = self.lm.model.config
            types = set(getattr(cfg, "layer_types", None) or [])
            assert types and types <= {"full_attention", "linear_attention"}, f"block attention supports full/linear-attention hybrids, got {types}"
            assert cfg._attn_implementation in ("sdpa", "eager", "flex_attention"), f"block attention needs a mask-based kernel, not {cfg._attn_implementation}"
        if grad_ckpt:
            if gc_reentrant:
                self.lm.gradient_checkpointing_enable()
            else:   # non-reentrant checkpointing when the model is sharded
                self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.register_buffer("letters", torch.tensor(letter_ids(self.tok)), persistent=False)

    def block_masks(self, input_ids, attention_mask, block_ids):
        """Per-layer-type masks for a full/linear-attention hybrid (block-bidirectional checkpoints, v1.2). Full-attention
        layers: causal, plus bidirectional within a block (transformers' block_sequence_ids; -1 = causal, used for padding and
        for rows without blocks). Linear-attention (recurrent) layers keep their causal scan and get the 2-D padding mask. The
        mask builders read only the batch size, length, dtype and device of `inputs_embeds`, so a zero-width stand-in is passed."""
        from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
        stub = torch.empty(input_ids.shape[0], input_ids.shape[1], 0, dtype=self.dtype, device=input_ids.device)
        kw = dict(config=self.lm.model.config, inputs_embeds=stub, attention_mask=attention_mask, past_key_values=None)
        return {"full_attention": create_causal_mask(**kw, block_sequence_ids=block_ids.to(input_ids.device)),
                "linear_attention": create_recurrent_attention_mask(**kw)}

    def slot_logits(self, input_ids, attention_mask, slot_idx, slot_batch, nopts, block_ids=None):
        """input_ids [B,T]; slot_idx/slot_batch [N] flat slot positions; nopts [N]; block_ids [B,T] or None.
        Returns [N, MAX_OPTIONS] logits with invalid options masked to -inf."""
        if self.attention == "block" and block_ids is not None:
            attention_mask = self.block_masks(input_ids, attention_mask, block_ids)
        h = self.lm.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hs = h[slot_batch, slot_idx]                                   # [N,H]
        if self.fsdp:   # sharded model: the head's parameters are only materialised inside its forward
            logits = self.lm.lm_head(hs).float()[:, self.letters]      # [N,K]
        else:
            W = self.lm.lm_head.weight[self.letters]                   # [K,H]
            logits = F.linear(hs, W).float()                           # [N,K]
        cap = getattr(getattr(self.lm, "config", None), "final_logit_softcapping", None)
        if cap:   # models pretrained with capped logits (cap * tanh(z / cap)) start the readout from the same scale
            logits = float(cap) * torch.tanh(logits / float(cap))
        ar = torch.arange(MAX_OPTIONS, device=logits.device)[None, :]
        logits = logits.masked_fill(ar >= nopts[:, None], float("-inf"))
        return logits

    def forward(self, batch):
        return self.slot_logits(batch["input_ids"], batch["attention_mask"], batch["slot_idx"], batch["slot_batch"], batch["nopts"],
                                batch.get("block_ids"))


def collate(items, pad_id):
    """items: list of dicts from prompt.build (+ 'task', 'ex_id'). Right-pad."""
    T = max(len(it["ids"]) for it in items)
    T = ((T + 63) // 64) * 64          # few distinct shapes -> fewer kernel (re)compiles
    B = len(items)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    slot_idx, slot_batch, golds, nopts, tasks, qidx, soft, has_soft = [], [], [], [], [], [], [], []
    with_blocks = any(it.get("blocks") is not None for it in items)
    block_ids = torch.full((B, T), -1, dtype=torch.long) if with_blocks else None   # -1 = causal (padding, rows without blocks)
    for b, it in enumerate(items):
        n = len(it["ids"])
        input_ids[b, :n] = torch.tensor(it["ids"])
        attn[b, :n] = 1
        if with_blocks and it.get("blocks") is not None:
            block_ids[b, :n] = torch.tensor(it["blocks"])
        for k, s in enumerate(it["slots"]):
            slot_idx.append(s); slot_batch.append(b); golds.append(it["golds"][k]); nopts.append(it["nopts"][k])
            tasks.append(it.get("task", "")); qidx.append(k)
            sv = (it.get("soft") or [None] * len(it["slots"]))[k]
            row = [0.0] * MAX_OPTIONS
            if sv: row[:len(sv)] = sv
            soft.append(row); has_soft.append(bool(sv))
    return dict(input_ids=input_ids, attention_mask=attn, slot_idx=torch.tensor(slot_idx), slot_batch=torch.tensor(slot_batch),
                golds=torch.tensor(golds), nopts=torch.tensor(nopts), tasks=tasks, qidx=qidx, soft=torch.tensor(soft), has_soft=torch.tensor(has_soft),
                block_ids=block_ids)
