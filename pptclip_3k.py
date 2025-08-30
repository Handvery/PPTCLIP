import os
from collections import defaultdict
from itertools import product

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.amp import autocast, GradScaler
import torch.nn.functional as F

import clip  # openai/clip
import numpy as np
import scipy.stats
from tqdm import tqdm

# ===== project‑specific utils =====
from utils import (
    set_dataset_aigc_3k, _preprocess2, _preprocess3,
    convert_models_to_fp32, get_logger, log_and_print,
)
from MNL_Loss import loss_m3
from easy_model import QualityWeightFlatten1

# ---------- 相关性损失 ----------
def corr_loss(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Spearman-like correlation loss to maximize correlation."""
    p = (preds - preds.mean()) / (preds.std(unbiased=False) + 1e-6)
    t = (target - target.mean()) / (target.std(unbiased=False) + 1e-6)
    return 1.0 - torch.mean(p * t)

# ---------------------------------------------------
# ---------------- Hyper‑parameters -----------------
# ---------------------------------------------------
checkpoint_dir = "3k_perception/10"
os.makedirs(checkpoint_dir, exist_ok=True)

content_quality_words = ["badly", "poorly", "fairly", "well", "perfectly"]
pure_quality_words = ["low", "poor", "medium", "good", "excellent"]
n_ctx = 6

# training params
device         = "cuda:4" if torch.cuda.is_available() else "cpu"
freeze_epochs  = 5
partial_gap    = 3
clip_base_lr   = 2e-5
prompt_lr      = 5e-4
head_lr        = 5e-4
num_epoch      = 30
bs             = 32
train_patch    = 9
num_workers    = 8
max_grad_norm  = 0.5
now_session    = 4

preprocess2 = _preprocess2()
preprocess3 = _preprocess3()

eos_id = 49407  # CLIP EOS token ID

class PromptLearner(nn.Module):
    def __init__(self, clip_model, content_words, pure_words, n_ctx=6):
        super().__init__()
        self.clip_model = clip_model
        d_model = clip_model.token_embedding.weight.shape[1]
        self.ctx = nn.Parameter(torch.randn(n_ctx, d_model) * 0.02)
        self.content_words = content_words
        self.pure_words = pure_words

        # 预compute pure template tokens + EOS positions
        pure_template = ("X " * n_ctx + "a photo with {} quality").strip()
        pure_tok = [clip.tokenize(pure_template.format(w), truncate=True) for w in pure_words]
        pure_tokenized = torch.cat(pure_tok, dim=0)
        self.register_buffer("pure_tokenized", pure_tokenized)
        self.register_buffer("pure_eos_pos", (pure_tokenized == eos_id).float().argmax(dim=-1))

    def forward(self, prompts):
        batch_size = len(prompts)
        # 构造文本列表
        content_texts = [f"a photo that {w} matches '{p}'"
                         for p in prompts for w in self.content_words]
        pure_texts = [f"a photo with {w} quality" for _ in prompts for w in self.pure_words]

        all_texts = content_texts + pure_texts
        tokenized = torch.cat([clip.tokenize(t, truncate=True) for t in all_texts], dim=0).to(device)

        # 计算 EOS 位置
        content_count = len(content_texts)
        content_tok = tokenized[:content_count]
        pure_tok = tokenized[content_count:]
        content_eos_pos = (content_tok == eos_id).float().argmax(dim=-1)
        pure_eos_pos = (pure_tok == eos_id).float().argmax(dim=-1)
        eos_positions = torch.cat([content_eos_pos, pure_eos_pos], dim=0)

        # 文本嵌入 + 插入 learnable ctx
        tok_emb = self.clip_model.token_embedding(tokenized)  # [M, 77, D]
        n_ctx = self.ctx.size(0)
        tok_emb[:, 1:1+n_ctx] = self.ctx.unsqueeze(0)

        x = tok_emb + self.clip_model.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x)

        # 取 EOS 特征
        eos = x[torch.arange(x.size(0), device=device), eos_positions]
        eos = eos @ self.clip_model.text_projection

        # 分离并 reshape
        content_feats = eos[:content_count].view(batch_size, len(self.content_words), -1)
        pure_feats = eos[content_count:].view(batch_size, len(self.pure_words), -1)
        all_feats = torch.cat([content_feats, pure_feats], dim=1)
        return all_feats / all_feats.norm(dim=-1, keepdim=True)

class CustomCLIP(nn.Module):
    def __init__(self, clip_model, content_words, pure_words, n_ctx):
        super().__init__()
        self.clip = clip_model
        self.prompt_learner = PromptLearner(clip_model, content_words, pure_words, n_ctx)
        self.quality_head = QualityWeightFlatten1(num_patch=train_patch, in_dim=len(content_words)+len(pure_words))
        self.logit_scale = clip_model.logit_scale
        self.logit_scale.requires_grad = False

    def encode_image(self, x):
        z = self.clip.encode_image(x)
        return z / z.norm(dim=-1, keepdim=True)

    def encode_text(self, prompts):
        return self.prompt_learner(prompts)


def forward_iqa(model, images, prompts):
    B, N = images.shape[:2]
    img_f = model.encode_image(images.view(-1, *images.shape[2:]))
    txt_f = model.encode_text(prompts)
    txt_f = txt_f.unsqueeze(1).expand(-1, N, -1, -1).reshape(B*N, txt_f.size(1), -1)
    logits = model.logit_scale.exp() * torch.einsum('bd,bkd->bk', img_f, txt_f)
    logits = logits.view(B, N, -1)
    return model.quality_head(logits)

# 其余代码保持不变 (param_groups_lr, train_one_epoch, evaluate, main)
def param_groups_lr(model, base_lr=clip_base_lr, decay=0.85):
    seen, lr2params = {}, defaultdict(list)
    # layer-wise lr for visual transformer blocks
    for idx, blk in enumerate(model.clip.visual.transformer.resblocks):
        lr = base_lr * (decay ** (11 - idx))
        for p in blk.parameters():
            seen[id(p)] = lr
    # base lr for other clip parameters
    for name, p in model.clip.named_parameters():
        if id(p) not in seen:
            seen[id(p)] = base_lr
    # prompt learner and head lr
    for p in list(model.prompt_learner.parameters()):
        seen[id(p)] = prompt_lr
    for p in list(model.quality_head.parameters()):
        seen[id(p)] = head_lr
    # collect params by lr
    for p in list(model.clip.parameters()) + list(model.prompt_learner.parameters()) + list(model.quality_head.parameters()):
        lr2params[seen[id(p)]].append(p)
    return [{"params": params, "lr": lr} for lr, params in lr2params.items()]

def train_one_epoch(model, opt, loader, epoch, scaler):
    model.train()
    acc_loss = 0.0
    loop = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in loop:
        imgs = batch["I"].to(device)
        mos = batch["mos"].to(device).float()
        prompts = batch["prompt"]  
        
        opt.zero_grad()
        with autocast(device_type='cuda:4'):
            preds = forward_iqa(model, imgs, prompts)
         
            l1 = loss_m3(preds, mos).mean()
            l2 = corr_loss(preds, mos)
            loss = l1 
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(opt); scaler.update()
        convert_models_to_fp32(model.clip)

        acc_loss += loss.item()
        loop.set_postfix(loss = acc_loss / (loop.n + 1))
    return acc_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    q_mos, q_hat = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["I"].to(device)
            mos = batch["mos"].to(device).float()
            prompts = batch["prompt"]  
            
            preds = forward_iqa(model, imgs, prompts)
            q_mos.extend(mos.cpu()); q_hat.extend(preds.cpu())
    srcc = scipy.stats.mstats.spearmanr(q_mos, q_hat)[0]
    plcc = scipy.stats.pearsonr(q_mos, q_hat)[0]
    return (srcc + plcc) / 2, srcc, plcc


def main():
    log = get_logger(os.path.join(checkpoint_dir, "train.log"), "log")

    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False, download_root="./clip")
    # 使用两组提示词初始化
    model = CustomCLIP(clip_model, content_quality_words, pure_quality_words, n_ctx).to(device)

    root = "./Database/AGIQA-3K"; session = now_session
    train_loader = set_dataset_aigc_3k(os.path.join(root, str(session), "train.csv"), bs,
                                    "/home/data/wpy/IPCE-main/data/AGIQA-3K/file",
                                    num_workers, preprocess3, train_patch-1, False)
    val_loader   = set_dataset_aigc_3k(os.path.join(root, str(session), "val.csv"), bs,
                                    "/home/data/wpy/IPCE-main/data/AGIQA-3K/file",
                                    num_workers, preprocess2, train_patch-1, True)

    optimizer = torch.optim.AdamW(param_groups_lr(model), weight_decay=1e-3)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)
    scaler    = GradScaler()

    best = {"score":0}


    for epoch in range(num_epoch):
        # —— 冻结策略 ——
        if epoch < freeze_epochs:
            for p in model.clip.parameters(): p.requires_grad = False
        elif epoch == freeze_epochs:
            # 解冻最后两层 ViT
            for name, p in model.clip.named_parameters():
                if "visual.transformer.resblocks.10" in name or "visual.transformer.resblocks.11" in name:
                    p.requires_grad = True
            log_and_print(log, f"[Epoch {epoch}] unfroze last 2 ViT blocks.")
        elif epoch == freeze_epochs + partial_gap:
            for p in model.clip.parameters(): p.requires_grad = True  # 全网解冻
            # 缩小 Prompt/Head lr
            for group in optimizer.param_groups:
                if abs(group["lr"] - prompt_lr) < 1e-8:  # prompt lr
                    group["lr"] *= 0.1
                elif abs(group["lr"] - head_lr) < 1e-8:  # head lr
                    group["lr"] *= 0.1
            log_and_print(log, f"[Epoch {epoch}] unfroze full ViT & decayed prompt/head lr.")

        loss = train_one_epoch(model, optimizer, train_loader, epoch, scaler)
        scheduler.step()
        avg, srcc, plcc = evaluate(model, val_loader)
        log_and_print(log, f"Epoch {epoch}: loss={loss:.4f}, SRCC={srcc:.4f}, PLCC={plcc:.4f}")

        if avg > best["score"]:
            best.update(dict(score=avg, srcc=srcc, plcc=plcc, epoch=epoch))
            #torch.save({"model_state_dict": model.state_dict()},
            #          os.path.join(checkpoint_dir, "best_ckpt.pt"))
            log_and_print(log, "[New Best] model saved.")

    log_and_print(log, f"Finished. Best @ epoch {best['epoch']} — SRCC {best['srcc']:.4f}, PLCC {best['plcc']:.4f}")


if __name__ == "__main__":
    main()
