import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from tokenizer.tokenizer import Tokenizer
from model.dataset import CricketTextDataset

# -------------------------
# Tiny GPT model
# -------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.register_buffer("mask", None, persistent=False)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)  # (B,T,3C)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B,nh,T,hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,nh,T,T)

        # causal mask
        if self.mask is None or self.mask.size(-1) < T:
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            self.mask = mask
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B,nh,T,hd)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, n_embd, ffn_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.drop(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, ffn_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, ffn_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer=4, n_head=8, n_embd=256, ffn_dim=1024, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, ffn_dim, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # tie weights (helps small models)
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.block_size}")

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B,T,V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# -------------------------
# Train / Eval
# -------------------------

@torch.no_grad()
def estimate_loss(model, loader, device, max_batches=50):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))

def main():
    # Paths
    train_path = os.environ.get("TRAIN_PATH", "data/processed/train.txt")
    val_path = os.environ.get("VAL_PATH", "data/processed/val.txt")
    tok_model = os.environ.get("TOKENIZER_MODEL", "tokenizer/cricket_bpe.model")
    out_dir = os.environ.get("OUT_DIR", "checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    # Hyperparams
    block_size = int(os.environ.get("BLOCK_SIZE", "256"))
    batch_size = int(os.environ.get("BATCH_SIZE", "32"))
    lr = float(os.environ.get("LR", "3e-4"))
    max_steps = int(os.environ.get("MAX_STEPS", "20000"))
    eval_every = int(os.environ.get("EVAL_EVERY", "500"))
    save_every = int(os.environ.get("SAVE_EVERY", "1000"))
    grad_clip = float(os.environ.get("GRAD_CLIP", "1.0"))

    # Model size (tune within your 1–4M params)
    n_layer = int(os.environ.get("N_LAYER", "6"))
    n_head = int(os.environ.get("N_HEAD", "6"))
    n_embd = int(os.environ.get("N_EMBD", "384"))
    ffn_dim = int(os.environ.get("FFN_DIM", "1024"))
    dropout = float(os.environ.get("DROPOUT", "0.1"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tokenizer = Tokenizer(tok_model)
    vocab_size = tokenizer.vocab_size
    print("Vocab size:", vocab_size)

    train_ds = CricketTextDataset(train_path, tokenizer, block_size)
    val_ds = CricketTextDataset(val_path, tokenizer, block_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    model = TinyGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        ffn_dim=ffn_dim,
        dropout=dropout
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # Training loop (step-based)
    model.train()
    t0 = time.time()
    step = 0

    pbar = tqdm(total=max_steps)
    while step < max_steps:
        for x, y in train_loader:
            step += 1
            x, y = x.to(device), y.to(device)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                _, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            pbar.update(1)
            pbar.set_description(f"step {step} loss {loss.item():.4f}")

            if step % eval_every == 0:
                train_loss = estimate_loss(model, train_loader, device, max_batches=25)
                val_loss = estimate_loss(model, val_loader, device, max_batches=50)
                dt = time.time() - t0
                print(f"\nEval @ {step}: train {train_loss:.4f} | val {val_loss:.4f} | {dt/60:.1f} min\n")

            if step % save_every == 0:
                ckpt_path = os.path.join(out_dir, f"ckpt_step{step}.pt")
                torch.save({
                    "step": step,
                    "model_state": model.state_dict(),
                    "config": {
                        "vocab_size": vocab_size,
                        "block_size": block_size,
                        "n_layer": n_layer,
                        "n_head": n_head,
                        "n_embd": n_embd,
                        "ffn_dim": ffn_dim,
                        "dropout": dropout
                    }
                }, ckpt_path)
                print(f"Saved {ckpt_path}")

            if step >= max_steps:
                break

    pbar.close()
    print("Done.")

if __name__ == "__main__":
    main()
