import os
import torch
import torch.nn.functional as F

from tokenizer.tokenizer import Tokenizer
from model.train import TinyGPT  # reuse model definition

@torch.no_grad()
def sample(model, idx, max_new_tokens, temperature=1.0, top_k=50, eos_id=None):
    model.eval()
    device = next(model.parameters()).device

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(1e-8, temperature)

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (B,1)

        idx = torch.cat([idx, next_id], dim=1)

        if eos_id is not None and int(next_id.item()) == int(eos_id):
            break

    return idx

def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = TinyGPT(**cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok_model = os.environ.get("TOKENIZER_MODEL", "tokenizer/cricket_bpe.model")
    ckpt_path = os.environ.get("CKPT", "checkpoints/ckpt_step1000.pt")

    prompt = os.environ.get("PROMPT", "Over 18.6: Bumrah to Kohli, ")
    max_new = int(os.environ.get("MAX_NEW", "120"))
    temperature = float(os.environ.get("GEN_TEMP", "0.9"))
    top_k = int(os.environ.get("GEN_TOP_K", "50"))

    tokenizer = Tokenizer(tok_model)
    model, _ = load_checkpoint(ckpt_path, device)

    # Encode prompt
    ids = tokenizer.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    out = sample(model, x, max_new_tokens=max_new, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
    text = tokenizer.decode(out[0].tolist())
    print(text)

if __name__ == "__main__":
    main()
