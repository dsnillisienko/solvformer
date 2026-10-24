from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class Config:
    vocab: int = 16000
    d: int = 512
    heads: int = 8
    n_enc: int = 6
    n_dec: int = 6
    ff_mult: int = 4
    max_src: int = 192
    max_tgt: int = 160
    ffn: str = 'gelu'

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h = cfg.heads
        self.dh = cfg.d // cfg.heads
        self.q = nn.Linear(cfg.d, cfg.d, bias=False)
        self.kv = nn.Linear(cfg.d, 2 * cfg.d, bias=False)
        self.o = nn.Linear(cfg.d, cfg.d, bias=False)

    def heads_q(self, x):
        B, T, _ = x.shape
        return self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)

    def heads_kv(self, src):
        B, S, _ = src.shape
        return self.kv(src).view(B, S, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)

    def merge(self, y):
        B, _, T, _ = y.shape
        return self.o(y.transpose(1, 2).reshape(B, T, self.h * self.dh))

    def forward(self, x, mem=None, mask=None, causal=False):
        q = self.heads_q(x)
        k, v = self.heads_kv(x if mem is None else mem)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal)
        return self.merge(y)

class FFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.swiglu = cfg.ffn == 'swiglu'
        if self.swiglu:
            hidden = round(2 / 3 * cfg.ff_mult * cfg.d / 64) * 64
            self.up = nn.Linear(cfg.d, 2 * hidden, bias=False)
        else:
            hidden = cfg.ff_mult * cfg.d
            self.up = nn.Linear(cfg.d, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d, bias=False)

    def forward(self, x):
        if self.swiglu:
            gate, value = self.up(x).chunk(2, dim=-1)
            return self.down(F.silu(gate) * value)
        return self.down(F.gelu(self.up(x)))


class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.attn = nn.LayerNorm(cfg.d), Attention(cfg)
        self.n2, self.ffn = nn.LayerNorm(cfg.d), FFN(cfg)

    def forward(self, x, mask):
        x = x + self.attn(self.n1(x), mask=mask)
        x = x + self.ffn(self.n2(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.self_attn = nn.LayerNorm(cfg.d), Attention(cfg)
        self.n2, self.cross = nn.LayerNorm(cfg.d), Attention(cfg)
        self.n3, self.ffn = nn.LayerNorm(cfg.d), FFN(cfg)

    def forward(self, x, mem, mem_mask):
        x = x + self.self_attn(self.n1(x), causal=True)
        x = x + self.cross(self.n2(x), mem=mem, mask=mem_mask)
        x = x + self.ffn(self.n3(x))
        return x

    def step(self, x, t, cache_k, cache_v, cross_k, cross_v, mem_mask):
        """Один шаг генерации для нового токена x [B, 1, d] на позиции t.

        cache_k/cache_v [B, h, max_new, dh] хранят K и V уже сгенерированных токенов этого слоя;
        cross_k/cross_v [B, h, S, dh] посчитаны по выходу энкодера один раз на весь вопрос.
        """
        a = self.self_attn
        h = self.n1(x)
        k, v = a.heads_kv(h)
        cache_k[:, :, t:t + 1] = k
        cache_v[:, :, t:t + 1] = v
        y = F.scaled_dot_product_attention(a.heads_q(h), cache_k[:, :, :t + 1], cache_v[:, :, :t + 1])
        x = x + a.merge(y)

        c = self.cross
        y = F.scaled_dot_product_attention(c.heads_q(self.n2(x)), cross_k, cross_v, attn_mask=mem_mask)
        x = x + c.merge(y)
        return x + self.ffn(self.n3(x))


PAD, BOS, EOS = 0, 1, 2


def sample(logits, top_p=1.0, generator=None):
    """Один токен на строку из softmax(logits); top-p оставляет минимальный набор токенов с суммой вероятностей ≥ p."""
    probs = torch.softmax(logits, -1)
    if top_p < 1.0:
        sp, si = probs.sort(-1, descending=True)
        drop = sp.cumsum(-1) - sp > top_p
        sp = sp.masked_fill(drop, 0.0)
        probs = torch.zeros_like(probs).scatter_(-1, si, sp)
    return torch.multinomial(probs, 1, generator=generator)[:, 0]

class MishaGustokashin(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d)
        self.pos_src = nn.Embedding(cfg.max_src, cfg.d)
        self.pos_tgt = nn.Embedding(cfg.max_tgt, cfg.d)
        self.enc = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_enc)])
        self.dec = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_dec)])
        self.enc_norm = nn.LayerNorm(cfg.d)
        self.dec_norm = nn.LayerNorm(cfg.d)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def encode(self, src):
        mask = (src != PAD)[:, None, None, :]
        pos = torch.arange(src.shape[1], device=src.device)
        x = self.emb(src) + self.pos_src(pos)

        for layer in self.enc:
            x = layer(x, mask)

        return self.enc_norm(x), mask

    def forward(self, src, tgt_in):
        mem, mask = self.encode(src)
        pos = torch.arange(tgt_in.shape[1], device=tgt_in.device)
        y = self.emb(tgt_in) + self.pos_tgt(pos)

        for layer in self.dec:
            y = layer(y, mem, mask)

        return self.logits(y)

    def logits(self, y):
        return F.linear(self.dec_norm(y), self.emb.weight)

    @torch.no_grad()
    def generate(self, src, max_new=96, temperature=0.0, top_p=1.0, return_logprob=False, generator=None):
        """Генерация с KV-cache: src [B, S] -> id кода [B, <= max_new], после <eos> идёт PAD.

        При temperature = 0 выбор жадный (argmax), при temperature > 0 токены сэмплируются из softmax(logits / T),
        по желанию с top-p. С return_logprob=True возвращается ещё сумма log p выбранных токенов по каждой строке,
        то есть уверенность модели в программе.
        """
        B = src.shape[0]
        mem, mask = self.encode(src)
        cross = [layer.cross.heads_kv(mem) for layer in self.dec]
        k0 = cross[0][0]
        shape = (B, self.cfg.heads, max_new, self.cfg.d // self.cfg.heads)
        caches = [(k0.new_zeros(shape), k0.new_zeros(shape)) for _ in self.dec]

        tok = torch.full((B,), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        out = []
        logprob = torch.zeros(B, device=src.device)
        for t in range(max_new):
            x = (self.emb(tok) + self.pos_tgt.weight[t])[:, None]
            for layer, (ck, cv), (xk, xv) in zip(self.dec, caches, cross):
                x = layer.step(x, t, ck, cv, xk, xv, mask)
            logits = self.logits(x)[:, -1].float()
            if temperature > 0:
                nxt = sample(logits / temperature, top_p, generator)
            else:
                nxt = logits.argmax(-1)
            if return_logprob:
                lp = torch.log_softmax(logits, -1).gather(1, nxt[:, None])[:, 0]
                logprob += lp.masked_fill(done, 0.0)
            nxt = nxt.masked_fill(done, PAD)
            out.append(nxt)
            done |= nxt == EOS
            tok = nxt
            if bool(done.all()):
                break
        ids = torch.stack(out, 1)
        return (ids, logprob) if return_logprob else ids

    @torch.no_grad()
    def generate_naive(self, src, max_new=96):
        """Эталон без кэша: на каждом шаге полный forward по всему префиксу. Только для проверки."""
        ys = torch.full((src.shape[0], 1), BOS, dtype=torch.long, device=src.device)
        for _ in range(max_new):
            nxt = self(src, ys)[:, -1].argmax(-1, keepdim=True)
            ys = torch.cat([ys, nxt], 1)
        return ys[:, 1:]
