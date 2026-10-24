"""Обучение MishaGustokashin.

    # отладка на маке
    python src/train.py --out runs/debug --shards 0-1 --steps 300 --d 128 --heads 4 --n_enc 2 --n_dec 2 --batch 64 --eval_every 100 --eval_n 100
    # Kaggle T4
    python src/train.py --out runs/base --shards 0-59 --steps 45000 --batch 256 --max_minutes 600
    # Kaggle T4 x2: каждая GPU берёт свой батч, эффективный батч = 2 × --batch
    torchrun --nproc_per_node 2 src/train.py --out runs/base --shards 0-59 --steps 48000 --batch 256
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.batches import Data
from src.execute import pass1
from src.generate import generate_codes, pick_device, save_checkpoint
from src.model import Config, MishaGustokashin


def parse_shards(spec):
    """'0-3,13' -> [0, 1, 2, 3, 13]"""
    out = []
    for part in spec.split(','):
        lo, _, hi = part.partition('-')
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def progress(step, elapsed, a):
    """Доля пройденного обучения по шагам или, если задан --max_minutes, по времени (что наступит раньше)."""
    t = step / a.steps
    if a.max_minutes:
        t = max(t, elapsed / (60 * a.max_minutes))
    return min(1.0, t)


def lr_at(step, prog, a):
    """Линейный warmup, затем cosine от lr до lr * min_lr_frac."""
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    w = a.warmup / a.steps
    t = min(1.0, max(0.0, (prog - w) / (1 - w)))
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def train_step(model, opt, scaler, src, tgt_in, tgt_out, use_fp16, clip):
    with torch.autocast('cuda', dtype=torch.float16, enabled=use_fp16):
        logits = model(src, tgt_in)
    loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), tgt_out.view(-1))
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    return loss, gn


def memory_check(model, opt, data, a, device, use_fp16):
    """Forward и backward на самой тяжёлой форме батча, которую допускает --max_tokens.

    Если памяти не хватит, это выяснится сразу, а не через час обучения."""
    n, Lq, Lc = data.worst_shape(a.batch, a.max_tokens, a.pad_to)
    V = model.cfg.vocab
    src = torch.randint(3, V, (n, Lq), device=device)
    tgt_in = torch.randint(3, V, (n, Lc), device=device)
    tgt_out = torch.randint(3, V, (n, Lc), device=device)
    torch.cuda.reset_peak_memory_stats()
    check_scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
    try:
        train_step(model, opt, check_scaler, src, tgt_in, tgt_out, use_fp16, a.clip)
    except torch.OutOfMemoryError:
        raise SystemExit(f'OOM на батче {n}×({Lq}+{Lc}): уменьши --max_tokens (сейчас {a.max_tokens})')
    opt.zero_grad(set_to_none=True)
    peak = torch.cuda.max_memory_allocated() / 2**30
    total = torch.cuda.get_device_properties(device).total_memory / 2**30
    print(f'memory check: батч {n}×({Lq}+{Lc}) → пик {peak:.1f} из {total:.1f} ГБ', flush=True)


def setup_distributed(a):
    """torchrun выставляет WORLD_SIZE, RANK и LOCAL_RANK; без него обучение идёт на одном устройстве."""
    world, rank, local = (int(os.environ.get(k, d)) for k, d in [('WORLD_SIZE', 1), ('RANK', 0), ('LOCAL_RANK', 0)])
    if world == 1:
        return 1, 0, pick_device(a.device)
    if torch.cuda.is_available() and a.device != 'cpu':
        dist.init_process_group('nccl')
        torch.cuda.set_device(local)
        return world, rank, torch.device('cuda', local)
    dist.init_process_group('gloo')
    return world, rank, torch.device('cpu')


def evaluate(model, tok, val, device, n):
    codes = generate_codes(model, tok, [v['question'] for v in val[:n]], device)
    return pass1(codes, [v['gold'] for v in val[:n]]), codes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--shards', default='0-59')

    p.add_argument('--d_model', '--d', dest='d', type=int, default=512)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--n_enc', type=int, default=6)
    p.add_argument('--n_dec', type=int, default=6)
    p.add_argument('--ffn', choices=['gelu', 'swiglu'], default='gelu')

    p.add_argument('--steps', type=int, default=45000)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--max_tokens', type=int, default=16384,
                   help='потолок «примеров × max(длина вопроса, длина кода)» на батч; длинные батчи меньше --batch')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--min_lr_frac', type=float, default=0.05)
    p.add_argument('--warmup', type=int, default=1000)
    p.add_argument('--wd', type=float, default=0.1)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--max_minutes', type=float, default=0, help='остановиться (докрутив cosine) к этому времени')

    p.add_argument('--eval_every', type=int, default=3000)
    p.add_argument('--eval_n', type=int, default=1000)
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--pad_to', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='auto')
    p.add_argument('--resume', type=Path, default=None, help='продолжить с last.pt')
    p.add_argument('--init_from', type=Path, default=None,
                   help='веса и архитектура из этого чекпойнта; оптимизатор и расписание LR начинаются заново')
    p.add_argument('--keep_ckpts', type=int, default=0,
                   help='хранить N последних чекпойнтов ckpt-<шаг>.pt для усреднения весов; 0 значит не хранить')
    a = p.parse_args()

    world, rank, device = setup_distributed(a)
    main_rank = rank == 0
    say = print if main_rank else (lambda *x, **k: None)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    use_fp16 = device.type == 'cuda'
    if main_rank:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / 'args.json').write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=1))

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(a.data / 'tokenizer.json'))
    val = [json.loads(line) for line in open(a.data / 'val.jsonl')]
    paths = [a.data / 'enc' / f'part-{i:05d}.npz' for i in parse_shards(a.shards)]
    missing = [p_ for p_ in paths if not p_.exists()]
    if missing:
        say(f'нет {len(missing)} шардов, например {missing[0]}', flush=True)
    data = Data([p_ for p_ in paths if p_.exists()])
    say(f'train: {data.n:,} примеров, токенов q {data.q_len.sum():,} / c {data.c_len.sum():,}', flush=True)

    cfg = Config(vocab=tok.get_vocab_size(), d=a.d, heads=a.heads, n_enc=a.n_enc, n_dec=a.n_dec, ffn=a.ffn)
    init = torch.load(a.init_from, map_location='cpu', weights_only=False) if a.init_from else None
    if init:
        cfg = Config(**init['cfg'])
        a.ffn = cfg.ffn
    model = MishaGustokashin(cfg)
    if init:
        model.load_state_dict(init['model'])
        say(f'веса из {a.init_from} (шаг {init.get("step")}), оптимизатор и LR с нуля', flush=True)
    model = model.to(device)
    say(f'model: {sum(p_.numel() for p_ in model.parameters()) / 1e6:.1f}M параметров, ffn={a.ffn}, '
        f'{device} × {world} (эффективный батч ≤ {a.batch * world})', flush=True)

    decay = [p_ for p_ in model.parameters() if p_.dim() >= 2]
    no_decay = [p_ for p_ in model.parameters() if p_.dim() < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': a.wd}, {'params': no_decay, 'weight_decay': 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)

    start, best = 0, -1.0
    if a.resume:
        ck = torch.load(a.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        scaler.load_state_dict(ck['scaler'])
        start, best = ck['step'], ck.get('best', -1.0)
        rng = np.random.default_rng(a.seed + start)
        say(f'resume from step {start}', flush=True)

    if device.type == 'cuda':
        memory_check(model, opt, data, a, device, use_fp16)
    fwd = model if world == 1 else DDP(model, device_ids=[device.index] if device.type == 'cuda' else None)
    stream = (b for j, b in enumerate(data.batches(a.batch, rng, max_tokens=a.max_tokens, pad_to=a.pad_to))
              if j % world == rank)
    log = open(a.out / 'log.jsonl', 'a') if main_rank else None
    t0, tokens = time.time(), 0

    for step in range(start, a.steps):
        prog = progress(step, time.time() - t0, a)
        if world > 1:
            t = torch.tensor([prog], device=device)
            dist.broadcast(t, 0)
            prog = t.item()
        last = step == a.steps - 1 or prog >= 1.0
        lr = lr_at(step, prog, a)
        for g in opt.param_groups:
            g['lr'] = lr

        src, tgt_in, tgt_out = data.collate(next(stream), pad_to=a.pad_to)
        tokens += int((src != 0).sum()) + int((tgt_out != -100).sum())
        src, tgt_in, tgt_out = (x.to(device, non_blocking=True) for x in (src, tgt_in, tgt_out))

        loss, gn = train_step(fwd, opt, scaler, src, tgt_in, tgt_out, use_fp16, a.clip)
        scaler.step(opt)
        scaler.update()

        if main_rank and (step % a.log_every == 0 or last):
            el = time.time() - t0
            rec = {'step': step, 'loss': round(loss.item(), 4), 'lr': round(lr, 6), 'grad_norm': round(gn.item(), 3),
                   'tok/s': int(tokens * world / el), 'min': round(el / 60, 1)}
            if device.type == 'cuda':
                rec['mem_gb'] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + '\n')
            log.flush()

        if ((step + 1) % a.eval_every == 0 or last) and main_rank:
            p1, codes = evaluate(model, tok, val, device, a.eval_n)
            rec = {'step': step + 1, 'val_pass1': round(p1, 4), 'min': round((time.time() - t0) / 60, 1)}
            print(json.dumps(rec), '\n--- пример ---\n' + codes[0], flush=True)
            log.write(json.dumps(rec) + '\n')
            log.flush()
            best_now = p1 > best
            best = max(best, p1)
            save_checkpoint(a.out / 'last.pt', model, step=step + 1, val_pass1=p1, best=best,
                            opt=opt.state_dict(), scaler=scaler.state_dict())
            if best_now:
                save_checkpoint(a.out / 'best.pt', model, step=step + 1, val_pass1=p1)
            if a.keep_ckpts:
                save_checkpoint(a.out / f'ckpt-{step + 1:06d}.pt', model, step=step + 1, val_pass1=p1)
                for old in sorted(a.out.glob('ckpt-*.pt'))[:-a.keep_ckpts]:
                    old.unlink()
        if world > 1 and ((step + 1) % a.eval_every == 0 or last):
            dist.barrier()

        if last:
            break

    if main_rank:
        save_checkpoint(a.out / 'final.pt', model, step=step + 1)
    say(f'done: best val pass@1 = {best:.4f}', flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
