import numpy as np
import torch

from solvformer import BOS, PAD


class Data:
    """Закодированные шарды, склеенные в общие массивы токенов со смещениями; отдаёт батчи из примеров похожей длины."""

    def __init__(self, paths):
        qs, qo, cs, co = [], [], [], []
        q_base, c_base = 0, 0
        for p in paths:
            z = np.load(p)
            qs.append(z['q_ids'])
            cs.append(z['c_ids'])
            qo.append(z['q_off'][:-1] + q_base)
            co.append(z['c_off'][:-1] + c_base)
            q_base += len(z['q_ids'])
            c_base += len(z['c_ids'])

        self.q, self.c = np.concatenate(qs), np.concatenate(cs)
        self.q_off = np.concatenate(qo + [[q_base]])
        self.c_off = np.concatenate(co + [[c_base]])
        self.q_len = np.diff(self.q_off)
        self.c_len = np.diff(self.c_off)
        self.n = len(self.q_len)

    def collate(self, idx, pad_to=8):
        Lq = -(-(self.q_len[idx].max()) // pad_to) * pad_to
        Lc = -(-(self.c_len[idx].max()) // pad_to) * pad_to
        src = np.full((len(idx), Lq), PAD, dtype=np.int64)
        tgt_in = np.full((len(idx), Lc), PAD, dtype=np.int64)
        tgt_out = np.full((len(idx), Lc), -100, dtype=np.int64)
        for r, i in enumerate(idx):
            q = self.q[self.q_off[i]:self.q_off[i + 1]]
            c = self.c[self.c_off[i]:self.c_off[i + 1]]
            src[r, :len(q)] = q
            tgt_out[r, :len(c)] = c
            tgt_in[r, 0] = BOS
            tgt_in[r, 1:len(c)] = c[:-1]
        return torch.from_numpy(src), torch.from_numpy(tgt_in), torch.from_numpy(tgt_out)

    def batches(self, batch_size, rng, chunk=1024, max_tokens=None, pad_to=8):
        """Бесконечный поток батчей (массивов индексов примеров).

        Перемешанные примеры режутся на куски по batch_size * chunk; внутри куска они сортируются по длине кода
        и вопроса, так что в батч попадают примеры похожей длины и паддинга меньше.
        max_tokens ограничивает «примеров × max(длина вопроса, длина кода)» с паддингом: энкодер и декодер
        (вместе с логитами на 16k слов) получают не больше max_tokens позиций. Длинные батчи поэтому меньше
        batch_size, и память на шаг не зависит от того, какие примеры попались."""
        while True:
            perm = rng.permutation(self.n)
            out = []
            for s in range(0, self.n, batch_size * chunk):
                part = perm[s:s + batch_size * chunk]
                part = part[np.lexsort((self.q_len[part], self.c_len[part] // 8))]
                if max_tokens is None:
                    out += [part[i:i + batch_size] for i in range(0, len(part) - batch_size + 1, batch_size)]
                else:
                    out += self._cut(part, batch_size, max_tokens, pad_to)
            for k in rng.permutation(len(out)):
                yield out[k]

    def _cut(self, part, batch_size, max_tokens, pad_to):
        """Режет отсортированный кусок на батчи: батч растёт, пока не упрётся в batch_size или max_tokens."""
        up = lambda x: -(-x // pad_to) * pad_to
        ql, cl = self.q_len[part].tolist(), self.c_len[part].tolist()
        res, start, mq, mc = [], 0, 0, 0
        for j in range(len(part)):
            nq, nc = max(mq, ql[j]), max(mc, cl[j])
            if j - start + 1 > batch_size or (j - start + 1) * max(up(nq), up(nc)) > max_tokens:
                res.append(part[start:j])
                start, nq, nc = j, ql[j], cl[j]
            mq, mc = nq, nc
        if (len(part) - start) * max(up(mq), up(mc)) * 2 > max_tokens:
            res.append(part[start:])
        return res

    def worst_shape(self, batch_size, max_tokens, pad_to=8):
        """Самая тяжёлая форма батча, которую допускает лимит: (примеров, длина вопроса, длина кода)."""
        up = lambda x: -(-x // pad_to) * pad_to
        Lq, Lc = up(int(self.q_len.max())), up(int(self.c_len.max()))
        L = min(Lq, Lc)
        n = min(batch_size, max_tokens // max(L, Lc))
        return n, L, Lc
