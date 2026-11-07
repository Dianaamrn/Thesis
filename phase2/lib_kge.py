"""Frozen knowledge-graph embedding: train once, fold in new entities, score and featurise triples.

The embedding is trained ONCE on the reference graph (PyKEEN, same settings as Phase 1) and never
retrained during the stream. A new entity gets its own vector, fitted on the triple that introduced
it while every old vector stays exactly where it was ("hard fold-in"). So the coordinate system
never changes, and a detector trained on the reference keeps seeing the same map.

Features are built as in Phase 1: entity and relation vectors -> real numbers ([real, imag] for
ComplEx/RotatE) -> L2-normalised rows -> Hadamard s*p*o or concatenation [s, p, o].
"""
from __future__ import annotations

import time

import numpy as np
import torch

import settings


def pick_device(device='auto'):
    if device != 'auto':
        return device
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------
# Training (same as phase1/drift_sphere.py:_train_embeddings, but the model is kept)
# ---------------------------------------------------------------------------

def train_pykeen(triples, model_name, seed, device):
    import pykeen.models as pykeen_models
    from pykeen.training import SLCWATrainingLoop
    from pykeen.triples import TriplesFactory

    started = time.time()
    tf = TriplesFactory.from_labeled_triples(np.asarray(triples, dtype=str))
    dim = settings.KGE['real_dim'] // 2 if model_name in settings.COMPLEX_MODELS else settings.KGE['real_dim']
    model = getattr(pykeen_models, model_name)(triples_factory=tf, embedding_dim=dim, random_seed=seed)
    model = model.to(torch.device(device))
    loop = SLCWATrainingLoop(model=model, triples_factory=tf)
    loop.train(triples_factory=tf, num_epochs=settings.KGE['epochs'],
               batch_size=min(settings.KGE['batch_size'], max(32, len(triples))),
               use_tqdm=False, use_tqdm_batch=False, pin_memory=False, num_workers=0)
    model.eval()
    return FrozenKGE.from_pykeen(model, tf, device), time.time() - started


# ---------------------------------------------------------------------------
# Frozen embedding
# ---------------------------------------------------------------------------

class FrozenKGE:
    """Entity/relation representations of a trained model, plus the model's own score function.

    E          (n_entities, *shape) representation tensor (complex for ComplEx/RotatE)
    rels       list of relation representation tensors (TransH has two: translation, normal)
    score_fn   score(h, r_list, t) -> (batch,) higher = more plausible (the model's interaction)
    constrain  constraint applied to entity vectors after each update (e.g. TransE unit norm)
    """

    def __init__(self, entity_to_id, relation_to_id, E, rels, score_fn, constrain, device):
        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id
        self.E = E
        self.rels = rels
        self.score_fn = score_fn
        self.constrain = constrain
        self.device = device

    @classmethod
    def from_pykeen(cls, model, tf, device):
        reps = model.entity_representations
        if len(reps) != 1:
            raise NotImplementedError('models with several entity representations are not supported')
        with torch.no_grad():
            E = reps[0](indices=None).detach()
            rels = [r(indices=None).detach() for r in model.relation_representations]
        interaction = model.interaction

        def score_fn(h, r, t):
            return interaction.score_hrt(h=h, r=r[0] if len(r) == 1 else r, t=t).view(-1)

        constrain = reps[0].constrainer if reps[0].constrainer is not None else (lambda x: x)
        return cls(dict(tf.entity_to_id), dict(tf.relation_to_id), E, rels, score_fn, constrain, device)

    @classmethod
    def random(cls, triples, seed, device='cpu'):
        """Smoke-test stand-in: random vectors, TransE-style score. No training."""
        g = torch.Generator().manual_seed(seed)
        triples = np.asarray(triples, dtype=str)
        ents = sorted(set(triples[:, 0]).union(triples[:, 2]))
        rels = sorted(set(triples[:, 1]))
        d = settings.KGE['real_dim']
        E = torch.nn.functional.normalize(torch.randn(len(ents), d, generator=g), dim=-1).to(device)
        R = torch.randn(len(rels), d, generator=g).to(device) * 0.1

        def score_fn(h, r, t):
            return -(h + r[0] - t).norm(dim=-1)

        return cls({e: i for i, e in enumerate(ents)}, {r: i for i, r in enumerate(rels)}, E, [R], score_fn,
                   lambda x: torch.nn.functional.normalize(x, dim=-1), device)

    # -- fold-in ------------------------------------------------------------------------------

    def fold_in(self, triples, seed=None):
        """Vectors for unseen entities. `triples`: list of (h, r, t) labels, one per new entity copy.

        Each triple has one or two unseen entities. Every (unseen entity, triple) pair gets its own
        vector, fitted only on that triple, with all known vectors frozen. Returns
        {(entity_label, triple_index): vector}. Relations must be known (others are skipped).
        """
        cfg = settings.FOLDIN
        seed = cfg['seed'] if seed is None else seed
        torch.manual_seed(seed)
        keys, h_copy, t_copy, h_known, t_known, r_ids = [], [], [], [], [], []
        for i, (h, r, t) in enumerate(triples):
            if r not in self.relation_to_id:
                continue
            hc = tc = -1
            if h not in self.entity_to_id:
                hc = len(keys)
                keys.append((h, i))
            if t not in self.entity_to_id:
                if t == h and hc >= 0:
                    tc = hc
                else:
                    tc = len(keys)
                    keys.append((t, i))
            if hc < 0 and tc < 0:
                continue
            h_copy.append(hc)
            t_copy.append(tc)
            h_known.append(self.entity_to_id.get(h, 0))
            t_known.append(self.entity_to_id.get(t, 0))
            r_ids.append(self.relation_to_id[r])
        if not keys:
            return {}
        dev = self.device
        h_copy, t_copy = torch.tensor(h_copy, device=dev), torch.tensor(t_copy, device=dev)
        h_known, t_known = torch.tensor(h_known, device=dev), torch.tensor(t_known, device=dev)
        r_ids = torch.tensor(r_ids, device=dev)
        is_complex = torch.is_complex(self.E)
        real_E = torch.view_as_real(self.E) if is_complex else self.E
        # Start every new vector at a random known entity's vector (right scale and constraints).
        start = real_E[torch.randint(0, len(real_E), (len(keys),), device=dev)].clone()
        param = torch.nn.Parameter(start + 0.01 * torch.randn_like(start))
        opt = torch.optim.Adam([param], lr=cfg['lr'])
        n_ent, K = len(self.E), cfg['negatives']
        B = len(r_ids)

        def as_rep(x):
            return torch.view_as_complex(x.contiguous()) if is_complex else x

        def side(copy_idx, known_idx, new):
            use_new = (copy_idx >= 0).view(-1, *([1] * (self.E.dim() - 1)))
            return torch.where(use_new, new[copy_idx.clamp(min=0)], self.E[known_idx])

        r_vec = [R[r_ids] for R in self.rels]
        for _ in range(cfg['steps']):
            new = as_rep(param)
            h, t = side(h_copy, h_known, new), side(t_copy, t_known, new)
            pos = self.score_fn(h, r_vec, t)
            rand = self.E[torch.randint(0, n_ent, (B * K,), device=dev)]
            rK = [x.repeat_interleave(K, dim=0) for x in r_vec]
            neg_t = self.score_fn(h.repeat_interleave(K, dim=0), rK, rand)   # corrupt the tail
            neg_h = self.score_fn(rand, rK, t.repeat_interleave(K, dim=0))   # corrupt the head
            loss = torch.nn.functional.softplus(-pos).mean() + 0.5 * (
                torch.nn.functional.softplus(neg_t).mean() + torch.nn.functional.softplus(neg_h).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                x = self.constrain(as_rep(param.data))
                param.data = torch.view_as_real(x) if is_complex else x
        final = as_rep(param.detach())
        return {key: final[j] for j, key in enumerate(keys)}

    # -- vectors, scores, features --------------------------------------------------------------

    def entity_reps(self, labels, overrides):
        """Representation tensors for entity labels; overrides[label] wins for folded-in entities."""
        out = []
        for x in labels:
            out.append(overrides[x] if x in overrides else self.E[self.entity_to_id[x]])
        return torch.stack(out)

    def score(self, h_labels, r_labels, t_labels, overrides, batch=4096):
        out = np.full(len(h_labels), np.nan)
        with torch.no_grad():
            for s in range(0, len(h_labels), batch):
                sl = slice(s, s + batch)
                ok = [i for i, r in enumerate(r_labels[sl]) if r in self.relation_to_id]
                if not ok:
                    continue
                hl = [h_labels[sl][i] for i in ok]
                tl = [t_labels[sl][i] for i in ok]
                rid = torch.tensor([self.relation_to_id[r_labels[sl][i]] for i in ok], device=self.device)
                sc = self.score_fn(self.entity_reps(hl, overrides), [R[rid] for R in self.rels],
                                   self.entity_reps(tl, overrides))
                out[np.asarray(ok) + s] = sc.detach().cpu().numpy()
        return out

    @staticmethod
    def _realify(x):
        if torch.is_complex(x):
            x = torch.cat([x.real, x.imag], dim=-1)
        x = x.reshape(x.shape[0], -1).detach().cpu().numpy().astype(np.float64)
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return x / n

    def features(self, h_labels, r_labels, t_labels, overrides):
        """Phase 1 features: (hadamard, concatenation) arrays. Relation = first relation representation."""
        s = self._realify(self.entity_reps(h_labels, overrides))
        o = self._realify(self.entity_reps(t_labels, overrides))
        rid = torch.tensor([self.relation_to_id[r] for r in r_labels], device=self.device)
        p = self._realify(self.rels[0][rid])
        return s * p * o, np.concatenate([s, p, o], axis=1)

    def entity_matrix(self):
        """All known entity vectors as real, L2-normalised rows (label order = entity_to_id)."""
        return self._realify(self.E)

    def relation_matrix(self):
        return self._realify(self.rels[0])


def first_appearance_foldin(kge, pool):
    """Fold in every unseen entity of the pool on the triple where it first appears (stream order).

    Returns overrides {entity_label: vector} and the per-row flag `folded` (row uses a folded vector).
    The pool is in stream order, so in the real stream this is exactly what an online system can do.
    """
    first = {}
    rows = list(zip(pool['subject'].astype(str), pool['edge_type'].astype(str), pool['object'].astype(str)))
    for i, (h, r, t) in enumerate(rows):
        for e in (h, t):
            if e not in kge.entity_to_id and e not in first:
                first[e] = i
    needed = sorted(set(first.values()))
    fitted = kge.fold_in([rows[i] for i in needed]) if needed else {}
    local = {pos: i for pos, i in enumerate(needed)}
    overrides = {}
    for (ent, pos), vec in fitted.items():
        if first.get(ent) == local[pos]:
            overrides[ent] = vec
    folded = np.array([(h in overrides) or (t in overrides) for h, _, t in rows], dtype=np.int8)
    unscorable = np.array([(h not in kge.entity_to_id and h not in overrides)
                           or (t not in kge.entity_to_id and t not in overrides)
                           or (r not in kge.relation_to_id) for h, r, t in rows], dtype=np.int8)
    return overrides, folded, unscorable
