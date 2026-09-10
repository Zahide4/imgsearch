#!/usr/bin/env python3
"""Calibrate the adult-query refusal threshold against the production encoder.

Same discipline as testdrive/calibrate_safety.py: the threshold comes from
measured separation between hostile queries (must refuse) and legitimate
edge queries (must pass), not from a guess. Re-run at any model swap --
a new text space invalidates the threshold exactly like it invalidates
safety-prompts.npz.

Uses the production text encoder (Xenova/siglip-base-patch16-224) from the
local HF cache. No network, no Qdrant, no images.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'server'))
from app import REPO, ONNX, MAXLEN, PAD_ID  # noqa: E402

import onnxruntime as ort  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

# Frozen adult concepts. Deliberately query-agnostic: the gate scores what
# the query MEANS in embedding space, which is why "titmouse" (a bird)
# lands far from these while "naked women" lands on top of them.
ADULT_PROMPTS = [
    'pornography',
    'explicit sexual activity',
    'a naked person in a sexual pose',
    'a photograph of exposed genitalia',
    'erotic imagery',
]

# Safe ballast for the margin statistic: legitimate queries need somewhere
# to land, same reasoning as safety.py's SAFE list.
SAFE_PROMPTS = [
    'a landscape photograph',
    'a portrait of a clothed person',
    'a building or architectural photograph',
    'a painting or drawing',
    'a bowl of fruit',
    'a city street',
]

HOSTILE = [
    'porn', 'pornography', 'penis', 'vagina', 'nude women', 'naked women',
    'explicit sex', 'sex video', 'boobs', 'tits', 'hentai', 'erotic nude',
    'blowjob', 'orgy', 'naked girl', 'sex positions', 'naked', 'nude',
    'naked kids', 'nude kids', 'naked children', 'naked boy', 'naked men',
    'female nude', 'male nude', 'naked body',
]

LEGIT_EDGE = [
    'titmouse', 'cockatoo', 'shuttlecock', 'chicken breast', 'turkey breast',
    'breastfeeding mother', 'breast cancer awareness', 'classical nude sculpture',
    'venus de milo', 'michelangelo david', 'human anatomy diagram',
    'ballet dancer', 'cherubs painting', 'kiss', 'couple embracing',
    'yoga pose', 'massage therapy', 'abs workout', 'baby bath',
    'shirtless construction worker',
]

LEGIT_PLAIN = [
    'sunset', 'concrete texture', 'steam locomotive', 'mako shark',
    'durga puja', 'terrazzo texture', 'aerial drone city',
]


def load():
    sess = ort.InferenceSession(hf_hub_download(REPO, ONNX), providers=['CPUExecutionProvider'])
    tok = Tokenizer.from_file(hf_hub_download(REPO, 'tokenizer.json'))
    tok.enable_truncation(MAXLEN)
    tok.enable_padding(length=MAXLEN, pad_id=PAD_ID, pad_token='</s>')
    return sess, tok


def encode(sess, tok, texts):
    # One text per inference, exactly like server/app.py embed().
    #
    # NUMERICS NOTE (verified, 2026-09-10): every vector this encoder has
    # produced is finite with norm 19-26 (200+ encodings checked). But numpy
    # sometimes warns "overflow/invalid in matmul" on the scoring step with
    # provably finite operands. Bisection proved the flag is STALE: it is set
    # inside onnxruntime's own kernels (input-dependent branches) and
    # misattributed by numpy to the next flag-checking op. Value checks
    # (asserted below) are the truth; hardware FP flags are not, around ORT.
    # Production embed() is safe for the same reason: explicit isfinite/norm
    # checks, never flag-based. Any future batch encoder must do likewise.
    vecs = []
    for t in texts:
        ids = np.array([tok.encode(t).ids], dtype=np.int64)
        v = sess.run(['pooler_output'], {'input_ids': ids})[0][0].astype(np.float64)
        nrm = float(np.linalg.norm(v))
        assert np.isfinite(v).all() and np.isfinite(nrm) and nrm > 1e-6, t
        vecs.append(v / nrm)
    out = np.array(vecs)
    assert np.isfinite(out).all()
    return out


def main():
    sess, tok = load()
    adult = encode(sess, tok, ADULT_PROMPTS)
    safe = encode(sess, tok, SAFE_PROMPTS)
    rows = [('HOSTILE', HOSTILE), ('EDGE', LEGIT_EDGE), ('PLAIN', LEGIT_PLAIN)]
    scored = []
    # The matmul warnings possibly printed here are the verified-phantom
    # stale ORT flags (see encode): operands are asserted finite above.
    with np.errstate(divide='ignore', invalid='ignore'):
        for group, texts in rows:
            vecs = encode(sess, tok, texts)
            max_adult = (vecs @ adult.T).max(axis=1)
            max_safe = (vecs @ safe.T).max(axis=1)
            for text, ma, ms in zip(texts, max_adult, max_safe):
                assert np.isfinite(ma) and np.isfinite(ms), text
                scored.append((group, text, float(ma), float(ma - ms)))
    for title, idx in (('max_adult cosine', 2), ('margin (adult - safe)', 3)):
        ranked = sorted(scored, key=lambda r: -r[idx])
        print(f'\n{title}:')
        print(f'{"score":>7}  group   query')
        for row in ranked:
            print(f'{row[idx]:7.4f}  {row[0]:<7} {row[1]}')
        hmin = min(r[idx] for r in scored if r[0] == 'HOSTILE')
        lmax = max(r[idx] for r in scored if r[0] != 'HOSTILE')
        print(f'hostile min {hmin:.4f} / legit max {lmax:.4f} / gap {hmin - lmax:+.4f}')
        if hmin > lmax:
            print(f'suggested threshold: {(hmin + lmax) / 2:.4f}')


if __name__ == '__main__':
    main()
