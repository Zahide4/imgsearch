#!/usr/bin/env python3
"""Safety scoring in embedding space, using the model already loaded.

Wikimedia Commons is an educational repository, not a curated stock library:
the 435k prototype contains explicit sexual imagery, gore and graphic medical
material, and enumeration makes the proportion WORSE -- the 435k came from 481
curated topics, while 10M walks the whole namespace.

SigLIP has a text tower and the query path already uses it, so scoring an
image costs one matrix multiply against a handful of cached prompt vectors.
No second model, no extra download, nothing measurable against GPU
throughput. This is broadly how LAION filtered their datasets.

Scores go in the payload; the API filters at QUERY time. That way the
threshold is configuration rather than a re-crawl, and "show me everything"
stays possible for the searches where it is legitimate.

Two honest limits, unchanged from the plan:

  * No filter is clean. Some classical nudes and medical illustration will
    score high; some genuinely explicit material will not.
  * The threshold is a product decision, not a technical one. A tool for
    commercial creative work should filter hard and let people opt back in.
"""
import numpy as np

# Grouping is deliberate. Softmax needs somewhere for an image to land that is
# not "unsafe", so the safe list has to cover the corpus broadly -- Commons is
# mostly maps, species photographs, buildings and scans, and without those
# categories present a botanical illustration distributes its probability mass
# over whatever IS listed.
UNSAFE = [
    'explicit sexual activity',
    'pornography',
    'a photograph of exposed genitalia',
    'a naked person in a sexual pose',
    'graphic violence with blood and injury',
    'a mutilated human corpse',
    'a severe open wound with exposed tissue',
]

# Scored but reported separately: legitimate for a medical or educational
# search, unwanted in a mood board. The API decides what to do with it.
CLINICAL = [
    'a clinical photograph of a surgical procedure',
    'a medical illustration of human anatomy',
    'a photograph of a diseased body part',
]

# Fine art gets its own bucket for the same reason. A Renaissance nude that
# lands here rather than in UNSAFE is the entire point -- without this row it
# has nowhere to go but 'a naked person'.
ARTISTIC_NUDE = [
    'a classical nude painting or marble sculpture',
    'an anatomical figure study drawing',
]

SAFE = [
    'a landscape photograph', 'a portrait of a clothed person',
    'a building or architectural photograph', 'a painting or drawing',
    'a diagram, chart or schematic', 'a map', 'an animal or bird',
    'a plant, flower or tree', 'a photograph of food', 'a vehicle',
    'a historical black and white photograph', 'a scanned document or page',
    'a group of people at an event', 'a manufactured object on a plain background',
    'a microscope or telescope image', 'a sports photograph',
]

GROUPS = [('unsafe', UNSAFE), ('clinical', CLINICAL),
          ('artistic_nude', ARTISTIC_NUDE), ('safe', SAFE)]


class Scorer:
    """Prompt vectors, computed once, reused for the whole run."""

    def __init__(self, model, tokenizer, device='cpu'):
        import torch
        prompts, self.spans = [], []
        for name, group in GROUPS:
            self.spans.append((name, len(prompts), len(prompts) + len(group)))
            prompts.extend(group)
        with torch.inference_mode():
            tokens = tokenizer(prompts).to(device)
            text = model.encode_text(tokens)
            text = text / text.norm(dim=-1, keepdim=True)
        self.text = text.float().cpu().numpy().astype(np.float32)
        # SigLIP trains with a learned temperature; reusing it keeps the
        # softmax on the scale the model was calibrated at rather than one
        # invented here.
        self.scale = float(model.logit_scale.exp().detach().cpu()) if hasattr(
            model, 'logit_scale') else 100.0

    def score(self, vectors):
        """Image vectors (N, 768), L2-normalised, in the same space.

        Returns a dict of (N,) arrays, one per group, each the softmax
        probability mass that group attracts. `unsafe` is the number the API
        filters on; the others are kept because they are the ones a human has
        to look at when tuning the threshold.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        logits = self.scale * (vectors @ self.text.T)
        logits -= logits.max(axis=1, keepdims=True)      # stable softmax
        probability = np.exp(logits)
        probability /= probability.sum(axis=1, keepdims=True)
        return {name: probability[:, lo:hi].sum(axis=1)
                for name, lo, hi in self.spans}

    def payload(self, vectors):
        """Per-image dicts ready to merge into a Qdrant payload."""
        scored = self.score(vectors)
        return [{'safety': round(float(scored['unsafe'][i]), 4),
                 'clinical': round(float(scored['clinical'][i]), 4),
                 'artistic_nude': round(float(scored['artistic_nude'][i]), 4)}
                for i in range(len(scored['unsafe']))]
