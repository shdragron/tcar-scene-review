"""Per-scene image embeddings (open_clip), cached in <dataroot>/selection/.

For every scene: 6 cameras x key frames nearest to 2 / 6 / 10 / 14 / 18 s into the scene.
Images are letterboxed to a square before the model's own resize/crop so the full wide FOV
is kept (a plain centre crop would cut the left/right thirds of the 16:9 frame).

    python tools/scene_embed.py            # compute / refresh the cache
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sample_viewer as sv  # noqa: E402

CAMS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
TIMES_S = [2, 6, 10, 14, 18]
MODEL, PRETRAINED = 'ViT-B-32', 'laion2b_s34b_b79k'
EMBED_VERSION = 1


def cache_path(dataroot):
    return os.path.join(dataroot, 'selection', f'image_emb_{MODEL}_{PRETRAINED}_v{EMBED_VERSION}.npz')


def frame_files(db, scene):
    """[time][camera] -> image path for the key frames nearest to TIMES_S."""
    ts = scene['sample_ts']
    out = []
    for t in TIMES_S:
        k = int(np.argmin(np.abs(np.array(ts) - (ts[0] + t * 1e6))))
        chans = db['by_sample'][scene['samples'][k]]
        out.append([chans[c]['filename'] for c in CAMS])
    return out


def _letterbox(img):
    from PIL import Image
    w, h = img.size
    s = max(w, h)
    canvas = Image.new('RGB', (s, s), (0, 0, 0))
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


def compute(db, dataroot, names=None, batch=32):
    import open_clip
    import torch
    from PIL import Image
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    model.eval()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    scenes = [s for s in db['scenes'] if names is None or s['name'] in names]
    jobs = [(si, ti, ci, os.path.join(dataroot, f))
            for si, s in enumerate(scenes) for ti, row in enumerate(frame_files(db, s)) for ci, f in enumerate(row)]
    emb = None
    t0 = time.time()
    for b in range(0, len(jobs), batch):
        chunk = jobs[b:b + batch]
        ims = []
        for *_, path in chunk:
            im = Image.open(path)
            im.draft('RGB', (im.size[0] // 2, im.size[1] // 2))   # fast JPEG decode at half size
            ims.append(preprocess(_letterbox(im.convert('RGB'))))
        with torch.no_grad():
            f = model.encode_image(torch.stack(ims)).float()
            f = torch.nn.functional.normalize(f, dim=-1).numpy()
        if emb is None:
            emb = np.zeros((len(scenes), len(TIMES_S), len(CAMS), f.shape[1]), np.float32)
        for (si, ti, ci, _), v in zip(chunk, f):
            emb[si, ti, ci] = v
        print(f'  embedded {min(b + batch, len(jobs))}/{len(jobs)} images ({time.time() - t0:.0f}s)', flush=True)
    return [s['name'] for s in scenes], emb


_MODEL = {}


def text_features(prompts):
    """L2-normalised text embeddings (same model as the image cache) for zero-shot tagging."""
    import open_clip
    import torch
    if 'm' not in _MODEL:
        _MODEL['m'], _, _ = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
        _MODEL['m'].eval()
        _MODEL['tok'] = open_clip.get_tokenizer(MODEL)
    with torch.no_grad():
        f = _MODEL['m'].encode_text(_MODEL['tok'](prompts)).float()
    return torch.nn.functional.normalize(f, dim=-1).numpy()


def load(db, dataroot, recompute=False):
    """{scene name: array [time, camera, dim]} -- computes missing scenes and updates the cache."""
    path = cache_path(dataroot)
    have = {}
    if os.path.isfile(path) and not recompute:
        z = np.load(path, allow_pickle=False)
        have = {n: e for n, e in zip(z['names'].tolist(), z['emb'])}
    missing = [s['name'] for s in db['scenes'] if s['name'] not in have]
    if missing:
        print(f'[embed] {MODEL}/{PRETRAINED}: {len(missing)} scenes to embed', flush=True)
        names, emb = compute(db, dataroot, set(missing))
        have.update(zip(names, emb))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        allnames = sorted(have)
        np.savez_compressed(path, names=np.array(allnames), emb=np.stack([have[n] for n in allnames]))
        with open(path.replace('.npz', '.json'), 'w', encoding='utf-8') as f:
            json.dump({'model': MODEL, 'pretrained': PRETRAINED, 'version': EMBED_VERSION, 'cams': CAMS,
                       'times_s': TIMES_S, 'preprocess': 'letterbox to square, then model resize/crop'}, f, indent=1)
    return have


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataroot', default=os.path.dirname(HERE))
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--recompute', action='store_true')
    args = ap.parse_args()
    db = sv.load_db(args.dataroot, args.version)
    emb = load(db, args.dataroot, args.recompute)
    print(f'[embed] cached {len(emb)} scenes -> {cache_path(args.dataroot)}')


if __name__ == '__main__':
    sys.exit(main())
