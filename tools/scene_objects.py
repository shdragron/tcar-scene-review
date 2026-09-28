"""Per-scene traffic-participant counts from a pretrained COCO detector -> <dataroot>/selection/objects.json.

6 cameras x key frames nearest to 2 / 6 / 10 / 14 / 18 s (same frames as scene_embed).
Detector: torchvision Faster R-CNN MobileNetV3-Large FPN (COCO), score >= 0.5.
Classes: vehicles (car, truck, bus), VRUs (person, bicycle, motorcycle), traffic lights.
"near" = box height >= 50 px (1080p), a rough proxy for "close enough to matter".
The ego vehicle's own body (a wide box touching the bottom edge, seen by the rear cameras) is dropped.
Per scene: per-time sums over the 6 cameras (overlapping cameras can double count -- a relative
measure), then mean and max over the 5 time points.

    python tools/scene_objects.py
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sample_viewer as sv  # noqa: E402
import scene_embed as se  # noqa: E402

VEH, VRU, LIGHT = ('car', 'truck', 'bus'), ('person', 'bicycle', 'motorcycle'), ('traffic light',)
NEAR_PX, SCORE = 50, 0.5
VERSION = 1


def detector():
    import torch
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
    w = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    return fasterrcnn_mobilenet_v3_large_fpn(weights=w).eval(), w.transforms(), w.meta['categories']


def count_frame(out, cats, W, H):
    c = {k: 0 for k in ('veh', 'vru', 'light', 'veh_near', 'vru_near')}
    for lab, s, b in zip(out['labels'].tolist(), out['scores'].tolist(), out['boxes'].tolist()):
        if s < SCORE:
            continue
        name = cats[lab]
        x1, y1, x2, y2 = b
        if name in VEH and y2 > 0.97 * H and (x2 - x1) > 0.4 * W:
            continue                                    # ego body in the rear cameras
        near = (y2 - y1) >= NEAR_PX
        if name in VEH:
            c['veh'] += 1; c['veh_near'] += near
        elif name in VRU:
            c['vru'] += 1; c['vru_near'] += near
        elif name in LIGHT:
            c['light'] += 1
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataroot', default=os.path.dirname(HERE))
    ap.add_argument('--version', default='v1.0-trainval')
    args = ap.parse_args()
    import torch
    from PIL import Image
    db = sv.load_db(args.dataroot, args.version)
    out_path = os.path.join(args.dataroot, 'selection', 'objects.json')
    done = {}
    if os.path.isfile(out_path):
        with open(out_path, encoding='utf-8') as f:
            j = json.load(f)
        if j.get('version') == VERSION:
            done = j['scenes']
    model, tf, cats = detector()
    t0 = time.time()
    todo = [s for s in db['scenes'] if s['name'] not in done]
    for si, s in enumerate(todo):
        per_t = []
        for row in se.frame_files(db, s):                # [time][camera]
            tot = {k: 0 for k in ('veh', 'vru', 'light', 'veh_near', 'vru_near')}
            ims = [Image.open(os.path.join(args.dataroot, f)).convert('RGB') for f in row]
            with torch.no_grad():
                outs = model([tf(im) for im in ims])
            for im, o in zip(ims, outs):
                for k, v in count_frame(o, cats, *im.size).items():
                    tot[k] += v
            per_t.append(tot)
        keys = per_t[0].keys()
        done[s['name']] = {'per_time': per_t,
                           'mean': {k: float(np.mean([p[k] for p in per_t])) for k in keys},
                           'max': {k: int(max(p[k] for p in per_t)) for k in keys}}
        print(f'  {s["name"]} ({si + 1}/{len(todo)}) near veh {done[s["name"]]["mean"]["veh_near"]:.1f} '
              f'vru {done[s["name"]]["mean"]["vru_near"]:.1f}  {time.time() - t0:.0f}s', flush=True)
        with open(out_path, 'w', encoding='utf-8') as f:     # save as we go
            json.dump({'version': VERSION, 'created': datetime.datetime.now().isoformat(timespec='seconds'),
                       'model': 'fasterrcnn_mobilenet_v3_large_fpn (COCO)', 'score': SCORE, 'near_px': NEAR_PX,
                       'times_s': se.TIMES_S, 'cams': se.CAMS, 'scenes': done}, f, ensure_ascii=False, indent=1)
    print(f'[objects] {len(done)} scenes -> {out_path}')


if __name__ == '__main__':
    sys.exit(main())
