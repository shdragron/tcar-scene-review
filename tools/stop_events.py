"""What happens around the ego while it is stopped -> <dataroot>/selection/stop_events.json.

Stage 1 of curate.py needs to know whether two same-place stops show the same *kind* of events.
The LiDAR change count (scene_select) is a size-weighted amount: a truck turning near the ego
produces hundreds of changed cells, a pedestrian crossing a few (blobs < 1 m^2 are dropped as
noise), so a crosswalk full of pedestrians can look "quiet". Here the events are typed instead.

Key frames (2 Hz) while the ego is stopped (< 1 km/h), 3 front cameras, COCO detector
(same as scene_objects.py). A detection is *moving* when its box overlaps no box of the same kind
in the neighbouring stopped frame of the same camera (IoU < 0.5 with both neighbours) -- with the
ego standing still, a walking pedestrian or a passing car shifts by about its own width in 0.5 s,
a parked car or a waiting pedestrian stays put. Only near boxes (height >= 50 px) count; moving boxes
>= 100 px are also counted as *close* (a pedestrian on the crosswalk in front vs one on a far corner).
Per scene: mean boxes, moving boxes and close moving boxes per frame for pedestrians / two-wheelers /
vehicles.
Scenes with fewer than 6 stopped key frames (3 s) are skipped.

    python tools/stop_events.py
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
import scene_objects as so  # noqa: E402

CAMS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT')
KIND = {'person': 'ped', 'bicycle': 'two', 'motorcycle': 'two', 'car': 'veh', 'truck': 'veh', 'bus': 'veh'}
KINDS = ('ped', 'two', 'veh')
STOP_KMH, MIN_FRAMES, IOU_STATIC = 1.0, 6, 0.5
CLOSE_PX = 100              # box height >= 100 px: a pedestrian within ~10 m (crosswalk in front), a car within ~8 m
VERSION = 2


def boxes(out, cats, W, H):
    res = []
    for lab, s, b in zip(out['labels'].tolist(), out['scores'].tolist(), out['boxes'].tolist()):
        k = KIND.get(cats[lab])
        x1, y1, x2, y2 = b
        if s < so.SCORE or k is None or (y2 - y1) < so.NEAR_PX:
            continue
        if k == 'veh' and y2 > 0.97 * H and (x2 - x1) > 0.4 * W:
            continue                                        # ego body
        res.append((k, x1, y1, x2, y2))
    return res


def iou(a, b):
    ix = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    iy = max(0.0, min(a[4], b[4]) - max(a[2], b[2]))
    inter = ix * iy
    return inter / ((a[3] - a[1]) * (a[4] - a[2]) + (b[3] - b[1]) * (b[4] - b[2]) - inter + 1e-9)


def moving_flags(frames):
    """frames: [frame][cam] -> list of boxes; returns the same shape with a moving flag per box."""
    out = []
    for k, fr in enumerate(frames):
        row = []
        for c, bs in enumerate(fr):
            nb = [frames[j][c] for j in (k - 1, k + 1) if 0 <= j < len(frames)]
            row.append([all(max((iou(b, o) for o in n if o[0] == b[0]), default=0.0) < IOU_STATIC for n in nb)
                        for b in bs])
        out.append(row)
    return out


def scene_events(db, scene, dataroot, model, tf, cats):
    import torch
    from PIL import Image
    tr = db['tracks'][scene['name']]
    tt = np.array(tr['t'], dtype=float)
    sp = np.array(tr['speed']) * 3.6
    keys = []
    for tok in scene['samples']:
        ch = db['by_sample'][tok]
        ts = ch['LIDAR_TOP']['timestamp']
        if float(np.interp(ts, tt, sp)) < STOP_KMH:
            keys.append(((ts - scene['sample_ts'][0]) / 1e6, [ch[c]['filename'] for c in CAMS]))
    if len(keys) < MIN_FRAMES:
        return None
    frames = []
    for _, files in keys:
        ims = [Image.open(os.path.join(dataroot, f)).convert('RGB') for f in files]
        with torch.no_grad():
            outs = model([tf(im) for im in ims])
        frames.append([boxes(o, cats, *im.size) for im, o in zip(ims, outs)])
    mov = moving_flags(frames)
    per = []
    for fr, mv in zip(frames, mov):
        c = {k: 0 for k in KINDS}
        c.update({k + '_mov': 0 for k in KINDS})
        c.update({k + '_movc': 0 for k in KINDS})
        for bs, ms in zip(fr, mv):
            for b, m in zip(bs, ms):
                c[b[0]] += 1
                c[b[0] + '_mov'] += int(m)
                c[b[0] + '_movc'] += int(m and (b[4] - b[2]) >= CLOSE_PX)
        per.append(c)
    keys_all = list(per[0])
    return {'n_frames': len(per), 't_s': [round(t, 2) for t, _ in keys], 'per_frame': per,
            'mean': {k: float(np.mean([p[k] for p in per])) for k in keys_all},
            'moving_s': {k: float(sum(p[k + '_mov'] for p in per) * 0.5) for k in KINDS}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataroot', default=os.path.dirname(HERE))
    ap.add_argument('--version', default='v1.0-trainval')
    args = ap.parse_args()
    db = sv.load_db(args.dataroot, args.version)
    out_path = os.path.join(args.dataroot, 'selection', 'stop_events.json')
    done = {}
    if os.path.isfile(out_path):
        with open(out_path, encoding='utf-8') as f:
            j = json.load(f)
        if j.get('version') == VERSION:
            done = j['scenes']
    model, tf, cats = so.detector()
    t0 = time.time()
    todo = [s for s in db['scenes'] if s['name'] not in done]
    for i, s in enumerate(todo):
        done[s['name']] = scene_events(db, s, args.dataroot, model, tf, cats)
        e = done[s['name']]
        msg = 'not stopped' if e is None else ' '.join(
            f"{k} {e['mean'][k]:.1f}/{e['mean'][k + '_mov']:.2f}/{e['mean'][k + '_movc']:.2f}" for k in KINDS)
        print(f'  {s["name"]} ({i + 1}/{len(todo)}) {msg}  {time.time() - t0:.0f}s', flush=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'version': VERSION, 'created': datetime.datetime.now().isoformat(timespec='seconds'),
                       'model': 'fasterrcnn_mobilenet_v3_large_fpn (COCO)', 'cams': CAMS, 'near_px': so.NEAR_PX,
                       'close_px': CLOSE_PX, 'iou_static': IOU_STATIC, 'stop_kmh': STOP_KMH, 'scenes': done},
                      f, ensure_ascii=False, indent=1)
    print(f'[stop events] {sum(1 for v in done.values() if v)} stopped scenes of {len(done)} -> {out_path}')


if __name__ == '__main__':
    sys.exit(main())
