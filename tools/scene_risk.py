"""Per-scene driving-risk proxies without perception labels -> <dataroot>/selection/risk.json.

Events (times are seconds from the scene's first key frame, like the viewer)
  hard_brake    longitudinal acceleration (0.5 s mean) <= --brake m/s^2
  lateral       |lateral acceleration| (0.5 s mean) >= --lat m/s^2
  close_follow  time headway to the nearest obstacle in the ego path <= --thw s (ego > 3 m/s)
  closing       time to collision (gap / closing speed) <= --ttc s
The gap comes from LiDAR key frames (2 Hz): points 0.3-1.8 m above the fitted ground inside a 2.0 m
wide corridor (the ego's own width) along the path the ego *actually drove* in the next 5 s (INS track
of the whole log, extended straight past its end), measured along that path from the front bumper
(assumed 2.3 m ahead of the LiDAR) out to 40 m; a 0.5 m bin is an obstacle with >= 5 points spanning
>= 0.3 m in height. v3 bent a 2.4 m corridor by the current yaw rate instead: in turns it swept over
parked cars, fences and bollards on the corner (5 of 10 flagged scenes, none with braking); cars
passed at the side and the road surface rising ahead (a thin layer just above 0.3 m) also counted.
Score = worst event severity: 1 at the threshold, +1 per step beyond it (brake/lat 1 m/s^2,
headway 0.3 s, TTC 0.5 s). These are proxies: no object classes, calibration is still default.

    python tools/scene_risk.py [--brake -3 --lat 3 --thw 0.6 --ttc 2]
"""
import argparse
import datetime
import json
import math
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sample_viewer as sv  # noqa: E402
import scene_select as ss  # noqa: E402

BUMPER_M, HALF_W, RANGE_M = 2.3, 1.0, 40.0   # HALF_W: ego half width (~0.95 m); wider caught cars passed at the side
MIN_SPREAD = 0.3            # an obstacle bin must span >= 0.3 m in height (a rising road ahead is a thin layer)
OBJ_H = (0.3, 1.8)          # collision-relevant height above the ground plane
MIN_PTS = 5                 # points per 0.5 m bin along the corridor
LIDAR_H = 1.9               # LiDAR height above ground (ground-plane fit, analysis 2026-09-24)
CACHE_VERSION = 5
PATH_AHEAD_S, PATH_STEP = 5.0, 0.25
# v1 used scene_select's min-based local ground; stray points below the road pulled that ground
# down and the road surface itself (lowest rings, 2-8 m ahead) came out as obstacles 0.5-1.5 m
# away at 50 km/h. v2 fitted a RANSAC plane but kept overhead returns (branches / gantries seen by
# the upward rings at 2.3-2.9 m) and once locked onto a raised surface. v3: ground candidates must
# lie >= 1.2 m below the LiDAR, the plane must sit near the mounting height, obstacles 0.3-1.8 m.
DEFAULT_PLANE = (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -LIDAR_H]))


def ground_plane(x, y, z, rng):
    r = np.hypot(x, y)
    m = (r > 3) & (r < 15) & (z < -1.2)
    P = np.stack([x[m], y[m], z[m]], 1)
    if len(P) < 200:
        return DEFAULT_PLANE
    if len(P) > 4000:
        P = P[rng.choice(len(P), 4000, replace=False)]
    best, plane = 0, DEFAULT_PLANE
    for _ in range(80):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0]); nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n = n / nn
        if abs(n[2]) < 0.95:
            continue
        k = int((np.abs((P - s[0]) @ n) < 0.08).sum())
        if k > best:
            best, plane = k, (n if n[2] > 0 else -n, s[0])
    n, p0 = plane
    h0 = (p0 @ n) / n[2]                                    # plane height below the sensor origin
    return plane if -LIDAR_H - 0.5 <= h0 <= -LIDAR_H + 0.5 else DEFAULT_PLANE


def log_track(db, log):
    """INS track of a whole log (all its scenes, time-sorted) so a path can run past a scene's end."""
    parts = [db['tracks'][s['name']] for s in db['scenes'] if s['log'] == log and db['tracks'].get(s['name'])]
    t = np.concatenate([np.array(p['t'], dtype=float) for p in parts])
    o = np.argsort(t, kind='stable')
    t, keep = t[o], np.ones(len(o), bool)
    keep[1:] = np.diff(t) > 0
    return {k: np.concatenate([np.array(p[k], dtype=float) for p in parts])[o][keep] for k in ('x', 'y', 'yaw')} | {'t': t[keep]}


def future_path(ltr, ts):
    """Driven path from ts on, in the ego frame at ts, resampled every PATH_STEP m out to the corridor end;
    past the end of the recording (or once the ego stops) it continues straight along the last heading."""
    t = ltr['t']
    i0, i1 = np.searchsorted(t, ts), np.searchsorted(t, ts + PATH_AHEAD_S * 1e6)
    x0, y0 = float(np.interp(ts, t, ltr['x'])), float(np.interp(ts, t, ltr['y']))
    yaw0 = float(np.interp(ts, t, np.unwrap(ltr['yaw'])))
    c, s = math.cos(yaw0), math.sin(yaw0)
    dx, dy = ltr['x'][i0:i1] - x0, ltr['y'][i0:i1] - y0
    px = np.concatenate([[0.0], c * dx + s * dy]); py = np.concatenate([[0.0], -s * dx + c * dy])
    seg = np.hypot(np.diff(px), np.diff(py))
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    L = BUMPER_M + RANGE_M + 1.0
    q = np.arange(0.0, L, PATH_STEP)
    if arc[-1] > 0.5:
        k = np.searchsorted(arc, arc[-1] - 1.0)                     # heading over the last metre
        hd = math.atan2(py[-1] - py[k], px[-1] - px[k])
    else:
        hd = 0.0
    on = q <= arc[-1]
    qx = np.where(on, np.interp(q, arc, px), px[-1] + (q - arc[-1]) * math.cos(hd))
    qy = np.where(on, np.interp(q, arc, py), py[-1] + (q - arc[-1]) * math.sin(hd))
    return qx, qy, q


def front_gaps(db, scene, dataroot, ltr):
    """[(t_s, gap_m or None, v_mps)] for every LiDAR key frame of the scene."""
    tr = db['tracks'][scene['name']]
    t0 = scene['sample_ts'][0]
    tt = np.array(tr['t'], dtype=float)
    v = np.array(tr['speed'])
    rng = np.random.default_rng(0)
    out = []
    for tok in scene['samples']:
        lid = db['by_sample'][tok]['LIDAR_TOP']
        ts = lid['timestamp']
        vk = float(np.interp(ts, tt, v))
        a = np.fromfile(os.path.join(dataroot, lid['filename']), dtype=np.float32).reshape(-1, 5)
        x, y, z = a[:, 0].astype(float), a[:, 1].astype(float), a[:, 2].astype(float)
        n, p0 = ground_plane(x, y, z, rng)
        h = (np.stack([x, y, z], 1) - p0) @ n
        m = (h > OBJ_H[0]) & (h < OBJ_H[1]) & (np.hypot(x, y) < BUMPER_M + RANGE_M + 2.0) & (x > -2.0)
        gap = None
        if m.any():
            qx, qy, q = future_path(ltr, ts)
            X, Y = x[m], y[m]
            best_d = np.full(len(X), np.inf); best_s = np.zeros(len(X))
            for i in range(0, len(q), 40):                           # nearest path sample, in chunks
                d = np.hypot(X[:, None] - qx[None, i:i + 40], Y[:, None] - qy[None, i:i + 40])
                j = np.argmin(d, axis=1); dj = d[np.arange(len(X)), j]
                better = dj < best_d
                best_d[better] = dj[better]; best_s[better] = q[i:i + 40][j[better]]
            inside = (best_d <= HALF_W) & (best_s > BUMPER_M) & (best_s < BUMPER_M + RANGE_M)
            if inside.any():
                b = ((best_s[inside] - BUMPER_M) / 0.5).astype(int)
                hb = h[m][inside]
                cnt = np.bincount(b)
                hi = np.full(len(cnt), -np.inf); lo = np.full(len(cnt), np.inf)
                np.maximum.at(hi, b, hb); np.minimum.at(lo, b, hb)
                hit = np.nonzero((cnt >= MIN_PTS) & (hi - lo >= MIN_SPREAD))[0]
                gap = float(hit[0] * 0.5) if len(hit) else None
        out.append(((ts - t0) / 1e6, gap, vk))
    return out


def persistent(gaps):
    """Keep a gap only if an adjacent key frame (0.5 s away) also sees an obstacle at a consistent range."""
    out = []
    for k, (t, g, vk) in enumerate(gaps):
        ok = False
        if g is not None:
            for j in (k - 1, k + 1):
                if 0 <= j < len(gaps) and gaps[j][1] is not None and abs(gaps[j][1] - g) <= vk * 0.5 + 3.0:
                    ok = True
        out.append((t, g if ok else None, vk))
    return out


def smooth(x, n):
    return np.convolve(x, np.ones(n) / n, mode='same')


def scene_risk(db, scene, gaps, P):
    tr = db['tracks'][scene['name']]
    t = (np.array(tr['t'], dtype=float) - scene['sample_ts'][0]) / 1e6
    n = max(1, int(round(0.5 / np.median(np.diff(t)))))
    ax, ay = smooth(np.array(tr['ax']), n), smooth(np.array(tr['ay']), n)
    inside = (t >= 0) & (t <= 20)
    ax, ay, t = ax[inside], ay[inside], t[inside]
    events, metrics = [], {}
    i = int(np.argmin(ax)); metrics['min_ax'] = float(ax[i])
    if ax[i] <= P['brake']:
        events.append({'type': 'hard_brake', 't': float(t[i]), 'value': float(ax[i]),
                       'severity': 1 + (P['brake'] - ax[i]) / 1.0, 'text': f'급제동 {ax[i]:.1f} m/s²'})
    j = int(np.argmax(np.abs(ay))); metrics['max_ay'] = float(abs(ay[j]))
    if abs(ay[j]) >= P['lat']:
        events.append({'type': 'lateral', 't': float(t[j]), 'value': float(ay[j]),
                       'severity': 1 + (abs(ay[j]) - P['lat']) / 1.0,
                       'text': f"급한 {'좌' if ay[j] > 0 else '우'}측 횡가속 {abs(ay[j]):.1f} m/s²"})
    thw = [(g / vk, tk, g, vk) for tk, g, vk in gaps if g is not None and vk > 3.0]
    if thw:
        h, tk, g, vk = min(thw)
        metrics['min_thw'] = float(h)
        if h <= P['thw']:
            events.append({'type': 'close_follow', 't': float(tk), 'value': float(h), 'severity': 1 + (P['thw'] - h) / 0.3,
                           'text': f'근접 추종: 전방 {g:.1f} m, 차간시간 {h:.2f} s ({vk * 3.6:.0f} km/h)'})
    ttc = []
    for (t1, g1, _), (t2, g2, v2) in zip(gaps[:-1], gaps[1:]):
        if g1 is None or g2 is None or abs(g2 - g1) > 8 or v2 <= 3.0:
            continue                                        # ego must be moving (crossing traffic at a stop is not closing)
        c = -(g2 - g1) / (t2 - t1)
        if 0.5 < c <= v2 + 2.0:                             # a lead object cannot close faster than the ego drives
            ttc.append((g2 / c, t2, g2, c))
    if ttc:
        k, tk, g, c = min(ttc)
        metrics['min_ttc'] = float(k)
        if k <= P['ttc']:
            events.append({'type': 'closing', 't': float(tk), 'value': float(k), 'severity': 1 + (P['ttc'] - k) / 0.5,
                           'text': f'빠른 접근: 전방 {g:.1f} m를 {c * 3.6:.0f} km/h로 좁힘, TTC {k:.1f} s'})
    moving = [g for _, g, vk in gaps if g is not None and vk > 3.0]
    metrics['min_gap_moving'] = float(min(moving)) if moving else None
    events.sort(key=lambda e: -e['severity'])
    score = float(events[0]['severity']) if events else 0.0
    return {'score': round(score, 2), 'level': '높음' if score >= 2 else '주의' if score >= 1 else '',
            'events': [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()} for e in events],
            'metrics': {k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataroot', default=os.path.dirname(HERE))
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--brake', type=float, default=-3.0)
    ap.add_argument('--lat', type=float, default=3.0)
    ap.add_argument('--thw', type=float, default=0.6)
    ap.add_argument('--ttc', type=float, default=2.0)
    args = ap.parse_args()
    P = {k: getattr(args, k) for k in ('brake', 'lat', 'thw', 'ttc')}

    db = sv.load_db(args.dataroot, args.version)
    out_dir = os.path.join(args.dataroot, 'selection')
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, 'risk_cache.pkl')
    cache = {}
    if os.path.isfile(cache_path):
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        if cache.get('_version') != CACHE_VERSION:
            cache = {}
    cache['_version'] = CACHE_VERSION
    todo = [s for s in db['scenes'] if s['name'] not in cache]
    ltracks = {}
    for i, s in enumerate(todo):
        if s['log'] not in ltracks:
            ltracks[s['log']] = log_track(db, s['log'])
        cache[s['name']] = front_gaps(db, s, args.dataroot, ltracks[s['log']])
        print(f'  front gaps {s["name"]} ({i + 1}/{len(todo)})', flush=True)
    if todo:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)

    res = {s['name']: scene_risk(db, s, persistent(cache[s['name']]), P) for s in db['scenes']}
    with open(os.path.join(out_dir, 'risk.json'), 'w', encoding='utf-8') as f:
        json.dump({'created': datetime.datetime.now().isoformat(timespec='seconds'), 'params': P, 'scenes': res},
                  f, ensure_ascii=False, indent=1)
    ranked = sorted(res.items(), key=lambda kv: -kv[1]['score'])
    print(f"\n[risk] thresholds {P}; scenes with events: {sum(1 for _, r in ranked if r['score'] > 0)}/{len(ranked)}")
    for n, r in ranked:
        m = r['metrics']
        line = (f"  {n[-4:]} score {r['score']:4.2f} {r['level']:2s} | min ax {m['min_ax']:5.2f} max |ay| {m['max_ay']:4.2f}"
                f" thw {m.get('min_thw', float('nan')):5.2f} ttc {m.get('min_ttc', float('nan')):5.2f}")
        print(line + ('  <- ' + '; '.join(f"{e['text']} @{e['t']:.1f}s" for e in r['events']) if r['events'] else ''))


if __name__ == '__main__':
    sys.exit(main())
