"""Remove only redundant 20 s scenes: same place + same ego behaviour (+ similar surroundings).

A scene i may be removed only if a kept scene j can substitute it:
  1. place    : j covers i's trajectory (GPS within --place-m and heading within --heading-deg)
  2. ego      : speed / yaw-rate profiles (5 x 4 s windows, can_bus) are within --ego-tau
  3. lidar    : auxiliary, only when both scenes are (near) stationary -- the amount and spatial
                pattern of moving objects must not clearly differ. While driving, the frame-difference
                residual grows with speed (occlusion / scan pattern), so it is recorded but not used.
Scenes tagged as must-keep (departure / stopping / turn by default) are never removed.

Modes
  threshold : remove every scene that has a substitute (removal count follows the redundancy)
  budget    : keep exactly ceil(--keep-ratio * N) scenes over ALL routes; duplicates are removed
              first (most redundant first); if that is not enough, the least distinctive non-duplicate
              scenes are removed and flagged as budget-forced.

Outputs go to <dataroot>/selection/: result.json (read by the viewer), scene_features.csv, pairs.csv.
If selection/human_labels.json exists (written by the viewer's label tab), agreement is reported;
--sweep grid-searches the thresholds against those labels.

    python tools/scene_select.py [--mode threshold|budget] [--keep-ratio 0.667] [--sweep]
"""
import argparse
import csv
import datetime
import itertools
import json
import math
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sample_viewer as sv  # noqa: E402  (reuses dataset + can_bus loading)

STOP_KMH = 1.0          # below this the ego is considered stopped
STATIONARY_KMH = 5.0    # lidar criterion is applied only when both scenes average below this

# ---------------------------------------------------------------- lidar change features
R, CELL = 40.0, 0.5                 # grid half-size and cell (m), lidar frame
N = int(2 * R / CELL)
GCELL = 2.0                         # coarse cell for local ground height
GN = int(2 * R / GCELL)
BAND = (0.4, 2.5)                   # height above local ground kept as "objects"
MIN_BLOB = 4                        # cells (1 m^2); smaller change blobs are treated as noise
_c = (np.arange(N) + 0.5) * CELL - R
XC, YC = np.meshgrid(_c, _c, indexing='ij')
LIDAR_FEATURE_VERSION = 2


def _shifts(p, n):
    return [p[1 + di:n + 1 + di, 1 + dj:n + 1 + dj] for di in (-1, 0, 1) for dj in (-1, 0, 1)]


def occupancy(path):
    a = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    m = (np.abs(x) < R) & (np.abs(y) < R)
    x, y, z = x[m], y[m], z[m]
    gi = np.clip(((x + R) / GCELL).astype(np.int32), 0, GN - 1)   # float32 rounding can hit the edge
    gj = np.clip(((y + R) / GCELL).astype(np.int32), 0, GN - 1)
    g = np.full((GN, GN), np.inf, np.float32)
    np.minimum.at(g, (gi, gj), z)
    ground = np.min(np.stack(_shifts(np.pad(g, 1, constant_values=np.inf), GN)), axis=0)
    h = z - ground[gi, gj]
    k = (h > BAND[0]) & (h < BAND[1]) & (x * x + y * y > 2.5 ** 2)   # drop ego-body returns
    ci = np.clip(((x[k] + R) / CELL).astype(np.int32), 0, N - 1)
    cj = np.clip(((y[k] + R) / CELL).astype(np.int32), 0, N - 1)
    cnt = np.bincount(ci * N + cj, minlength=N * N).reshape(N, N)     # same counts as np.add.at, ~7x faster
    return cnt >= 2


def _dilate(m):
    return np.any(np.stack(_shifts(np.pad(m, 1), N)), axis=0)


def _to_frame(xs, ys, src, dst):
    """Points in pose `src`'s frame -> pose `dst`'s frame (poses are x, y, yaw in UTM)."""
    xw = math.cos(src[2]) * xs - math.sin(src[2]) * ys + src[0]
    yw = math.sin(src[2]) * xs + math.cos(src[2]) * ys + src[1]
    dx, dy = xw - dst[0], yw - dst[1]
    return math.cos(dst[2]) * dx + math.sin(dst[2]) * dy, -math.sin(dst[2]) * dx + math.cos(dst[2]) * dy


def change_blobs(occ_a, occ_b, pa, pb):
    """Cells that appeared/disappeared between two scans after ego-motion compensation (in A's frame)."""
    import cv2
    ii, jj = np.nonzero(occ_b)
    xa, ya = _to_frame(XC[ii, jj], YC[ii, jj], pb, pa)
    b_in_a = np.zeros_like(occ_b)
    k = (np.abs(xa) < R) & (np.abs(ya) < R)
    b_in_a[np.clip(((xa[k] + R) / CELL).astype(int), 0, N - 1), np.clip(((ya[k] + R) / CELL).astype(int), 0, N - 1)] = True
    bx, by = _to_frame(np.array([0.0]), np.array([0.0]), pb, pa)
    common = (XC ** 2 + YC ** 2 < 38 ** 2) & ((XC - bx[0]) ** 2 + (YC - by[0]) ** 2 < 38 ** 2)
    dyn = ((occ_a & ~_dilate(b_in_a)) | (b_in_a & ~_dilate(occ_a))) & common
    n, lab, st, _ = cv2.connectedComponentsWithStats(dyn.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= MIN_BLOB
    return keep[lab], int(keep.sum())


def pose_at(tr, t):
    yaw = np.unwrap(np.array(tr['yaw']))
    return (float(np.interp(t, tr['t'], tr['x'])), float(np.interp(t, tr['t'], tr['y'])),
            float(np.interp(t, tr['t'], yaw)))


def lidar_features(db, scene, dataroot):
    tr = db['tracks'][scene['name']]
    lids = [db['by_sample'][t]['LIDAR_TOP'] for t in scene['samples']]
    occ = [occupancy(os.path.join(dataroot, l['filename'])) for l in lids]
    poses = [pose_at(tr, l['timestamp']) for l in lids]
    cells, blobs, heat = [], [], {}
    for k in range(len(occ) - 1):
        dyn, nb = change_blobs(occ[k], occ[k + 1], poses[k], poses[k + 1])
        cells.append(int(dyn.sum())); blobs.append(nb)
        xs, ys = _to_frame(XC[dyn], YC[dyn], poses[k], (0.0, 0.0, 0.0))   # -> UTM
        for key in zip(np.floor(xs).astype(np.int64).tolist(), np.floor(ys).astype(np.int64).tolist()):
            heat[key] = heat.get(key, 0) + 1                                 # 1 m world grid
    return {'cells': cells, 'blobs': blobs, 'heat': heat}


def load_lidar_features(db, dataroot, cache_path):
    cache = {}
    if os.path.isfile(cache_path):
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        if cache.get('_version') != LIDAR_FEATURE_VERSION:
            cache = {}
    cache['_version'] = LIDAR_FEATURE_VERSION
    todo = [s for s in db['scenes'] if s['name'] not in cache]
    for i, s in enumerate(todo):
        t0 = time.time()
        cache[s['name']] = lidar_features(db, s, dataroot)
        print(f'  lidar features {s["name"]} ({i + 1}/{len(todo)}) {time.time() - t0:.1f}s', flush=True)
    if todo:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
    return cache


def heat_cosine(a, b):
    """Cosine similarity of two 1 m world heatmaps after a 3x3 box blur (1 m tolerance)."""
    def blur(h):
        out = {}
        for (x, y), c in h.items():
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    out[(x + dx, y + dy)] = out.get((x + dx, y + dy), 0) + c
        return out
    a, b = blur(a), blur(b)
    dot = sum(c * b.get(k, 0) for k, c in a.items())
    na = math.sqrt(sum(c * c for c in a.values())); nb = math.sqrt(sum(c * c for c in b.values()))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------- ego / place features
def ego_features(tr):
    t = (np.array(tr['t']) - tr['t'][0]) / 1e6
    v = np.array(tr['speed']) * 3.6
    wz = np.degrees(np.array(tr['wz']))
    yaw = np.degrees(np.unwrap(np.array(tr['yaw'])))
    ax = np.array(tr['ax'])
    k = max(1, int(round(1.0 / np.median(np.diff(t)))))             # 1 s moving average
    ker = np.ones(k) / k
    v1 = np.convolve(v, ker, mode='same')
    ax1 = np.convolve(ax, ker, mode='same')
    win_v, win_w = [], []
    for w in range(5):
        m = (t >= 4 * w) & (t < 4 * w + 4)
        win_v.append(float(v[m].mean()) if m.any() else 0.0)
        win_w.append(float(wz[m].mean()) if m.any() else 0.0)
    departure = bool(np.any((np.minimum.accumulate(v1) < STOP_KMH) & (v1 > 10)))
    stopping = bool(np.any((np.maximum.accumulate(v1) > 10) & (v1 < STOP_KMH)))
    turn_deg = 0.0
    for i in range(len(t)):
        j = np.searchsorted(t, t[i] + 6.0)
        if j >= len(t):
            break
        if v[i:j].mean() > 3:
            turn_deg = max(turn_deg, abs(yaw[j] - yaw[i]))
    tags = []
    stop_frac = float(np.mean(v1 < STOP_KMH))
    if stop_frac >= 0.95:
        tags.append('stopped')
    if departure:
        tags.append('departure')
    if stopping:
        tags.append('stopping')
    if turn_deg >= 30:
        tags.append('turn')
    if np.max(np.abs(ax1)) >= 3.0:
        tags.append('hard_accel')
    if not tags:
        tags.append('cruise' if v.mean() > 5 else 'slow')
    return {'win_v': win_v, 'win_w': win_w, 'stop_frac': stop_frac, 'v_mean': float(v.mean()),
            'v_max': float(v.max()), 'turn_deg': float(turn_deg), 'tags': tags}


def ego_distance(a, b, s_v=5.0, s_w=2.0):
    """RMS over the 5 windows of speed (per 5 km/h) and yaw-rate (per 2 deg/s) differences."""
    d = [((va - vb) / s_v) ** 2 + ((wa - wb) / s_w) ** 2
         for va, vb, wa, wb in zip(a['win_v'], b['win_v'], a['win_w'], b['win_w'])]
    return math.sqrt(sum(d) / len(d))


def place_track(tr, step=50):
    """Positions / headings every 0.5 s (100 Hz track)."""
    idx = list(range(0, len(tr['t']), step))
    return (np.array([tr['x'][i] for i in idx]), np.array([tr['y'][i] for i in idx]),
            np.degrees(np.array([tr['yaw'][i] for i in idx])))


def place_cover(pi, pj, place_m, heading_deg):
    """Share of i's points that j passes within place_m with a heading difference below heading_deg."""
    xi, yi, hi = pi
    xj, yj, hj = pj
    d = np.hypot(xi[:, None] - xj[None], yi[:, None] - yj[None])
    k = d.argmin(1)
    dist = d[np.arange(len(xi)), k]
    dh = np.abs((hi - hj[k] + 180) % 360 - 180)
    ok = (dist <= place_m) & (dh <= heading_deg)
    return float(ok.mean()), float(np.median(dist)), float(np.median(dh))


# ---------------------------------------------------------------- pairwise substitution
DEFAULTS = dict(place_m=10.0, heading_deg=30.0, cover=0.8, ego_tau=1.0,
                lidar_ratio=2.0, lidar_cos=0.3, lidar_floor=20.0, cross_route=True,
                must_keep=('departure', 'stopping', 'turn'))


def pair_table(feat, lidar, P):
    names = list(feat)
    rows = {}
    for i, j in itertools.permutations(names, 2):
        fi, fj = feat[i], feat[j]
        if not P['cross_route'] and fi['log'] != fj['log']:
            continue
        cover, dmed, hmed = place_cover(fi['place'], fj['place'], P['place_m'], P['heading_deg'])
        if cover < P['cover']:
            continue                                   # not the same place -> never a substitute
        e = ego_distance(fi['ego'], fj['ego'])
        ai, aj = float(np.mean(lidar[i]['cells'])), float(np.mean(lidar[j]['cells']))
        both_slow = fi['ego']['v_mean'] < STATIONARY_KMH and fj['ego']['v_mean'] < STATIONARY_KMH
        ratio = (max(ai, aj) + P['lidar_floor']) / (min(ai, aj) + P['lidar_floor'])
        cos = heat_cosine(lidar[i]['heat'], lidar[j]['heat']) if both_slow else None
        if not both_slow:
            lidar_ok, lidar_note = True, 'not used (driving)'
        elif max(ai, aj) <= P['lidar_floor']:
            lidar_ok, lidar_note = True, 'both quiet'
        else:
            active_both = min(ai, aj) > P['lidar_floor']
            lidar_ok = bool(ratio <= P['lidar_ratio'] and (not active_both or cos >= P['lidar_cos']))
            lidar_note = f'ratio {ratio:.2f}' + (f', cos {cos:.2f}' if active_both else '')
        ok = e <= P['ego_tau'] and lidar_ok
        lidar_pen = (math.log(ratio) / math.log(P['lidar_ratio']) if both_slow else 0.0) + \
                    (0.5 * (1 - cos) if cos is not None and min(ai, aj) > P['lidar_floor'] else 0.0)
        cost = e / P['ego_tau'] + dmed / P['place_m'] + lidar_pen
        rows[(i, j)] = dict(i=i, j=j, cover=cover, place_med_m=dmed, heading_med_deg=hmed, ego_dist=e,
                            lidar_i=ai, lidar_j=aj, lidar_ratio=ratio, heat_cos=cos, lidar_ok=lidar_ok,
                            lidar_note=lidar_note, substitutable=ok, cost=cost)
    return rows


# ---------------------------------------------------------------- selection
def components(names, edges):
    adj = {n: set() for n in names}
    for (i, j) in edges:
        adj[i].add(j); adj[j].add(i)
    seen, comps = set(), []
    for n in names:
        if n in seen:
            continue
        stack, comp = [n], []
        seen.add(n)
        while stack:
            u = stack.pop(); comp.append(u)
            for w in adj[u] - seen:
                seen.add(w); stack.append(w)
        comps.append(sorted(comp))
    return comps


def component_options(comp, sub, must):
    """All feasible kept sets of one component: {n_removed: (cost, kept, assignment)} (best per count)."""
    best = {0: (0.0, tuple(comp), {})}
    free = [n for n in comp if n not in must]
    if len(comp) > 16:
        raise RuntimeError(f'component too large for exhaustive search: {comp}')
    for r in range(1, len(free) + 1):
        for removed in itertools.combinations(free, r):
            kept = [n for n in comp if n not in removed]
            cost, assign = 0.0, {}
            for i in removed:
                cands = [(sub[(i, j)], j) for j in kept if (i, j) in sub]
                if not cands:
                    break
                c, j = min(cands)
                cost += c; assign[i] = j
            else:
                if r not in best or cost < best[r][0]:
                    best[r] = (cost, tuple(kept), assign)
    return best


def select(feat, pairs, P, mode='threshold', keep_ratio=2 / 3, soft=None):
    names = sorted(feat)
    must = {n for n in names if set(feat[n]['ego']['tags']) & set(P['must_keep'])}
    sub = {(p['i'], p['j']): p['cost'] for p in pairs.values() if p['substitutable']}
    comps = components(names, sub.keys())
    opts = [component_options(c, sub, must) for c in comps]
    max_dup = sum(max(o) for o in opts)
    n = len(names)
    target = max_dup if mode == 'threshold' else n - math.ceil(keep_ratio * n)

    # choose how many to remove from each component: exact DP over components (min total cost)
    dup_target = min(target, max_dup)
    dp = {0: (0.0, [])}
    for o in opts:
        nxt = {}
        for have, (c0, picks) in dp.items():
            for r, (c1, _, _) in o.items():
                if have + r > dup_target:
                    continue
                val = (c0 + c1, picks + [r])
                if have + r not in nxt or val[0] < nxt[have + r][0]:
                    nxt[have + r] = val
        dp = nxt
    _, picks = dp[dup_target]
    status, subst, cost, forced = {n: 'keep' for n in names}, {}, {}, set()
    for n in must:
        status[n] = 'must_keep'
    for o, r in zip(opts, picks):
        _, _, assign = o[r]
        for i, j in assign.items():
            status[i] = 'remove'; subst[i] = j; cost[i] = sub[(i, j)]

    # budget mode: not enough duplicates -> remove least distinctive scenes (greedy on L(S) over ALL scenes)
    extra = target - dup_target
    if extra > 0 and soft is not None:
        for _ in range(extra):
            kept = [x for x in names if status[x] != 'remove']
            cands = [x for x in kept if status[x] == 'keep']
            if not cands:
                break
            def loss(drop):
                ks = [x for x in kept if x != drop]
                return sum(min(soft[(a, b)] if a != b else 0.0 for b in ks) for a in names)
            drop = min(cands, key=loss)
            status[drop] = 'remove'; forced.add(drop)
    # a forced removal may take away the substitute of an earlier duplicate: re-point every removed scene to
    # its best KEPT stand-in (a real same-place substitute if one is left, else the nearest by soft distance)
    kept = [x for x in names if status[x] != 'remove']
    weak = set()
    for i in (x for x in names if status[x] == 'remove'):
        real = [(sub[(i, j)], j) for j in kept if (i, j) in sub]
        if real:
            cost[i], subst[i] = min(real)
        elif soft is not None:
            subst[i] = min(kept, key=lambda b: soft[(i, b)]); cost[i] = soft[(i, subst[i])]
            weak.add(i)
    return status, subst, cost, forced | weak, comps


def soft_distance(feat, lidar):
    """Place-free distance used only when a budget forces removing non-duplicates."""
    out = {}
    for a, b in itertools.permutations(feat, 2):
        e = ego_distance(feat[a]['ego'], feat[b]['ego'])
        la, lb = np.mean(lidar[a]['cells']), np.mean(lidar[b]['cells'])
        out[(a, b)] = e + abs(math.log((la + 20) / (lb + 20)))
    return out


# ---------------------------------------------------------------- human labels
def load_labels(path):
    if not os.path.isfile(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return {k: v for k, v in json.load(f).get('labels', {}).items() if v.get('label')}


def agreement(status, subst, labels):
    if not labels:
        return None
    lab = {k: v for k, v in labels.items() if k in status}
    human_dup = {k for k, v in lab.items() if v['label'] == 'duplicate'}
    algo_rm = {k for k in lab if status[k] == 'remove'}
    tp, fp, fn = human_dup & algo_rm, algo_rm - human_dup, human_dup - algo_rm
    prec = len(tp) / len(algo_rm) if algo_rm else None
    rec = len(tp) / len(human_dup) if human_dup else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else 0.0
    same_sub = sum(1 for k in tp if lab[k].get('dup_of') == subst.get(k))
    must_violations = sorted(k for k, v in lab.items() if v['label'] == 'must_keep' and status[k] == 'remove')
    return dict(n_labeled=len(lab), n_human_dup=len(human_dup), n_algo_removed=len(algo_rm),
                tp=sorted(tp), fp=sorted(fp), fn=sorted(fn), precision=prec, recall=rec, f1=f1,
                same_substitute=same_sub, must_keep_violations=must_violations)


# ---------------------------------------------------------------- main
def build_features(db):
    feat = {}
    for s in db['scenes']:
        tr = db['tracks'][s['name']]
        if not tr:
            continue
        feat[s['name']] = {'log': s['log'], 'ego': ego_features(tr), 'place': place_track(tr)}
    return feat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataroot', default=os.path.dirname(HERE))
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--mode', choices=['threshold', 'budget'], default='threshold')
    ap.add_argument('--keep-ratio', type=float, default=2 / 3)
    for k in ('place_m', 'heading_deg', 'cover', 'ego_tau', 'lidar_ratio', 'lidar_cos', 'lidar_floor'):
        ap.add_argument('--' + k.replace('_', '-'), type=float, default=DEFAULTS[k])
    ap.add_argument('--must-keep', default=','.join(DEFAULTS['must_keep']),
                    help='comma separated tags: departure,stopping,turn,hard_accel (empty = none)')
    ap.add_argument('--same-route-only', action='store_true', help='substitutes must come from the same log')
    ap.add_argument('--sweep', action='store_true', help='grid-search thresholds against human labels')
    args = ap.parse_args()

    out_dir = os.path.join(args.dataroot, 'selection')
    os.makedirs(out_dir, exist_ok=True)
    db = sv.load_db(args.dataroot, args.version)
    feat = build_features(db)
    print('[select] lidar change features (cached in selection/lidar_cache.pkl)', flush=True)
    lidar = load_lidar_features(db, args.dataroot, os.path.join(out_dir, 'lidar_cache.pkl'))
    P = {k: getattr(args, k) for k in ('place_m', 'heading_deg', 'cover', 'ego_tau', 'lidar_ratio', 'lidar_cos', 'lidar_floor')}
    P['cross_route'] = not args.same_route_only
    P['must_keep'] = tuple(t for t in args.must_keep.split(',') if t)
    labels = load_labels(os.path.join(out_dir, 'human_labels.json'))

    if args.sweep:
        return sweep(feat, lidar, P, labels)

    pairs = pair_table(feat, lidar, P)
    soft = soft_distance(feat, lidar) if args.mode == 'budget' else None
    status, subst, cost, forced, comps = select(feat, pairs, P, args.mode, args.keep_ratio, soft)
    agr = agreement(status, subst, labels)

    # ---- write outputs
    scenes_out = {}
    for n in sorted(feat):
        f = feat[n]
        scenes_out[n] = dict(log=f['log'], status=status[n], substitute=subst.get(n), cost=cost.get(n),
                             forced=n in forced, tags=f['ego']['tags'], stop_frac=f['ego']['stop_frac'],
                             v_mean_kmh=f['ego']['v_mean'], turn_deg=f['ego']['turn_deg'],
                             lidar_cells=float(np.mean(lidar[n]['cells'])),
                             lidar_blobs=float(np.mean(lidar[n]['blobs'])),
                             pos=[float(f['place'][0].mean()), float(f['place'][1].mean())])
    result = dict(created=datetime.datetime.now().isoformat(timespec='seconds'), mode=args.mode,
                  keep_ratio=args.keep_ratio if args.mode == 'budget' else None,
                  params={k: (list(v) if isinstance(v, tuple) else v) for k, v in P.items()},
                  n_scenes=len(feat), n_removed=sum(s == 'remove' for s in status.values()),
                  n_forced=len(forced), scenes=scenes_out,
                  groups=[c for c in comps if len(c) > 1],
                  pairs=[{k: v for k, v in p.items()} for p in pairs.values()],
                  human=agr)
    with open(os.path.join(out_dir, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, 'scene_features.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['scene', 'log', 'status', 'substitute', 'cost', 'forced', 'tags', 'stop_frac', 'v_mean_kmh',
                    'turn_deg', 'lidar_cells', 'lidar_blobs', 'win_v_kmh', 'win_yawrate_dps', 'utm_x', 'utm_y'])
        for n, s in scenes_out.items():
            e = feat[n]['ego']
            w.writerow([n, s['log'], s['status'], s['substitute'] or '', f"{s['cost']:.3f}" if s['cost'] is not None else '',
                        int(s['forced']), '|'.join(s['tags']), f"{s['stop_frac']:.3f}", f"{s['v_mean_kmh']:.1f}",
                        f"{s['turn_deg']:.1f}", f"{s['lidar_cells']:.1f}", f"{s['lidar_blobs']:.1f}",
                        ' '.join(f'{x:.1f}' for x in e['win_v']), ' '.join(f'{x:.2f}' for x in e['win_w']),
                        f"{s['pos'][0]:.1f}", f"{s['pos'][1]:.1f}"])
    with open(os.path.join(out_dir, 'pairs.csv'), 'w', newline='', encoding='utf-8') as f:
        keys = ['i', 'j', 'cover', 'place_med_m', 'heading_med_deg', 'ego_dist', 'lidar_i', 'lidar_j',
                'lidar_ratio', 'heat_cos', 'lidar_ok', 'lidar_note', 'substitutable', 'cost']
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for p in pairs.values():
            w.writerow({k: (f'{v:.3f}' if isinstance(v, float) else v) for k, v in p.items()})

    # ---- console report
    print(f"\n[select] mode={args.mode}  scenes={len(feat)}  removed={result['n_removed']}"
          + (f" (budget keep {math.ceil(args.keep_ratio * len(feat))}, forced {len(forced)})" if args.mode == 'budget' else ''))
    print(f"{'scene':11s} {'log':4s} {'status':9s} {'sub':5s} {'tags':28s} {'v_mean':>6s} {'stop%':>5s} {'lidar':>6s}")
    for n, s in scenes_out.items():
        print(f"{n:11s} {s['log'].split('_')[0]:4s} {s['status']:9s} {(s['substitute'] or '')[-4:]:5s} "
              f"{','.join(s['tags'])[:28]:28s} {s['v_mean_kmh']:6.1f} {s['stop_frac'] * 100:5.0f} {s['lidar_cells']:6.0f}"
              + ('  <- budget-forced' if s['forced'] else ''))
    print('\nsame-place pairs (i covered by j):')
    for p in sorted(pairs.values(), key=lambda p: (p['i'], p['j'])):
        print(f"  {p['i'][-4:]} <- {p['j'][-4:]}  place {p['place_med_m']:5.1f} m  ego {p['ego_dist']:5.2f}  "
              f"lidar {p['lidar_note']:24s} -> {'SUBSTITUTE' if p['substitutable'] else '-'}")
    per_log = {}
    for n, s in scenes_out.items():
        per_log.setdefault(s['log'].split('_')[0], [0, 0])[s['status'] == 'remove'] += 1
    print('\nper route (kept, removed):', {k: tuple(v) for k, v in per_log.items()})
    if agr:
        print(f"\nhuman labels: {agr['n_labeled']} labeled, precision={agr['precision']}, recall={agr['recall']}, "
              f"F1={agr['f1']:.2f}\n  FP (algo removed, human keeps)={agr['fp']}\n  FN (human dup, algo keeps)={agr['fn']}")
        if agr['must_keep_violations']:
            print('  must-keep violated:', agr['must_keep_violations'])
    else:
        print("\nhuman labels: none yet (label scenes in the viewer's label tab)")
    print(f'\nwrote {out_dir}\\result.json, scene_features.csv, pairs.csv')


def sweep(feat, lidar, P, labels):
    if not labels:
        print('no human labels (selection/human_labels.json) -- label scenes in the viewer first')
        return 1
    grid = dict(place_m=[5, 10, 20], ego_tau=[0.3, 0.5, 1.0, 1.5], lidar_ratio=[1.5, 2.0, 3.0, 1e9], lidar_cos=[0.0, 0.3, 0.5])
    res = []
    for vals in itertools.product(*grid.values()):
        Q = dict(P, **dict(zip(grid, vals)))
        pairs = pair_table(feat, lidar, Q)
        status, subst, _, _, _ = select(feat, pairs, Q)
        a = agreement(status, subst, labels)
        res.append((a['f1'], a['precision'] or 0, dict(zip(grid, vals)), a))
    res.sort(key=lambda r: (-r[0], -r[1]))
    print(f'{len(labels)} labeled scenes, {res[0][3]["n_human_dup"]} human duplicates -- small sample, treat as a guide')
    for f1, prec, q, a in res[:10]:
        print(f"  F1={f1:.2f} P={prec:.2f} R={a['recall'] or 0:.2f}  {q}  FP={a['fp']} FN={a['fn']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
