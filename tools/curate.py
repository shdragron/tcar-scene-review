"""Route-by-route scene curation.

--mode stops (default for now)
    Only stops: inside each route, scenes stopped at the same place (>= 80 % of the time stopped, same place
    + same ego state as below) keep only the one with the largest LiDAR diff (0.5 s frame difference, changed
    cells per pair); the others are deleted. Nothing else is deleted: no quota, no stage 2, no review.
    Reads only what that needs: INS tracks for the stop / place grouping, LiDAR only for members of a stop
    group (cached per scene token in selection/lidar_diff.json, missing scenes read in parallel), file sizes
    for the reduction figures (selection/scene_bytes.json). No CLIP, detector or risk results; torch never loads.

--mode quota: every incoming route keeps 2/3 and deletes 1/3.

Routes are processed in import order; everything kept so far is the accumulated corpus. Two stages,
and every deletion says which kind it is.

Stage 1 -- duplicates (중복), inside the route
    Same place (GPS track + heading) and same ego state (4 s x 5 window speed / yaw-rate profile).
    A member is deleted only if a kept scene of the group actually stands in for it. For stops the
    surroundings decide, typed (tools/stop_events.py: moving pedestrians / two-wheelers / vehicles per
    frame, 3 front cameras): representatives are added until every kind of event in the group is
    covered; a member is a duplicate if no kind exceeds the representatives' by more than --tol-*,
    a different event (kept) beyond twice that, and a human review item (kept, not deleted) in between.
    The LiDAR change count is not used for this: it is size-weighted (a turning truck makes hundreds of
    changed cells, a crowd on the crosswalk a few). Human "same situation?" pair labels override.
    Without event data the old LiDAR rule (--big-change) applies. Duplicates go quietest first.
Stage 2 -- quantity adjustment (수량 조정), against the accumulated corpus
    Situation group = condition (CLIP zero-shot day/night; rain only at p >= 0.9) x ego manoeuvre.
    Target retained share of group g ~ A_g ** gamma (A_g = scenes of g seen so far; gamma=1 keeps
    input proportions, gamma=0 equalises groups). Delete from the group most above its target, only
    scenes of the incoming route (the corpus is counted, never deleted). Inside the group the traffic
    type decides what stays (nearby road users few / some / many, pedestrian-rich or not; camera
    detections): delete from the most common type the scene closest to another of that type, so the
    only scene of a type goes last. No image similarity: across places CLIP scores sit in 0.85-0.95.
Protected (never deleted automatically): scenes with risk events (tools/scene_risk.py), stage-1
review items, --hard-keep. If the quota can only be met by deleting a protected scene, or a group
would drop below its minimum holding (--min-hold, --min-hold-event), the route gets a conflict for a
human decision instead. Quota per route: floor(n/3 + carry); the fraction carries to the next route.

Outputs: selection/result.json (viewer tabs): decisions with `detail` / `summary` / `compare`,
`review` (stage-1 pairs to judge), `conflicts`, and `coverage` (situation types before/after).

    python tools/curate.py [--gamma 0.5] [--fixed-routes A-10,A-8] [--compare]
"""
import argparse
import datetime
import glob
import itertools
import json
import math
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sample_viewer as sv  # noqa: E402
import scene_embed as se  # noqa: E402
import scene_select as ss  # noqa: E402

EVENT_CLASSES = ('departure', 'stopping', 'left_turn', 'right_turn')
CLASS_KO = {'stopped': '정차', 'departure': '출발', 'stopping': '정지', 'left_turn': '좌회전', 'right_turn': '우회전',
            'slow': '서행', 'straight_low': '저속 직진', 'straight_mid': '중속 직진', 'straight_high': '고속 직진'}
STOP_KMH, SLOW_KMH = 1.0, 5.0
PLACE = dict(place_m=10.0, heading_deg=30.0, ego_tau=1.0)   # same place: path cover within 10 m / 30 deg, ego state <= 1
EV_KINDS = (('ped', '보행자'), ('two', '이륜차'), ('veh', '차량'))
TRAFFIC_KO = {'low': '교통 적음', 'mid': '교통 보통', 'high': '교통 많음'}


# ---------------------------------------------------------------- scene description
def behaviour(tr):
    """Ego behaviour class + tags + distribution features from the 100 Hz INS track.

    A scene stopped for >= 80 % of the time is 'stopped' even if it starts/ends with a short
    manoeuvre (kept in tags); events need >= 2 s of standstill on the stopped side.
    """
    t = (np.array(tr['t']) - tr['t'][0]) / 1e6
    hz = max(1, int(round(1.0 / np.median(np.diff(t)))))
    v = np.array(tr['speed']) * 3.6
    k1 = np.ones(hz) / hz
    v1 = np.convolve(v, k1, mode='same')
    ax1 = np.convolve(np.array(tr['ax']), k1, mode='same')
    ay1 = np.convolve(np.array(tr['ay']), k1, mode='same')
    yaw_r = np.unwrap(np.array(tr['yaw']))
    yaw = np.degrees(yaw_r)
    stopped = v1 < STOP_KMH
    run = np.zeros(len(t))
    for i in range(1, len(t)):
        run[i] = run[i - 1] + (t[i] - t[i - 1]) if stopped[i] else 0.0
    departure = bool(np.any((np.maximum.accumulate(run) >= 2.0) & (v1 > 10)))
    after = np.zeros(len(t))
    for i in range(len(t) - 2, -1, -1):
        after[i] = after[i + 1] + (t[i + 1] - t[i]) if stopped[i] else 0.0
    stopping = bool(np.any((np.maximum.accumulate(v1) > 10) & (after >= 2.0)))
    left = right = 0.0
    for i in range(0, len(t), hz):
        j = np.searchsorted(t, t[i] + 6.0)
        if j >= len(t):
            break
        if v[i:j].mean() > 3:
            d = yaw[j] - yaw[i]
            left, right = max(left, d), min(right, d)
    stop_frac = float(np.mean(stopped))
    tags = [x for x, on in (('departure', departure), ('stopping', stopping),
                            ('left_turn', left >= 30), ('right_turn', right <= -30)) if on]
    vm = float(v.mean())
    if stop_frac >= 0.8:
        cls = 'stopped'
    elif tags:
        cls = tags[0]
    elif vm < SLOW_KMH:
        cls = 'slow'
    else:
        cls = 'straight_low' if vm < 20 else 'straight_mid' if vm < 40 else 'straight_high'
    mv = v1 > 3
    sel = yaw_r[mv] if mv.any() else yaw_r
    mean_yaw = math.atan2(float(np.mean(np.sin(sel))), float(np.mean(np.cos(sel))))
    return {'cls': cls, 'tags': tags, 'v_mean': vm, 'v_max': float(v.max()), 'stop_frac': stop_frac,
            'net_turn': float(yaw[-1] - yaw[0]), 'max_acc': float(np.max(np.abs(ax1))),
            'min_ax': float(ax1.min()), 'max_ay': float(np.max(np.abs(ay1))),
            'bearing': float((90 - math.degrees(mean_yaw)) % 360), 'moving': bool(mv.any()),
            'left_deg': float(left), 'right_deg': float(right)}


CONDITION_PROMPTS = {
    'light': ['a photo of a street in daylight', 'a photo of a street at night'],
    'weather': ['a photo of a street on a dry day', 'a photo of a street on a rainy day with a wet road'],
}


def conditions(emb, conf=0.9):
    """Zero-shot day/night x dry/rain from the forward-facing cameras (no labels needed).

    The non-default class (night / rain) needs probability >= conf; on this data the weather prompt is
    weak (two sunny scenes came out 'rain' at 0.58-0.71), so anything uncertain stays day / dry.
    """
    out, img = {}, {}
    for n, e in emb.items():                     # e: [time, cam, dim]; cams 0..2 = front, front-left, front-right
        v = e[:, :3].reshape(-1, e.shape[-1]).mean(0)
        img[n] = v / np.linalg.norm(v)
    for axis, prompts in CONDITION_PROMPTS.items():
        T = se.text_features(prompts)
        for n, v in img.items():
            p = np.exp(100 * (T @ v)); p /= p.sum()
            labels = ['day', 'night'] if axis == 'light' else ['dry', 'rain']
            out.setdefault(n, {})[axis] = labels[1] if p[1] >= conf else labels[0]
            out[n][axis + '_p'] = float(p[1])
    return out


def short(n):
    return n[-4:]


def group_of(feat, n):
    b = feat[n]
    if b['cond'] is None:                        # --mode stops skips the CLIP day/night check
        return b['beh']['cls']
    return f"{b['cond']['light']}-{b['cond']['weather']} · {b['beh']['cls']}"


def group_ko(g):
    if ' · ' not in g:
        return CLASS_KO.get(g, g)
    cond, cls = g.split(' · ')
    return f"{'주간' if cond.startswith('day') else '야간'}·{'비' if cond.endswith('rain') else '맑음'} · {CLASS_KO.get(cls, cls)}"


def road_users(feat, n):
    o = (feat[n].get('obj') or {}).get('mean')
    return None if o is None else o['veh_near'] + o['vru_near']


def traffic_type(feat, n, P):
    """Traffic type from camera detections (6 cameras, 5 times): nearby road users per frame and pedestrian-rich."""
    o = (feat[n].get('obj') or {}).get('mean')
    if not o:
        return None
    u = o['veh_near'] + o['vru_near']
    lvl = 'low' if u < P['traffic_cut'][0] else 'mid' if u < P['traffic_cut'][1] else 'high'
    return lvl + ('+ped' if o['vru_near'] >= P['ped_rich'] else '')


def type_ko(t):
    if not t:
        return '교통 정보 없음'
    return TRAFFIC_KO[t.split('+')[0]] + (' · 보행자 많음' if t.endswith('+ped') else '')


def risky(feat, n):
    return feat[n]['risk']['score'] >= 1


def route_of(feat, n):
    return feat[n]['log'].split('_')[0]


def pair_key(a, b):
    return '|'.join(sorted((a, b)))


# ---------------------------------------------------------------- stage 1: duplicates
def stop_activity(feat, n):
    e = feat[n].get('stop_ev')
    return None if not e else {k: e['mean'][k + '_mov'] for k, _ in EV_KINDS}


def act_text(a):
    return ' · '.join(f'{ko} {a[k]:.2f}' for k, ko in EV_KINDS)


def cover_stops(g, feat, P, labels):
    """Typed-event coverage inside a same-place stop group.

    Returns representatives (kept) and {member: (status, ref, excess, by_human)}; status is 'dup'
    (every kind within tolerance of what the representatives show), 'review' (some kind above the
    tolerance but within twice of it). A member with more than twice the tolerance of some kind -- a
    kind of event the representatives do not show -- becomes a representative itself.
    """
    act = {m: stop_activity(feat, m) for m in g}
    tol = P['tol']

    def score(m):
        return sum(act[m][k] / tol[k] for k in tol)

    def excess(m, reps):
        return {k: act[m][k] - max(act[r][k] for r in reps) for k in tol}

    def best_ref(m, reps):
        return min(reps, key=lambda r: (max((act[m][k] - act[r][k]) / tol[k] for k in tol), r))

    order = sorted(g, key=lambda m: (-score(m), m))
    reps, human, by_label = [order[0]], {}, set()
    while True:
        add = None
        for m in order:
            if m in reps:
                continue
            same = [r for r in reps if labels.get(pair_key(m, r)) is True]
            if same:
                human[m] = same[0]
                continue
            ex = excess(m, reps)
            if labels.get(pair_key(m, best_ref(m, reps))) is False:
                add = m; by_label.add(m)
                break
            if any(ex[k] > 2 * tol[k] for k in tol):
                add = m
                break
        if add is None:
            break
        reps.append(add)
    status = {}
    for m in order:
        if m in reps:
            continue
        if m in human:
            status[m] = ('dup', human[m], excess(m, reps), True)
            continue
        ex = excess(m, reps)
        status[m] = ('dup' if all(ex[k] <= tol[k] for k in tol) else 'review', best_ref(m, reps), ex, False)
    return reps, status, act, by_label


def same_place_groups(names, feat, P):
    """Groups (size > 1) of scenes that cover each other's path (>= 80 % within place_m / heading_deg) with a
    4 s x 5 window ego state difference <= ego_tau, chained; plus the pairwise metrics."""
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    pm = {}
    for a, b in itertools.combinations(names, 2):
        ca, da, ha = ss.place_cover(feat[a]['place'], feat[b]['place'], P['place_m'], P['heading_deg'])
        cb, db_, hb = ss.place_cover(feat[b]['place'], feat[a]['place'], P['place_m'], P['heading_deg'])
        e = ss.ego_distance(feat[a]['ego'], feat[b]['ego'])
        pm[(a, b)] = pm[(b, a)] = dict(cover=(ca, cb), dmed=max(da, db_), hmed=max(ha, hb), ego=e)
        if ca >= 0.8 and cb >= 0.8 and e <= P['ego_tau']:
            parent[find(a)] = find(b)
    comps = {}
    for n in names:
        comps.setdefault(find(n), []).append(n)
    return [sorted(x) for x in comps.values() if len(x) > 1], pm


def stage1(route, feat, P, labels):
    """Same-place / same-state groups inside the route -> (duplicates ordered, groups, review, kept_diff)."""
    names = [n for n in route if feat[n]['beh']['cls'] not in EVENT_CLASSES]
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    pm = {}
    for a, b in itertools.combinations(names, 2):
        ca, da, ha = ss.place_cover(feat[a]['place'], feat[b]['place'], P['place_m'], P['heading_deg'])
        cb, db_, hb = ss.place_cover(feat[b]['place'], feat[a]['place'], P['place_m'], P['heading_deg'])
        e = ss.ego_distance(feat[a]['ego'], feat[b]['ego'])
        pm[(a, b)] = pm[(b, a)] = dict(cover=(ca, cb), dmed=max(da, db_), hmed=max(ha, hb), ego=e)
        if ca >= 0.8 and cb >= 0.8 and e <= P['ego_tau']:
            parent[find(a)] = find(b)
    comps = {}
    for n in names:
        comps.setdefault(find(n), []).append(n)
    groups = [sorted(x) for x in comps.values() if len(x) > 1]
    dups, review, diff = [], [], []
    for g in groups:
        still = all(feat[m]['beh']['v_mean'] < SLOW_KMH for m in g)
        if still and all(stop_activity(feat, m) for m in g):
            reps, status, act, by_label = cover_stops(g, feat, P, labels)
            for m, (st, ref, ex, human) in status.items():
                c = dict(m=m, rep=ref, reps=reps, group=g, mode='events', act=act, excess=ex, human=human, **pm[(m, ref)])
                if st == 'dup':
                    dups.append(dict(c, key=(sum(act[m][k] / P['tol'][k] for k in P['tol']), m)))
                else:
                    review.append(c)
            for r in reps[1:]:
                ref = min(reps[:reps.index(r)], key=lambda x: pm[(r, x)]['ego'])
                ex = {k: act[r][k] - max(act[x][k] for x in reps[:reps.index(r)]) for k in P['tol']}
                diff.append(dict(m=r, rep=ref, reps=reps, group=g, mode='events', act=act, excess=ex, human=r in by_label))
            continue
        if still:                                  # no event data: LiDAR change amount (old rule)
            rep = max(g, key=lambda m: feat[m]['act'])
        else:
            rep = min(g, key=lambda m: sum(pm[(m, x)]['ego'] for x in g if x != m))
        for m in g:
            if m == rep:
                continue
            lab = labels.get(pair_key(m, rep))
            c = dict(m=m, rep=rep, reps=[rep], group=g, mode='lidar' if still else 'ego', human=lab is not None, **pm[(m, rep)])
            if lab is False or (lab is None and still and feat[m]['act'] > P['big_change']):
                diff.append(c)
            else:
                dups.append(dict(c, key=(feat[m]['act'] if still else pm[(m, rep)]['ego'], m)))
    dups.sort(key=lambda c: c['key'])
    return dups, groups, review, diff


def place_lines(c, P, feat):
    bm, br = feat[c['m']]['beh'], feat[c['rep']]['beh']
    return [f"장소: 서로의 경로 커버 {c['cover'][0]:.0%} / {c['cover'][1]:.0%} (기준 80% 이상), 중앙 거리 {c['dmed']:.1f} m, "
            f"방향 차이 {c['hmed']:.0f}° (기준 {P['place_m']:.0f} m·{P['heading_deg']:.0f}° 이내)",
            f"주행 상태: 4초×5구간 속도·yaw rate 차이 {c['ego']:.2f} (기준 {P['ego_tau']:.1f} 이하) · "
            f"평균 속도 {bm['v_mean']:.1f} / {br['v_mean']:.1f} km/h · 정차 {bm['stop_frac']:.0%} / {br['stop_frac']:.0%}"]


def tol_text(P):
    return ' · '.join(f"{ko} +{P['tol'][k]:g}" for k, ko in EV_KINDS)


def stage1_detail(c, feat, P, Q, rank, order):
    m, rep = c['m'], c['rep']
    if c['mode'] == 'events':
        a = c['act']
        lines = ['1단계 중복 · 같은 route 안 같은 장소·같은 상태의 정차, 남긴 scene이 주변 사건까지 대신함',
                 f"남긴 대표: {', '.join(short(r) for r in c['reps'])} (같은 장소 그룹 [{', '.join(short(x) for x in c['group'])}]; "
                 f"그룹에 있는 사건 종류를 모두 덮을 때까지 대표 추가)",
                 f"정차 중 움직이는 객체 (프레임당, 전방 카메라 3대): 이 scene {act_text(a[m])} / "
                 f"대표 {short(rep)} {act_text(a[rep])}",
                 f"모든 종류가 대표들의 최대치 + 허용치 이내 (허용 {tol_text(P)}) → 대표가 사건을 대신함"
                 + (' · 사람이 같음으로 표시' if c.get('human') else '')]
    else:
        lines = ['1단계 중복 · 같은 route 안에서 같은 장소·같은 상태가 반복됨',
                 f"대표 {short(rep)}: 같은 장소 그룹 [{', '.join(short(x) for x in c['group'])}] 중 "
                 + ('LiDAR 변화가 가장 큰 scene (정차 사건 정보 없음)' if c['mode'] == 'lidar' else '주행 상태가 가장 가운데인 scene')]
        if c['mode'] == 'lidar':
            lines.append(f"주변 변화(LiDAR): {feat[m]['act']:.0f} ≤ 기준 {P['big_change']:.0f} (대표 {feat[rep]['act']:.0f})")
    lines += place_lines(c, P, feat)
    lines.append(f"순서: 이 route 할당 {Q}개 중 {rank}번째 (조용한 순서: {order})")
    return lines


def review_text(c, P):
    a, m, r = c['act'], c['m'], c['rep']
    over = [f"{ko} {a[m][k]:.2f} vs {max(a[x][k] for x in c['reps']):.2f}" for k, ko in EV_KINDS if c['excess'][k] > P['tol'][k]]
    return (f"대표 {short(r)}보다 많은 사건 (프레임당 움직이는 수): {', '.join(over)} — 허용치는 넘고 2배 이내 "
            f"→ 대신하는지 불확실, 판단 전까지 유지")


# ---------------------------------------------------------------- stage 2: quantity adjustment
def ego_dist(a, b, P):
    return math.sqrt(np.mean([((a['v_mean'] - b['v_mean']) / P['tol_v']) ** 2, ((a['stop_frac'] - b['stop_frac']) / P['tol_stop']) ** 2,
                              ((a['net_turn'] - b['net_turn']) / P['tol_turn']) ** 2, ((a['max_acc'] - b['max_acc']) / P['tol_acc']) ** 2]))


def comp_dist(feat, a, b):
    """Traffic-composition difference (log scale): nearby vehicles and nearby pedestrians/cycles."""
    oa, ob = (feat[a].get('obj') or {}).get('mean'), (feat[b].get('obj') or {}).get('mean')
    if not oa or not ob:
        return 0.0
    dv = (math.log1p(oa['veh_near']) - math.log1p(ob['veh_near'])) / 0.35
    dp = (math.log1p(oa['vru_near']) - math.log1p(ob['vru_near'])) / 0.7
    return math.sqrt((dv * dv + dp * dp) / 2)


def scene_dist(feat, a, b, P):
    """How replaceable: hypot(ego behaviour RMS, traffic composition RMS); 1.0 = one side off by one unit on average."""
    return math.hypot(ego_dist(feat[a]['beh'], feat[b]['beh'], P), comp_dist(feat, a, b))


DIST_UNIT = ('거리 1 ≈ 자차 또는 교통 한쪽이 평균 1단위 다른 정도; 1단위 = 속도 10 km/h · 정차 20%p · 회전 30° · '
             '가감속 1.5 m/s² · 가까운 차량 1.4배 · 보행자 2배')


def min_hold(feat, n, P):
    return P['min_hold_event'] if feat[n]['beh']['cls'] in EVENT_CLASSES else P['min_hold']


def stage2(new, retained, seen, feat, q, budget_after, P, protected, corpus_n):
    """Delete q scenes of `new` -- the incoming route only -- by situation-group quota.

    The accumulated corpus (everything in `retained` that is not in `new`) is only counted. Returns the
    deletions and, if protected scenes or minimum holdings stop it short, a conflict record.
    """
    route_set, rname = set(new), (route_of(feat, new[0]) if new else '?')
    A = Counter(group_of(feat, n) for n in seen)
    wsum = sum(a ** P['gamma'] for a in A.values())
    target = {g: budget_after * a ** P['gamma'] / wsum for g, a in A.items()}
    R = Counter(group_of(feat, n) for n in retained)
    deleted, conflict = [], None
    for _ in range(q):
        live = [n for n in new if n in retained]
        cand = [n for n in live if n not in protected and R[group_of(feat, n)] - 1 >= min_hold(feat, n, P)]
        if not cand:
            left = q - len(deleted)
            prot = [n for n in live if n in protected]
            held = [n for n in live if n not in protected]
            why = []
            if prot:
                why.append('보호 scene만 남음: ' + ', '.join(f"{short(n)}({protected[n]})" for n in prot))
            if held:
                why.append('최소 보유량 때문에 못 지움: ' + ', '.join(f"{short(n)}({group_ko(group_of(feat, n)).split(' · ')[1]})" for n in held))
            nxt = sorted(prot, key=lambda n: (-(R[group_of(feat, n)] - target.get(group_of(feat, n), 0)), n))[:3]
            conflict = dict(route=rname, missing=left, protected=prot, min_hold=held, next=nxt,
                            text=f"{rname}: 할당 {left}개 미달 — " + ' / '.join(why)
                                 + ' → 자동 삭제하지 않음. 보호를 풀지(예: ' + ', '.join(short(n) for n in nxt) + ') 미달로 둘지 결정 필요')
            break
        # a deletion must not make a (group, traffic type) disappear while another candidate avoids that
        gt = Counter((group_of(feat, m), traffic_type(feat, m, P)) for m in retained)
        safe = [x for x in cand if gt[(group_of(feat, x), traffic_type(feat, x, P))] >= 2]
        pool = safe or cand
        cgroups = sorted({group_of(feat, n) for n in pool}, key=lambda g: (-(R[g] - target[g]), -R[g], g))
        g = cgroups[0]
        skipped_groups = [x for x in sorted({group_of(feat, n) for n in cand}, key=lambda g: (-(R[g] - target[g]), g))
                          if R[x] - target[x] > R[g] - target[g]]
        mates = [m for m in retained if group_of(feat, m) == g]
        types = Counter(traffic_type(feat, m, P) for m in mates)

        def rank(x):
            same_t = [m for m in mates if m != x and traffic_type(feat, m, P) == traffic_type(feat, x, P)]
            near = min(same_t or [m for m in mates if m != x], key=lambda m: scene_dist(feat, x, m, P))
            return (-types[traffic_type(feat, x, P)], scene_dist(feat, x, near, P), x), near
        ranked = sorted((rank(x) for x in pool if group_of(feat, x) == g), key=lambda r: r[0])
        (_, d, n), near = ranked[0]
        t = traffic_type(feat, n, P)
        old_m = sorted(m for m in mates if m not in route_set)
        new_m = sorted(m for m in mates if m in route_set)
        type_line = ' · '.join(f"{type_ko(k)} {v}개({', '.join(short(m) for m in sorted(mates) if traffic_type(feat, m, P) == k)})"
                               for k, v in sorted(types.items(), key=lambda kv: (-kv[1], str(kv[0]))))
        on, ob = (feat[n].get('obj') or {}).get('mean') or {}, (feat[near].get('obj') or {}).get('mean') or {}
        bn, bb = feat[n]['beh'], feat[near]['beh']
        excess = R[g] - target[g]
        detail = [f'2단계 수량 조정 · 이번 route {rname}에서 덜어냄 (중복이라서가 아니라 상황 개수를 맞추려고; 누적 데이터는 개수 비교에만 사용)',
                  f"상황 그룹: {group_ko(g)} — 누적 {len(old_m)}개({', '.join(f'{short(m)}·{route_of(feat, m)}' for m in old_m) or '없음'})"
                  f" + 이번 route {len(new_m)}개({', '.join(short(m) for m in new_m)})",
                  f"목표 {target[g]:.1f}개 = 남길 총량 {budget_after}(누적 {corpus_n} + 이번 route {budget_after - corpus_n})"
                  f" × {A[g]}^{P['gamma']} ÷ Σ(상황별 입력^{P['gamma']}) {wsum:.2f} → 초과 {excess:+.1f}개"
                  + (', 가장 큰 초과' if not skipped_groups else '')
                  + (' (다음: ' + ', '.join(f"{group_ko(x).split(' · ')[1]} {R[x] - target[x]:+.1f}" for x in cgroups[1:3]) + ')' if len(cgroups) > 1 else ''),
                  f"그룹 안 교통 유형: {type_line} → " + (f"가장 많은 '{type_ko(t)}'에서 고름" if types[t] > 1 else
                  f"후보가 모두 그룹 안 유일한 유형 → '{type_ko(t)}'이 이 그룹에서 사라짐 (분포 손실, 거리가 가장 가까운 후보)"),
                  f"{'같은 유형' if types[t] > 1 else '그룹'} 중 다른 scene과 가장 가까운 것: {short(n)} ↔ {short(near)}({'이번 route' if near in route_set else '누적 ' + route_of(feat, near)}) "
                  f"거리 {d:.2f} (평균 속도 {bn['v_mean']:.0f}/{bb['v_mean']:.0f} km/h · 회전 {bn['net_turn']:+.0f}°/{bb['net_turn']:+.0f}° · "
                  f"가까운 차량 {on.get('veh_near', 0):.0f}/{ob.get('veh_near', 0):.0f} · 보행자 {on.get('vru_near', 0):.1f}/{ob.get('vru_near', 0):.1f})"]
        if skipped_groups:
            detail.append('초과가 더 큰 ' + ', '.join(f"{group_ko(x).split(' · ')[1]}({R[x] - target[x]:+.1f})" for x in skipped_groups)
                          + '은 후보를 지우면 그 교통 유형이 사라져서 건너뜀')
        if len(ranked) > 1:
            detail.append('그룹의 다른 후보: ' + ', '.join(f"{short(r[0][2])} ({type_ko(traffic_type(feat, r[0][2], P))} {-r[0][0]}개, 거리 {r[0][1]:.2f})"
                                                      for r in ranked[1:]))
        skipped = sorted(x for x in live if x in protected and group_of(feat, x) == g)
        if skipped:
            detail.append('보호되어 후보에서 뺀 scene: ' + ', '.join(f"{short(x)}({protected[x]})" for x in skipped))
        if excess <= 0:
            detail.append('주의: 이 그룹도 목표 이하 — 1/3 할당을 채우려고 삭제')
        others = sorted((m for m in mates if m != n), key=lambda m: scene_dist(feat, n, m, P))
        deleted.append((n, detail, [near] + [m for m in others if m != near],
                        f"{group_ko(g).split(' · ')[1]} 초과 {excess:+.1f} · {type_ko(t)} {types[t]}개 중"
                        + (' (분포 손실)' if types[t] == 1 else '')))
        retained.discard(n); R[g] -= 1
    return deleted, conflict


# ---------------------------------------------------------------- driver
def route_order(db, dataroot):
    created = {}
    for f in glob.glob(os.path.join(dataroot, '*.import.json')):
        with open(f, encoding='utf-8') as fh:
            created[os.path.basename(f).split('.import.json')[0]] = json.load(fh).get('created_at', '')
    logs = []
    for s in db['scenes']:
        if s['log'] not in logs:
            logs.append(s['log'])
    return sorted(logs, key=lambda l: (created.get(l, '~'), logs.index(l)))


def run(db, feat, P, labels):
    order = route_order(db, P['dataroot'])
    by_log = {l: sorted([s['name'] for s in db['scenes'] if s['log'] == l], key=lambda n: feat[n]['t0']) for l in order}
    corpus, seen, carry = [], [], 0.0
    routes, decisions, review, conflicts, stop_groups = [], {}, [], [], []
    for log_name in order:
        route = by_log[log_name]
        n = len(route)
        if any(log_name.split('_')[0] == f for f in P.get('fixed', ())):
            # already-stored route: joins the accumulated data as is (nothing deleted, no quota)
            corpus += route; seen += route
            routes.append({'log': log_name, 'n': n, 'quota': 0, 'fixed': True, 'carry_in': round(carry, 3), 'carry_out': round(carry, 3),
                           'stage1': [], 'stage2': [], 'review': [], 'conflict': None, 'stage1_groups': [],
                           'notes': ['누적 데이터로 고정 — 삭제하지 않음']})
            continue
        exact = n / 3 + carry
        Q = int(math.floor(exact + 1e-9)); carry_out = exact - Q
        notes = []
        dups, groups1, rev, diff = stage1(route, feat, P, labels)
        protected = {m: '위험 이벤트' for m in route if risky(feat, m)}
        protected.update({m: '필수 유지' for m in route if m in P['hard_keep']})
        d1 = [c for c in dups if c['m'] not in protected][:Q]
        order_txt = ' → '.join(short(c['m']) for c in dups)
        for rank, c in enumerate(d1, 1):
            a = c.get('act')
            decisions[c['m']] = {'stage': 1, 'detail': stage1_detail(c, feat, P, Q, rank, order_txt),
                                 'compare': [c['rep']] + [x for x in c['reps'] if x != c['rep']]
                                            + [x for x in c['group'] if x not in c['reps'] and x != c['m']],
                                 'summary': f"중복 · 대표 {short(c['rep'])} — 같은 장소·상태"
                                            + (', 정차 중 사건까지 포함' if a else '')}
        for c in dups[len(d1):]:
            if c['m'] in protected:
                notes.append(f"{short(c['m'])}: 중복이지만 보호({protected[c['m']]}) → 유지")
            elif c['m'] not in decisions:
                notes.append(f"{short(c['m'])}: 중복이지만 할당({Q}개)을 이미 채움 → 유지")
        for c in rev:
            protected[c['m']] = '검토 대기'
            review.append({'scene': c['m'], 'ref': c['rep'], 'reps': c['reps'], 'route': route_of(feat, c['m']),
                           'text': review_text(c, P), 'excess': c['excess'],
                           'act': {x: c['act'][x] for x in c['group']}})
        for c in diff:
            if c['mode'] == 'events' and not c.get('human'):
                why = ', '.join(f"{ko} +{c['excess'][k]:.2f}" for k, ko in EV_KINDS if c['excess'][k] > 2 * P['tol'][k])
            else:
                why = '사람 판단: 다름' if c.get('human') else f"LiDAR 변화 {feat[c['m']]['act']:.0f} > {P['big_change']:.0f}"
            notes.append(f"{short(c['m'])}: 대표 {short(c['rep'])} — 같은 장소 정차지만 다른 사건 ({why}) → 대표로 함께 유지")
        for g in groups1:
            if all(stop_activity(feat, m) for m in g) and all(feat[m]['beh']['v_mean'] < SLOW_KMH for m in g):
                stop_groups.append(g)
        after1 = [m for m in route if m not in decisions]
        seen += after1
        retained = set(corpus) | set(after1)
        q = max(0, Q - len(d1))
        budget_after = len(corpus) + n - Q
        d2, conflict = stage2(after1, retained, seen, feat, q, budget_after, P, protected, corpus_n=len(corpus))
        for m, detail, compare, summary in d2:
            decisions[m] = {'stage': 2, 'detail': detail, 'compare': compare, 'summary': '수량 조정 · ' + summary}
        if conflict:
            conflicts.append(conflict)
        kept = [m for m in route if m not in decisions]
        alive = set(corpus) | set(kept)
        for m in route:                          # comparison scenes: finally retained ones first
            d = decisions.get(m)
            if not d:
                continue
            first = d['compare'][0] if d['compare'] else None
            d['compare'] = [x for x in d['compare'] if x in alive] + [x for x in d['compare'] if x not in alive]
            d['evidence'] = d['compare'][0] if d['compare'] else None
            e = d['evidence']
            if d['stage'] == 2 and e:
                # stage 2 deletes by group excess, not because a look-alike exists: say whether one does
                dist = scene_dist(feat, m, e, P)
                d['sub_dist'] = round(dist, 2)
                if first and first not in alive:
                    d['detail'].append(f"최종 비교 대상: 결정 당시 가장 가까웠던 {short(first)}도 이후 삭제됨 → 남은 scene 중 가장 가까운 {short(e)}")
                if dist <= P['sim_d']:
                    d['detail'].append(f"비슷한 scene 있음: {short(e)}까지 거리 {dist:.2f} ≤ {P['sim_d']:.1f} ({DIST_UNIT})")
                    d['summary'] += f" · 비슷한 {short(e)} 거리 {dist:.2f}"
                else:
                    d['detail'].append(f"비슷한 scene 없음: 가장 가까운 {short(e)}도 거리 {dist:.2f} > {P['sim_d']:.1f} ({DIST_UNIT})")
                    d['summary'] += f" · 비슷한 scene 없음 (가장 가까운 {short(e)} {dist:.2f})"
        corpus += kept
        routes.append({'log': log_name, 'n': n, 'quota': Q, 'carry_in': round(exact - n / 3, 3), 'carry_out': round(carry_out, 3),
                       'stage1': [c['m'] for c in d1], 'stage2': [m for m, *_ in d2], 'review': [c['m'] for c in rev],
                       'conflict': conflict, 'stage1_groups': groups1, 'notes': notes})
        carry = carry_out
    return dict(routes=routes, decisions=decisions, corpus=corpus, seen=seen, review=review, conflicts=conflicts,
                stop_groups=stop_groups)


def _load_cache(path, version):
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            j = json.load(f)
        if isinstance(j, dict) and j.get('version') == version:
            return j['scenes']
    return {}


def _save_cache(path, version, scenes):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'version': version, 'scenes': scenes}, f, ensure_ascii=False, indent=1)


def scene_bytes(db, dataroot, out_dir):
    """Bytes of every sample_data file of a scene (key frames + sweeps, all channels). Cached per scene token
    in selection/scene_bytes.json (names can be reassigned by a re-import, tokens cannot); a new route only
    stats its own ~3.7k files per scene."""
    path = os.path.join(out_dir, 'scene_bytes.json')
    cache = _load_cache(path, 2)
    todo = [s for s in db['scenes'] if s['token'] not in cache]
    for s in todo:
        files = [fn for lst in db['frames'].get(s['name'], {}).values() for _, fn in lst]
        cache[s['token']] = {'name': s['name'], 'bytes': sum(os.path.getsize(os.path.join(dataroot, fn)) for fn in files),
                             'files': len(files)}
    if todo:
        _save_cache(path, 2, cache)
    return {s['name']: cache[s['token']] for s in db['scenes']}


def lidar_diff(db, dataroot, scenes, out_dir, workers=4):
    """Mean LiDAR diff (changed cells per 0.5 s frame pair, scene_select.lidar_features) for `scenes` only.

    Cached per scene token in selection/lidar_diff.json. Reading the ~155 MB of LiDAR key frames per scene is
    most of the cost, so missing scenes are processed in a few threads (numpy / OpenCV release the GIL)."""
    path = os.path.join(out_dir, 'lidar_diff.json')
    cache = _load_cache(path, ss.LIDAR_FEATURE_VERSION)
    todo = [s for s in scenes if s['token'] not in cache]
    if todo:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for s, f in zip(todo, ex.map(lambda s: ss.lidar_features(db, s, dataroot), todo)):
                cache[s['token']] = {'name': s['name'], 'diff': float(np.mean(f['cells']))}
        _save_cache(path, ss.LIDAR_FEATURE_VERSION, cache)
    return {s['name']: cache[s['token']]['diff'] for s in scenes}


def run_stops(db, feat, P):
    """--mode stops: inside each route, scenes stopped at the same place (class 'stopped', same place + same ego
    state) keep only the one with the largest LiDAR diff; the others are deleted. Nothing else is deleted."""
    order = route_order(db, P['dataroot'])
    routes, decisions, corpus, seen, stop_groups = [], {}, [], [], []
    for log_name in order:
        route = sorted([s['name'] for s in db['scenes'] if s['log'] == log_name], key=lambda n: feat[n]['t0'])
        groups, pm = same_place_groups([n for n in route if feat[n]['beh']['cls'] == 'stopped'], feat, P)
        dels, notes = [], []
        for g in groups:
            stop_groups.append(g)
            rep = max(g, key=lambda m: (feat[m]['act'], m))
            ranking = ' > '.join(f"{short(m)} {feat[m]['act']:.0f}" for m in sorted(g, key=lambda m: -feat[m]['act']))
            for m in sorted(g, key=lambda m: feat[m]['act']):
                if m == rep:
                    continue
                if m in P['hard_keep']:
                    notes.append(f'{short(m)}: 필수 유지로 지정 → 유지')
                    continue
                c = dict(m=m, rep=rep, **pm[(m, rep)])
                decisions[m] = {'stage': 1, 'kind': '정지 중복',
                                'detail': [f"정지 중복 · 같은 장소에 멈춘 scene [{', '.join(short(x) for x in g)}] 중 "
                                           f"LiDAR diff가 가장 큰 {short(rep)}만 남김",
                                           f'LiDAR diff (0.5초 간격 프레임 차, 바뀐 칸 수/쌍): {ranking}'] + place_lines(c, P, feat),
                                'compare': [rep] + [x for x in g if x not in (m, rep)],
                                'summary': f"정지 중복 · 남긴 {short(rep)} diff {feat[rep]['act']:.0f} / 이 scene {feat[m]['act']:.0f}"}
                dels.append(m)
        corpus += [m for m in route if m not in decisions]
        seen += route
        routes.append({'log': log_name, 'n': len(route), 'quota': len(dels), 'carry_in': 0.0, 'carry_out': 0.0,
                       'stage1': dels, 'stage2': [], 'review': [], 'conflict': None, 'stage1_groups': groups, 'notes': notes})
    for d in decisions.values():
        d['evidence'] = d['compare'][0]
    return dict(routes=routes, decisions=decisions, corpus=corpus, seen=seen, review=[], conflicts=[], stop_groups=stop_groups)


def coverage(feat, names, kept, P, stop_groups):
    """Situation types (group x traffic type) before / after, and stop-event kinds per same-place stop group."""
    rows = {}
    for n in names:
        key = f"{group_ko(group_of(feat, n))} · {type_ko(traffic_type(feat, n, P))}"
        r = rows.setdefault(key, {'input': [], 'kept': []})
        r['input'].append(n)
        if n in kept:
            r['kept'].append(n)
    stops = []
    for g in stop_groups:
        a = {m: stop_activity(feat, m) for m in g}
        k_in = [m for m in g if m in kept]
        lost = [ko for k, ko in EV_KINDS
                if max(a[m][k] for m in g) - max((a[m][k] for m in k_in), default=0.0) > P['tol'][k]]
        stops.append({'group': g, 'kept': k_in, 'lost': lost,
                      'max_in': {k: max(a[m][k] for m in g) for k, _ in EV_KINDS},
                      'max_kept': {k: max((a[m][k] for m in k_in), default=0.0) for k, _ in EV_KINDS}})
    return {'types': rows, 'lost_types': sorted(k for k, r in rows.items() if not r['kept']), 'stops': stops}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataroot', default=os.path.dirname(HERE))
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--mode', choices=['stops', 'quota'], default='stops',
                    help='stops: only same-place stops, keep the largest LiDAR diff; quota: two-stage 1/3 curation')
    ap.add_argument('--gamma', type=float, default=0.5, help='1 = keep input proportions, 0 = equal groups')
    ap.add_argument('--min-hold', type=int, default=1, help='minimum retained scenes per situation group')
    ap.add_argument('--min-hold-event', type=int, default=2, help='minimum for departure / stopping / turn groups')
    ap.add_argument('--tol-ped', type=float, default=0.5, help='stage 1: moving pedestrians per frame a duplicate may exceed its representatives by')
    ap.add_argument('--tol-two', type=float, default=0.15, help='stage 1: same for two-wheelers')
    ap.add_argument('--tol-veh', type=float, default=1.5, help='stage 1: same for vehicles')
    ap.add_argument('--big-change', type=float, default=350.0,
                    help='stage 1 fallback without stop events: LiDAR change (cells/pair) above which a stop is not a plain repeat')
    ap.add_argument('--traffic-cut', default='25,35', help='stage 2 traffic type: nearby road users per frame, few < a <= some < b <= many')
    ap.add_argument('--ped-rich', type=float, default=5.0, help='stage 2 traffic type: nearby pedestrians/cycles per frame for "pedestrian-rich"')
    ap.add_argument('--hard-keep', default='', help='comma separated scene names that must never be deleted')
    ap.add_argument('--fixed-routes', default='',
                    help='comma separated route prefixes (e.g. A-10,A-8) that are already-stored accumulated data: '
                         'kept as is, only counted; the remaining routes are the new ones that get thinned')
    ap.add_argument('--compare', action='store_true', help='also run gamma variants and report overlaps')
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors='replace')     # cp949 consoles: an odd character must not abort the run
    except AttributeError:
        pass

    stops = args.mode == 'stops'
    db = sv.load_db(args.dataroot, args.version, ego_poses=False)
    base = ss.build_features(db)
    beh = {s['name']: behaviour(db['tracks'][s['name']]) for s in db['scenes']}
    out_dir = os.path.join(args.dataroot, 'selection')

    def load(name, what):
        p = os.path.join(out_dir, name)
        if os.path.isfile(p):
            with open(p, encoding='utf-8') as f:
                return json.load(f)
        print(f'[curate] selection/{name} missing -> {what}')
        return {}
    if stops:
        # only what the stop filter needs: no CLIP (torch never loads), no detector / risk results,
        # LiDAR diff only for the stopped scenes (reading LiDAR is most of the cost)
        cond = {n: None for n in beh}
        risk, objects, stop_ev, labels = {}, {}, {}, {}
        members = set()                          # a stop with no same-place partner needs no diff at all
        for log in {s['log'] for s in db['scenes']}:
            st = [s['name'] for s in db['scenes'] if s['log'] == log and beh[s['name']]['cls'] == 'stopped']
            members |= {m for g in same_place_groups(st, base, PLACE)[0] for m in g}
        act = lidar_diff(db, args.dataroot, [s for s in db['scenes'] if s['name'] in members], out_dir)
    else:
        lidar = ss.load_lidar_features(db, args.dataroot, os.path.join(out_dir, 'lidar_cache.pkl'))
        act = {s['name']: float(np.mean(lidar[s['name']]['cells'])) for s in db['scenes']}
        cond = conditions(se.load(db, args.dataroot))
        risk = load('risk.json', 'run tools/scene_risk.py (no risk protection)').get('scenes', {})
        objects = load('objects.json', 'run tools/scene_objects.py (no traffic types)').get('scenes', {})
        stop_ev = load('stop_events.json', 'run tools/stop_events.py (stage 1 falls back to LiDAR change)').get('scenes', {})
        labels = {k: v['same'] for k, v in load('pair_labels.json', 'no human pair labels yet').get('pairs', {}).items()}
    feat = {}
    for s in db['scenes']:
        n = s['name']
        feat[n] = dict(base[n], beh=beh[n], cond=cond[n], act=act.get(n), t0=s['sample_ts'][0],
                       risk=risk.get(n, {'score': 0.0, 'events': []}), obj=objects.get(n), stop_ev=stop_ev.get(n))
    cut = tuple(float(x) for x in args.traffic_cut.split(','))
    P = dict(dataroot=args.dataroot, gamma=args.gamma, min_hold=args.min_hold, min_hold_event=args.min_hold_event,
             tol={'ped': args.tol_ped, 'two': args.tol_two, 'veh': args.tol_veh}, big_change=args.big_change,
             traffic_cut=cut, ped_rich=args.ped_rich,
             hard_keep=set(x for x in args.hard_keep.split(',') if x), fixed=tuple(x for x in args.fixed_routes.split(',') if x),
             **PLACE, sim_d=1.0,
             tol_v=10.0, tol_stop=0.2, tol_turn=30.0, tol_acc=1.5)

    res = run_stops(db, feat, P) if stops else run(db, feat, P, labels)
    decisions, corpus = res['decisions'], res['corpus']
    names = sorted(feat)
    kept = set(names) - set(decisions)
    cov = None if stops else coverage(feat, names, kept, P, res['stop_groups'])
    sizes = scene_bytes(db, args.dataroot, out_dir)
    size = lambda ns: sum(sizes[n]['bytes'] for n in ns)
    for r in res['routes']:                       # how much each route shrinks (scene count and bytes)
        rn = [n for n in names if feat[n]['log'] == r['log']]
        r['bytes'] = [size(rn), size([n for n in rn if n in kept])]
    reduction = {'scenes': [len(names), len(kept)], 'bytes': [size(names), size(kept)]}
    print(f"\n[curate] {len(names)} -> {len(kept)} scenes (-{1 - len(kept) / len(names):.1%}), "
          f"{reduction['bytes'][0] / 1e9:.1f} -> {reduction['bytes'][1] / 1e9:.1f} GB (-{1 - reduction['bytes'][1] / reduction['bytes'][0]:.1%})")

    if stops:
        print('\n[curate] mode=stops: same-place stops keep only the largest LiDAR diff, nothing else is deleted')
        for r in res['routes']:
            print(f"  {r['log'].split('_')[0]:5s} n={r['n']:2d}  정지 그룹 {[[short(m) for m in g] for g in r['stage1_groups']]}"
                  f"  삭제 {[short(m) for m in r['stage1']]}")
            for note in r['notes']:
                print(f'        {note}')
        print(f'  total deleted {len(decisions)} of {len(names)}; kept {len(kept)}')
        for m in sorted(decisions):
            print(f"  [{m}] {decisions[m]['summary']}")
    else:
        print(f"\n[curate] gamma={P['gamma']} min_hold={P['min_hold']}/{P['min_hold_event']} tol={P['tol']} labels={len(labels)}")
        for r in res['routes']:
            print(f"  {r['log'].split('_')[0]:5s} n={r['n']:2d} quota={r['quota']} (carry {r['carry_in']:.2f}->{r['carry_out']:.2f})"
                  f"  1단계={[short(m) for m in r['stage1']]}  2단계={[short(m) for m in r['stage2']]}  검토={[short(m) for m in r['review']]}")
            for note in r['notes']:
                print(f'        {note}')
            if r.get('conflict'):
                print(f"        충돌: {r['conflict']['text']}")
        print(f'  total deleted {len(decisions)} of {len(names)}; kept {len(kept)}; review {len(res["review"])}; conflicts {len(res["conflicts"])}\n')
        for m in sorted(decisions, key=lambda x: (decisions[x]['stage'], x)):
            d = decisions[m]
            print(f"  [{m}] {d['summary']}")
            for line in d['detail']:
                print(f'      {line}')
        for rv in res['review']:
            print(f"  [검토] {rv['text']}")
        print('\ncoverage: types that disappear:', cov['lost_types'] or 'none')
        for s in cov['stops']:
            print(f"  stop group {[short(m) for m in s['group']]} kept {[short(m) for m in s['kept']]} lost {s['lost'] or 'none'}")

    variants = {}
    if args.compare and not stops:
        for gm in (0.0, 1.0):
            v = run(db, feat, dict(P, gamma=gm), labels)
            variants[f'gamma {gm}'] = sorted(v['decisions'])
        ref = set(decisions)
        print('\nvariants (deleted scenes; Jaccard vs current run):')
        for k, v in variants.items():
            j = len(ref & set(v)) / len(ref | set(v)) if ref | set(v) else 1.0
            print(f"  {k:12s} J={j:.2f}  {[short(x) for x in v]}")

    review_of = {rv['scene']: rv for rv in res['review']}
    scenes_out, pairs = {}, []
    for n in names:
        b, d = feat[n]['beh'], decisions.get(n)
        t = traffic_type(feat, n, P)
        scenes_out[n] = dict(log=feat[n]['log'], status='remove' if d else 'keep', stage=d['stage'] if d else None,
                             kind=(d.get('kind') or ('중복' if d['stage'] == 1 else '수량 조정')) if d else None,
                             reason=d['detail'][0] if d else '', detail=d['detail'] if d else [], compare=d['compare'] if d else [],
                             summary=d.get('summary', '') if d else '',
                             review=review_of.get(n), obj=(feat[n]['obj'] or {}).get('mean'),
                             stop_act=stop_activity(feat, n), traffic_type=t, traffic_type_ko=type_ko(t),
                             substitute=d['evidence'] if d else None, sub_dist=d.get('sub_dist') if d else None,
                             group=group_of(feat, n), group_label=group_ko(group_of(feat, n)),
                             tags=[b['cls']] + [x for x in b['tags'] if x != b['cls']], cond=feat[n]['cond'],
                             v_mean_kmh=b['v_mean'], v_max_kmh=b['v_max'], stop_frac=b['stop_frac'], net_turn=b['net_turn'],
                             turn_deg=max(b['left_deg'], -b['right_deg']), max_acc=b['max_acc'], min_ax=b['min_ax'],
                             max_ay=b['max_ay'], bearing=b['bearing'], moving=b['moving'], lidar_cells=feat[n]['act'],
                             risk_score=feat[n]['risk']['score'], cost=None, forced=False, bytes=sizes[n]['bytes'],
                             pos=[float(feat[n]['place'][0].mean()), float(feat[n]['place'][1].mean())])
        if d and d['evidence']:
            pairs.append({'i': n, 'j': d['evidence'], 'substitutable': d['stage'] == 1, 'text': ' / '.join(d['detail'])})
    params = {k: (sorted(v) if isinstance(v, set) else v) for k, v in P.items() if k != 'dataroot'}
    result = dict(created=datetime.datetime.now().isoformat(timespec='seconds'), mode='stops' if stops else 'route_quota', params=params,
                  n_scenes=len(names), n_removed=len(decisions), reduction=reduction, routes=res['routes'], scenes=scenes_out,
                  review=res['review'], conflicts=res['conflicts'], coverage=cov, pairs=pairs, variants=variants,
                  groups={g: {'label': group_ko(g), 'seen': sum(1 for n in res['seen'] if group_of(feat, n) == g),
                              'retained': sum(1 for n in corpus if group_of(feat, n) == g)}
                          for g in sorted({group_of(feat, n) for n in names})})
    with open(os.path.join(out_dir, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    fin = sv.write_final(out_dir)                     # proposals + the viewer's human decisions -> final list
    print(f'\nwrote {out_dir}\\result.json and final_selection.json '
          f"(final: delete {fin['counts']['deleted']}, human decisions {fin['counts']['human_decisions']})")


if __name__ == '__main__':
    sys.exit(main())
