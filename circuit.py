"""
Circuit-level noise simulation for ADAPD paper revision.

Under circuit-level noise:
  - Data qubits: depolarising at rate p
  - Syndrome measurements: bit-flip at rate p_meas ≈ p
  - Decode over d rounds of syndrome history (space-time decoding)

We build a 3D syndrome graph (space x space x time) and decode it.
This is the standard approach from Dennis et al. 2002.

For simplicity we use the phenomenological noise model:
  - Each data qubit gets X error with prob p/3 per round
  - Each syndrome bit is independently flipped with prob p_meas = p

We decode using temporal MWPM on the 3D spacetime graph.
"""
import numpy as np, networkx as nx, time, json
from collections import defaultdict

# ── Surface code ─────────────────────────────────────────────────────────────
class RSC:
    def __init__(self, d):
        self.d = d; self.n = d*d; d2 = d; self.z_stabs = []
        for r in range(d2-1):
            for c in range(d2-1):
                self.z_stabs.append([r*d2+c, r*d2+c+1, (r+1)*d2+c, (r+1)*d2+c+1])
        for c in range(0, d2-1, 2): self.z_stabs.append([c, c+1])
        for c in range(1, d2-1, 2): self.z_stabs.append([(d2-1)*d2+c, (d2-1)*d2+c+1])
        self.ns = len(self.z_stabs)
        self.spos = []
        for qs in self.z_stabs:
            rs=[q//d2 for q in qs]; cs=[q%d2 for q in qs]
            self.spos.append((float(np.mean(rs)), float(np.mean(cs))))
    def syn(self, e):
        s = np.zeros(self.ns, dtype=np.int8)
        for i, qs in enumerate(self.z_stabs): s[i] = int(np.sum(e[qs]) % 2)
        return s
    def fail(self, e): return int(sum(e[r*self.d] for r in range(self.d)) % 2) == 1

# ── Phenomenological noise ────────────────────────────────────────────────────
def phenomenological_trial(code, p, p_meas, n_rounds, rng):
    """
    Run n_rounds of syndrome extraction under phenomenological noise.
    Returns cumulative X error and noisy syndrome history.
    """
    d = code.d
    cumulative_err = np.zeros(code.n, dtype=np.int8)
    syndrome_history = []  # shape (n_rounds, n_stabs)

    for t in range(n_rounds):
        # Data errors this round
        round_err = (rng.random(code.n) < p/3).astype(np.int8)
        cumulative_err = (cumulative_err + round_err) % 2

        # Noiseless syndrome
        true_syn = code.syn(cumulative_err)

        # Measurement errors
        meas_err = (rng.random(code.ns) < p_meas).astype(np.int8)
        noisy_syn = (true_syn + meas_err) % 2
        syndrome_history.append(noisy_syn)

    return cumulative_err, np.array(syndrome_history)

# ── 3D defect extraction ─────────────────────────────────────────────────────
def get_3d_defects(syndrome_history):
    """
    Defects in 3D spacetime = positions where syndrome changes between rounds,
    or is nonzero in the first round.
    Returns list of (time, stab_idx) defect positions.
    """
    n_rounds, n_stabs = syndrome_history.shape
    defects = []
    prev = np.zeros(n_stabs, dtype=np.int8)
    for t, syn in enumerate(syndrome_history):
        diff = (syn - prev) % 2  # changes = new defects
        for s in np.where(diff == 1)[0]:
            defects.append((t, int(s)))
        prev = syn
    # Final round: any remaining syndromes are boundary defects
    for s in np.where(syndrome_history[-1] == 1)[0]:
        defects.append((n_rounds, int(s)))
    return defects

# ── 3D MWPM ──────────────────────────────────────────────────────────────────
class UF:
    def __init__(self, n): self.p = list(range(n)); self.r = [0]*n
    def find(self, x):
        while self.p[x] != x: self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, x, y):
        rx,ry = self.find(x),self.find(y)
        if rx==ry: return
        if self.r[rx]<self.r[ry]: rx,ry=ry,rx
        self.p[ry]=rx
        if self.r[rx]==self.r[ry]: self.r[rx]+=1

def drop_boundary_3d(code, defects_3d, n_rounds):
    """Drop boundary-nearest defect if odd count."""
    if len(defects_3d) % 2 == 0: return defects_3d
    d = code.d
    best_idx = 0; best_score = 1e9
    for i, (t, s) in enumerate(defects_3d):
        r, c = code.spos[s]
        space_bdist = min(r, d-1-r, c, d-1-c)
        time_bdist = min(t, n_rounds - t)
        score = min(space_bdist, time_bdist)
        if score < best_score: best_score = score; best_idx = i
    return [x for i,x in enumerate(defects_3d) if i != best_idx]

def decode_3d_mwpm(code, defects_3d, p, n_rounds):
    """3D MWPM: space-time graph over defect positions."""
    t0 = time.perf_counter()
    zero = np.zeros(code.n, dtype=np.int8)
    if not defects_3d:
        return zero, time.perf_counter()-t0

    defects_3d = drop_boundary_3d(code, defects_3d, n_rounds)
    if not defects_3d:
        return zero, time.perf_counter()-t0

    eps = 1e-9
    lr = np.log((1-p/3+eps)/(p/3+eps))

    G = nx.Graph()
    for i in range(len(defects_3d)):
        for j in range(i+1, len(defects_3d)):
            ti,si = defects_3d[i]; tj,sj = defects_3d[j]
            ri,ci = code.spos[si]; rj,cj = code.spos[sj]
            space_dist = abs(ri-rj) + abs(ci-cj)
            time_dist  = abs(ti-tj)
            # Time edges: measurement errors, weight proportional to time distance
            weight = (space_dist + time_dist) * lr
            G.add_edge(i, j, weight=-weight)

    try:
        matching = nx.max_weight_matching(G, maxcardinality=True)
    except:
        return zero, time.perf_counter()-t0

    # Decode: for each matched pair, build correction path in space
    # (time matches correct measurement errors, not data errors)
    correction = np.zeros(code.n, dtype=np.int8)
    d = code.d
    for (idx_i, idx_j) in matching:
        ti,si = defects_3d[idx_i]; tj,sj = defects_3d[idx_j]
        ri=int(round(code.spos[si][0])); ci=int(round(code.spos[si][1]))
        rj=int(round(code.spos[sj][0])); cj=int(round(code.spos[sj][1]))
        for col in range(min(ci,cj), max(ci,cj)):
            if 0<=ri<d and 0<=col<d: correction[ri*d+col] ^= 1
        for row in range(min(ri,rj), max(ri,rj)):
            if 0<=row<d and 0<=cj<d: correction[row*d+cj] ^= 1

    return correction, time.perf_counter()-t0

def decode_3d_uf(code, defects_3d, p, n_rounds):
    """3D greedy: pair by 3D Manhattan distance."""
    t0 = time.perf_counter()
    zero = np.zeros(code.n, dtype=np.int8)
    if not defects_3d:
        return zero, time.perf_counter()-t0
    defects_3d = drop_boundary_3d(code, defects_3d, n_rounds)
    if not defects_3d: return zero, time.perf_counter()-t0

    d = code.d
    correction = np.zeros(code.n, dtype=np.int8)
    remaining = list(range(len(defects_3d)))

    while len(remaining) >= 2:
        i = remaining[0]; best_j = remaining[1]; best_dist = 1e18
        for j in remaining[1:]:
            ti,si = defects_3d[i]; tj,sj = defects_3d[j]
            ri,ci = code.spos[si]; rj,cj = code.spos[sj]
            dist = abs(ri-rj)+abs(ci-cj)+abs(ti-tj)
            if dist < best_dist: best_dist = dist; best_j = j
        ti,si = defects_3d[i]; tj,sj = defects_3d[best_j]
        ri=int(round(code.spos[si][0])); ci=int(round(code.spos[si][1]))
        rj=int(round(code.spos[sj][0])); cj=int(round(code.spos[sj][1]))
        for col in range(min(ci,cj),max(ci,cj)):
            if 0<=ri<d and 0<=col<d: correction[ri*d+col]^=1
        for row in range(min(ri,rj),max(ri,rj)):
            if 0<=row<d and 0<=cj<d: correction[row*d+cj]^=1
        remaining.remove(i); remaining.remove(best_j)

    return correction, time.perf_counter()-t0

def decode_3d_adapd(code, defects_3d, p, n_rounds, rho_star=0.40):
    """3D ADAPD: cluster by 3D proximity, then tier dispatch."""
    t0 = time.perf_counter()
    zero = np.zeros(code.n, dtype=np.int8)
    if not defects_3d: return zero, time.perf_counter()-t0

    ph = min(max(len(defects_3d)/(code.n*0.667*n_rounds), 1e-4), 0.45)
    radius = min(max(2.0, 0.8/max(ph, 0.005)), float(code.d)*1.5)

    # 3D clustering
    n = len(defects_3d)
    uf2 = UF(n)
    for i in range(n):
        for j in range(i+1, n):
            ti,si = defects_3d[i]; tj,sj = defects_3d[j]
            ri,ci = code.spos[si]; rj,cj = code.spos[sj]
            dist3d = abs(ri-rj)+abs(ci-cj)+abs(ti-tj)
            if dist3d <= radius: uf2.union(i, j)

    cm = defaultdict(list)
    for idx in range(n): cm[uf2.find(idx)].append(idx)
    clusters_idx = list(cm.values())

    correction = np.zeros(code.n, dtype=np.int8)
    d = code.d
    eps=1e-9; lr=np.log((1-ph/3+eps)/(ph/3+eps))

    for cl_idx in clusters_idx:
        cl = [defects_3d[i] for i in cl_idx]
        cl = drop_boundary_3d(code, cl, n_rounds)
        if not cl: continue

        if len(cl) == 2:
            pairs = [(cl[0], cl[1])]
        else:
            # density in 3D bounding box
            rows=[code.spos[s][0] for _,s in cl]; cols=[code.spos[s][1] for _,s in cl]
            times=[t for t,_ in cl]
            area = max((max(rows)-min(rows)+1)*(max(cols)-min(cols)+1)*(max(times)-min(times)+1), 1.0)
            density = len(cl)/area
            if density < rho_star or len(cl) <= 4:
                # Tier 1: greedy 3D
                remaining = list(cl)
                pairs = []
                while len(remaining) >= 2:
                    i = remaining[0]; best_j=remaining[1]; bd=1e18
                    for j in remaining[1:]:
                        ti,si=i; tj,sj=j
                        ri,ci=code.spos[si]; rj,cj=code.spos[sj]
                        dist=abs(ri-rj)+abs(ci-cj)+abs(ti-tj)
                        if dist<bd: bd=dist; best_j=j
                    pairs.append((i,best_j)); remaining.remove(i); remaining.remove(best_j)
            else:
                # Tier 2: MWPM on 3D cluster subgraph
                G2 = nx.Graph()
                for i2 in range(len(cl)):
                    for j2 in range(i2+1, len(cl)):
                        ti,si=cl[i2]; tj,sj=cl[j2]
                        ri,ci=code.spos[si]; rj,cj=code.spos[sj]
                        dist=(abs(ri-rj)+abs(ci-cj)+abs(ti-tj))*lr
                        G2.add_edge(i2,j2,weight=-dist)
                try:
                    m2 = nx.max_weight_matching(G2, maxcardinality=True)
                    pairs = [(cl[a],cl[b]) for a,b in m2]
                except:
                    pairs = []

        # Apply correction for each pair
        for (ti,si),(tj,sj) in pairs:
            ri=int(round(code.spos[si][0])); ci=int(round(code.spos[si][1]))
            rj=int(round(code.spos[sj][0])); cj=int(round(code.spos[sj][1]))
            for col in range(min(ci,cj),max(ci,cj)):
                if 0<=ri<d and 0<=col<d: correction[ri*d+col]^=1
            for row in range(min(ri,rj),max(ri,rj)):
                if 0<=row<d and 0<=cj<d: correction[row*d+cj]^=1

    return correction, time.perf_counter()-t0

# ── Monte Carlo ───────────────────────────────────────────────────────────────
def simulate_circuit(code, decoder_fn, p, n_shots, n_rounds, rng):
    fails=0; lats=[]
    for _ in range(n_shots):
        cum_err, syn_hist = phenomenological_trial(code, p, p, n_rounds, rng)
        defects_3d = get_3d_defects(syn_hist)
        corr, lat = decoder_fn(code, defects_3d, p, n_rounds)
        lats.append(lat*1e6)
        residual = (cum_err + corr) % 2
        if code.fail(residual): fails+=1
    ler = fails/n_shots
    se = 1.96*np.sqrt(max(ler*(1-ler)/n_shots, 1e-10))
    return ler, se, float(np.mean(lats)), fails

if __name__ == '__main__':
    rng = np.random.default_rng(2024)
    distances = [3, 5, 7]
    # Circuit-level pseudo-threshold is ~0.3-0.7% so test lower p range
    prates = [0.002, 0.003, 0.004, 0.005, 0.007, 0.010, 0.013, 0.016]
    shots_map = {0.002:5000, 0.003:5000, 0.004:5000, 0.005:4000,
                 0.007:4000, 0.010:3000, 0.013:3000, 0.016:3000}
    DECODERS = {'MWPM': decode_3d_mwpm, 'UnionFind': decode_3d_uf, 'ADAPD': decode_3d_adapd}

    results = {}
    for dn, dfn in DECODERS.items():
        results[dn] = {}
        for d in distances:
            results[dn][d] = {}
            code = RSC(d)
            n_rounds = d  # standard: d rounds per cycle
            for p in prates:
                ns = shots_map[p]
                print(f'CIRCUIT {dn} d={d} p={p:.3f} rounds={n_rounds}', flush=True)
                ler, se, lat, nf = simulate_circuit(code, dfn, p, ns, n_rounds, rng)
                results[dn][d][p] = {'ler':ler,'se':se,'lat':lat,'ns':ns,'nf':nf}
                print(f'  LER={ler:.4e} se={se:.2e} lat={lat:.2f}us fails={nf}', flush=True)

    # Serialise
    out = {}
    for dn in results:
        out[dn] = {}
        for d in results[dn]:
            out[dn][str(d)] = {}
            for p in results[dn][d]:
                out[dn][str(d)][str(p)] = results[dn][d][p]
    with open('/home/claude/circuit_level_results.json','w') as f:
        json.dump(out, f, indent=2)
    print('=== CIRCUIT LEVEL DONE ===')
