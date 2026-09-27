#!/usr/bin/env python3
"""gen_refinery.py -- a representative multi-period refinery planning model,
written as MPS, for demonstrating the solver on the problem class the
problem statement targets (refinery scheduling / crude blending / production
planning). Everything is synthetic but structured like a real planning LP/MILP:

  crude purchase   cargo decisions (binary) with min/max lot size and a fixed
                   cost per cargo; crude tank inventory balances
  CDU              crude-specific yields into LPG / naphtha / kerosene /
                   gasoil / VGO / residue cuts; throughput capacity
  cracker (FCC)    converts VGO; on/off decision (binary) with minimum rate
  blending         gasoline (naphtha + cracked gasoline, octane >= spec,
                   sulfur <= spec), jet (kerosene, sulfur <= spec),
                   diesel (kerosene + gasoil + cycle oil, sulfur <= spec),
                   fuel oil (residue + slurry); quality constraints are
                   linear volume-weighted blends
  sales/inventory  product demand caps, product tank inventories
  objective        maximize margin = sales - crude cost - cargo fixed cost
                   - unit operating cost - inventory holding cost

  python tools/gen_refinery.py --crudes 12 --periods 12 --out refinery_12x12.mps
  python tools/gen_refinery.py --qp ...   # adds a quadratic penalty on CDU
                                          # throughput changes (smooth operation);
                                          # with --lp the binaries are relaxed
The model is a MILP by default, an LP with --lp, and a convex QP with --qp --lp.
"""
import argparse, random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crudes", type=int, default=12)
    ap.add_argument("--periods", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lp", action="store_true", help="relax all binaries (pure LP)")
    ap.add_argument("--qp", action="store_true", help="quadratic smoothing of CDU throughput")
    ap.add_argument("--out", default="refinery.mps")
    a = ap.parse_args()
    R = random.Random(a.seed)
    C, T = range(a.crudes), range(a.periods)
    cuts = ["lpg", "nap", "ker", "gso", "vgo", "res"]

    crude = []
    for c in C:
        light = R.uniform(0.0, 1.0)            # 0 = heavy, 1 = light
        y = {"lpg": 0.01 + 0.03 * light, "nap": 0.10 + 0.15 * light, "ker": 0.10 + 0.06 * light,
             "gso": 0.18 + 0.06 * light, "vgo": 0.25 - 0.05 * light}
        y["res"] = 1.0 - sum(y.values())
        sulfur = R.uniform(0.1, 3.0)            # wt% in the crude
        crude.append({"yield": y, "sulfur": sulfur,
                      "price": 70 + 12 * light - 4 * sulfur + R.uniform(-2, 2),
                      "lot_min": 150.0, "lot_max": 600.0, "fixed": R.uniform(800, 2500),
                      "avail": [R.random() < 0.8 for _ in T]})
    # sulfur of a cut ~ crude sulfur scaled by cut heaviness
    s_factor = {"lpg": 0.0, "nap": 0.05, "ker": 0.15, "gso": 0.6, "vgo": 1.2, "res": 2.5}
    price = {"lpg": 60, "gas": 105, "jet": 100, "dsl": 98, "fo": 55}
    demand = {p: [R.uniform(0.8, 1.2) * base for _ in T]
              for p, base in {"lpg": 60, "gas": 450, "jet": 180, "dsl": 520, "fo": 400}.items()}
    CDU_CAP, FCC_CAP, FCC_MIN = 1800.0, 450.0, 180.0

    rows = {}      # name -> (type, rhs)
    cols = {}      # name -> {row: coef}
    bounds = {}    # name -> (lo, hi)
    ints = set()
    obj = "MARGIN"

    def col(name, row, v):
        cols.setdefault(name, {})
        cols[name][row] = cols[name].get(row, 0.0) + v

    def row(name, t, rhs=0.0):
        rows[name] = (t, rhs)

    # objective: minimize negative margin
    for t in T:
        for c in C:
            b, y, x, inv = f"BUY_{c}_{t}", f"CARGO_{c}_{t}", f"FEED_{c}_{t}", f"CINV_{c}_{t}"
            col(b, obj, crude[c]["price"])
            col(y, obj, crude[c]["fixed"]); ints.add(y)
            bounds[y] = (0, 1 if crude[c]["avail"][t] else 0)
            col(inv, obj, 0.4)
            # lot sizes: lot_min y <= buy <= lot_max y
            row(f"LOTMIN_{c}_{t}", "G"); col(b, f"LOTMIN_{c}_{t}", 1.0); col(y, f"LOTMIN_{c}_{t}", -crude[c]["lot_min"])
            row(f"LOTMAX_{c}_{t}", "L"); col(b, f"LOTMAX_{c}_{t}", 1.0); col(y, f"LOTMAX_{c}_{t}", -crude[c]["lot_max"])
            # crude inventory: inv_t = inv_{t-1} + buy - feed   (initial stock 200)
            r = f"CBAL_{c}_{t}"
            row(r, "E", 200.0 if t == 0 else 0.0)
            col(inv, r, 1.0); col(b, r, -1.0); col(x, r, 1.0)
            if t > 0: col(f"CINV_{c}_{t-1}", r, -1.0)
            bounds[inv] = (0, 1500)
            row(f"CDU_{t}", "L", CDU_CAP); col(x, f"CDU_{t}", 1.0)
            col(x, obj, 2.0)                                          # CDU operating cost
            for k in cuts:                                            # cut production
                col(x, f"CUT_{k}_{t}", -crude[c]["yield"][k])
                col(x, f"CUTS_{k}_{t}", -crude[c]["yield"][k] * crude[c]["sulfur"] * s_factor[k])
        for k in cuts:
            row(f"CUT_{k}_{t}", "E"); col(f"SR_{k}_{t}", f"CUT_{k}_{t}", 1.0)
            # sulfur mass of each straight-run cut (tracked as its own variable)
            row(f"CUTS_{k}_{t}", "E"); col(f"SRS_{k}_{t}", f"CUTS_{k}_{t}", 1.0)
        # FCC: feed <= VGO, on/off with min rate
        f, u = f"FCCFEED_{t}", f"FCCON_{t}"
        ints.add(u); bounds[u] = (0, 1)
        col(u, obj, 3000.0); col(f, obj, 4.0)
        row(f"FCCMIN_{t}", "G"); col(f, f"FCCMIN_{t}", 1.0); col(u, f"FCCMIN_{t}", -FCC_MIN)
        row(f"FCCMAX_{t}", "L"); col(f, f"FCCMAX_{t}", 1.0); col(u, f"FCCMAX_{t}", -FCC_CAP)
        # stream splits: every cut is fully used (sent to a pool) or sold/burned
        row(f"VGO_{t}", "E"); col(f"SR_vgo_{t}", f"VGO_{t}", 1.0); col(f, f"VGO_{t}", -1.0); col(f"VGO2FO_{t}", f"VGO_{t}", -1.0)
        row(f"NAP_{t}", "E"); col(f"SR_nap_{t}", f"NAP_{t}", 1.0); col(f"NAP2GAS_{t}", f"NAP_{t}", -1.0); col(f"NAP2FO_{t}", f"NAP_{t}", -1.0)
        row(f"KER_{t}", "E"); col(f"SR_ker_{t}", f"KER_{t}", 1.0); col(f"KER2JET_{t}", f"KER_{t}", -1.0); col(f"KER2DSL_{t}", f"KER_{t}", -1.0)
        row(f"GSO_{t}", "E"); col(f"SR_gso_{t}", f"GSO_{t}", 1.0); col(f"GSO2DSL_{t}", f"GSO_{t}", -1.0); col(f"GSO2FO_{t}", f"GSO_{t}", -1.0)
        row(f"RES_{t}", "E"); col(f"SR_res_{t}", f"RES_{t}", 1.0); col(f"RES2FO_{t}", f"RES_{t}", -1.0)
        row(f"LPGP_{t}", "E"); col(f"SR_lpg_{t}", f"LPGP_{t}", 1.0); col(f"PROD_lpg_{t}", f"LPGP_{t}", -1.0)
        # gasoline pool
        cg, cd, sl = 0.55, 0.30, 0.10                                # FCC yields: gasoline, cycle oil, slurry
        row(f"GAS_{t}", "E"); col(f"NAP2GAS_{t}", f"GAS_{t}", 1.0); col(f, f"GAS_{t}", cg); col(f"PROD_gas_{t}", f"GAS_{t}", -1.0)
        row(f"GASOCT_{t}", "G")                                       # octane: 68 naphtha, 92 cracked, spec 87
        col(f"NAP2GAS_{t}", f"GASOCT_{t}", 68 - 87); col(f, f"GASOCT_{t}", cg * (92 - 87))
        row(f"JET_{t}", "E"); col(f"KER2JET_{t}", f"JET_{t}", 1.0); col(f"PROD_jet_{t}", f"JET_{t}", -1.0)
        row(f"DSL_{t}", "E"); col(f"KER2DSL_{t}", f"DSL_{t}", 1.0); col(f"GSO2DSL_{t}", f"DSL_{t}", 1.0)
        col(f, f"DSL_{t}", cd); col(f"PROD_dsl_{t}", f"DSL_{t}", -1.0)
        row(f"FO_{t}", "E"); col(f"RES2FO_{t}", f"FO_{t}", 1.0); col(f"VGO2FO_{t}", f"FO_{t}", 1.0)
        col(f"NAP2FO_{t}", f"FO_{t}", 1.0); col(f"GSO2FO_{t}", f"FO_{t}", 1.0); col(f, f"FO_{t}", sl)
        col(f"PROD_fo_{t}", f"FO_{t}", -1.0)
        # Product sulfur specs as linear blending indices (volume-weighted):
        # diesel: 0.9 kerosene + 1.4 gasoil + 0.8 cycle oil <= 1.0 per unit diesel
        row(f"DSLS_{t}", "L")
        col(f"KER2DSL_{t}", f"DSLS_{t}", 0.9); col(f"GSO2DSL_{t}", f"DSLS_{t}", 1.4); col(f, f"DSLS_{t}", cd * 0.8)
        col(f"PROD_dsl_{t}", f"DSLS_{t}", -1.0)
        row(f"JETS_{t}", "L")   # jet sulfur index 0.9 vs spec 0.95
        col(f"KER2JET_{t}", f"JETS_{t}", 0.9 - 0.95)
        # sulfur recovery unit: total sulfur mass entering with the crude slate is capped
        row(f"SULF_{t}", "L", 0.012 * CDU_CAP * 1.4)
        for k in cuts: col(f"SRS_{k}_{t}", f"SULF_{t}", 1.0)
        # sales, product inventories
        for p in price:
            s, pi = f"SELL_{p}_{t}", f"PINV_{p}_{t}"
            col(s, obj, -price[p]); bounds[s] = (0, demand[p][t])
            col(pi, obj, 0.8); bounds[pi] = (0, 300)
            r = f"PBAL_{p}_{t}"
            row(r, "E"); col(f"PROD_{p}_{t}", r, 1.0); col(s, r, -1.0); col(pi, r, -1.0)
            if t > 0: col(f"PINV_{p}_{t-1}", r, 1.0)

    if a.lp:
        ints.clear()

    quad = []
    if a.qp:
        # 1/2 * w * sum_t (cdu_t - cdu_{t-1})^2 with cdu_t = sum_c FEED_c_t, expanded
        # over the feed variables: a PSD quadratic (Gram matrix of differences).
        w = 0.02
        diff = []
        for t in range(1, a.periods):
            v = {}
            for c in C:
                v[f"FEED_{c}_{t}"] = v.get(f"FEED_{c}_{t}", 0) + 1.0
                v[f"FEED_{c}_{t-1}"] = v.get(f"FEED_{c}_{t-1}", 0) - 1.0
            diff.append(v)
        Q = {}
        for v in diff:
            items = list(v.items())
            for i, (ci, ai) in enumerate(items):
                for cj, aj in items[:i + 1]:
                    key = (ci, cj)
                    Q[key] = Q.get(key, 0.0) + w * ai * aj
        order = {}
        for name in cols: order.setdefault(name, len(order))
        for (ci, cj), v in Q.items():
            if order[ci] < order[cj]: ci, cj = cj, ci
            if abs(v) > 0: quad.append((ci, cj, v))

    with open(a.out, "w") as f:
        f.write(f"NAME REFINERY_{a.crudes}x{a.periods}\nROWS\n N  {obj}\n")
        for r, (t, _) in rows.items(): f.write(f" {t}  {r}\n")
        f.write("COLUMNS\n")
        in_int = False
        for name, ent in cols.items():
            is_int = name in ints
            if is_int and not in_int: f.write("    MARKER  'MARKER'  'INTORG'\n"); in_int = True
            if not is_int and in_int: f.write("    MARKER  'MARKER'  'INTEND'\n"); in_int = False
            for r, v in ent.items():
                if v != 0.0: f.write(f"    {name}  {r}  {v:.10g}\n")
        if in_int: f.write("    MARKER  'MARKER'  'INTEND'\n")
        f.write("RHS\n")
        for r, (t, rhs) in rows.items():
            if rhs != 0.0: f.write(f"    RHS  {r}  {rhs:.10g}\n")
        f.write("BOUNDS\n")
        for name, (lo, hi) in bounds.items():
            if a.lp and name.startswith(("CARGO_", "FCCON_")):
                f.write(f" UP BND  {name}  {hi}\n")
                continue
            if lo != 0: f.write(f" LO BND  {name}  {lo}\n")
            f.write(f" UP BND  {name}  {hi}\n")
        if quad:
            f.write("QUADOBJ\n")
            for ci, cj, v in quad: f.write(f"    {ci}  {cj}  {v:.10g}\n")
        f.write("ENDATA\n")
    nint = len(ints)
    print(f"wrote {a.out}: {len(rows)} rows, {len(cols)} columns, {nint} integer, "
          f"{sum(len(e) for e in cols.values())} nonzeros" + (f", {len(quad)} Q entries" if quad else ""))


if __name__ == "__main__":
    main()
