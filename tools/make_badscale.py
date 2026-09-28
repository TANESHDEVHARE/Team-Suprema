# Badly scaled but mathematically equivalent Netlib LPs: A' = R A C, x = C x'.
# usage: python tools/make_badscale.py <K> <name,name,...> [netlib_dir] [out_dir]
# Every row and column is multiplied by a random power of ten in 10^-K..10^K
# (seeded by the model name, so the files are reproducible). highspy is used
# only to read the original model; the file is written by write_mps below.
# c'_j = C_j c_j, [l', u']_j = [l, u]_j / C_j, row bounds scaled by R_i.
# The optimal objective value is unchanged, so the HiGHS reference still applies.
import highspy, numpy as np, sys, os
INF = 1e30
def write_mps(path, nm, lp, start, index, value):
    # Free MPS written here (not by HiGHS) so no small coefficient is dropped.
    m, n = lp.num_row_, lp.num_col_
    rl, ru, cl, cu, c = lp.row_lower_, lp.row_upper_, lp.col_lower_, lp.col_upper_, lp.col_cost_
    f = lambda v: repr(float(v))
    L = [f"NAME {nm}", "ROWS", " N OBJ"]
    kind = []
    for i in range(m):
        lo, hi = rl[i] > -INF, ru[i] < INF
        k = "E" if lo and hi and rl[i] == ru[i] else "G" if lo else "L" if hi else "N"
        if lo and hi and rl[i] != ru[i]: k = "G"
        kind.append(k); L.append(f" {k} R{i}")
    L.append("COLUMNS")
    for j in range(n):
        if c[j] != 0: L.append(f" C{j} OBJ {f(c[j])}")
        for p in range(start[j], start[j + 1]): L.append(f" C{j} R{index[p]} {f(value[p])}")
    L.append("RHS")
    if lp.offset_ != 0: L.append(f" RHS OBJ {f(-lp.offset_)}")
    for i in range(m):
        k = kind[i]; v = rl[i] if k in "EG" else ru[i] if k == "L" else 0.0
        if k != "N" and v != 0: L.append(f" RHS R{i} {f(v)}")
    L.append("RANGES")
    for i in range(m):
        if rl[i] > -INF and ru[i] < INF and rl[i] != ru[i]: L.append(f" RNG R{i} {f(ru[i] - rl[i])}")
    L.append("BOUNDS")
    for j in range(n):
        lo, hi = cl[j], cu[j]
        if lo <= -INF and hi >= INF: L.append(f" FR BND C{j}"); continue
        if lo == hi: L.append(f" FX BND C{j} {f(lo)}"); continue
        if lo <= -INF: L.append(f" MI BND C{j}")
        elif lo != 0: L.append(f" LO BND C{j} {f(lo)}")
        if hi < INF: L.append(f" UP BND C{j} {f(hi)}")
    L.append("ENDATA")
    open(path, "w").write("\n".join(L) + "\n")

K = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
names = sys.argv[2].split(",")
NETLIB = sys.argv[3] if len(sys.argv) > 3 else "netlib"
OUT = sys.argv[4] if len(sys.argv) > 4 else "badscale"
os.makedirs(OUT, exist_ok=True)
for nm in names:
    h = highspy.Highs(); h.setOptionValue("output_flag", False)
    h.readModel(os.path.join(NETLIB, nm + ".mps"))
    lp = h.getLp()
    rng = np.random.default_rng(abs(hash(nm)) % 2**32)
    m, n = lp.num_row_, lp.num_col_
    R = 10.0 ** rng.integers(-K, K + 1, m); C = 10.0 ** rng.integers(-K, K + 1, n)
    A = lp.a_matrix_
    start, index, value = np.array(A.start_), np.array(A.index_), np.array(A.value_, dtype=float)
    for j in range(n):
        s, e = start[j], start[j + 1]
        value[s:e] = value[s:e] * R[index[s:e]] * C[j]
    lp.a_matrix_.value_ = value.tolist()
    lp.col_cost_ = (np.array(lp.col_cost_) * C).tolist()
    lp.col_lower_ = (np.array(lp.col_lower_) / C).tolist()
    lp.col_upper_ = (np.array(lp.col_upper_) / C).tolist()
    lp.row_lower_ = (np.array(lp.row_lower_) * R).tolist()
    lp.row_upper_ = (np.array(lp.row_upper_) * R).tolist()
    out = os.path.join(OUT, "%s_s%d.mps" % (nm, int(K)))
    write_mps(out, nm, lp, start, index, value)
    a = np.abs(value[value != 0])
    print(f"{nm}: coefficient range {a.min():.1e} .. {a.max():.1e}  (ratio {a.max()/a.min():.1e})")
