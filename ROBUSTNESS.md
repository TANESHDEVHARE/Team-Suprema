# Numerical robustness demonstration

```
sovereign robust --highs                       # all five classes, HiGHS as reference
sovereign robust --classes ill,degenerate      # a subset
python tools/robustness.py --help              # all options
```

Every model is solved three ways and judged against a reference optimum:

| Column | What it is |
|---|---|
| **ours** | the sovereign solver, default settings |
| **simple** | the same solver with its robustness machinery switched off — the "simpler implementation". LPs: textbook simplex (Dantzig pricing, plain ratio test, no bound flipping, no cost perturbation, no anti-cycling rule, no scaling). MILPs: branch-and-bound without cuts |
| **highs** | HiGHS (optional `--highs`, needs `highspy`); its answer is checked by our independent verifier |
| **highs (drop off)** | ill-conditioned class only: HiGHS with its default removal of coefficients below 1e-9 disabled (`small_matrix_value = 1e-12`), so it is not penalised for a reading default |

Verdicts: **correct** (status and objective match the reference: 1e-6 relative
for LP, the gap tolerance for MILP; infeasible models must be proven
infeasible), **wrong** (a claim that contradicts the reference), **failed**
(time limit, numerical failure, memory limit, error), **unreferenced**
(finished, but no reference optimum is known — never counted as correct).

## Model classes and sources

Files live in `bench_data/robustness/<class>/` (not in git: large, and
redistributable from their sources).

| Class | What makes it hard | Models | Source |
|---|---|---|---|
| `ill` | coefficient ratios up to 1e23; PDE discretisations | 20 Netlib LPs rescaled by `tools/make_badscale.py 4 ...` (every row and column × a random power of ten in 1e-4..1e4, seeded by name; optimum unchanged, reference = the original model's optimum); `cont1`, `cont4` | Netlib (coin-or Data-Netlib); Mittelmann LP test set, plato.asu.edu/ftp/lptestset/misc |
| `degenerate` | long runs of degenerate pivots, stalling, cycling | degen2, degen3, d6cube, cycle, greenbea, greenbeb, pilot, pilot87, perold, pilotnov, pilot4, stair, dfl001, d2q06c, maros-r7, fit2p; qap15, nug08-3rd (QAP lower-bound LPs), rail507 (set covering) | Netlib; Mittelmann LP test set (qap15, nug/, rail/) |
| `infeasible` | must prove there is no solution | the 29 Netlib infeasible LPs | Netlib infeas (coin-or Data-Infeas) |
| `weak` | LP bound far from the integer optimum (fixed charge, big-M, network design) | 28 MIPLIB 3 models (set1ch, fixnet6, pp08a, vpm2, qiu, mas74, mas76, modglob, …) and the 30×24 refinery planning MILP | MIPLIB 3 (coin-or Data-miplib3); `tools/gen_refinery.py --crudes 30 --periods 24` |
| `large` | size | ken-13, osa-30, pds-20 (Kennington), 1,000,000-variable transportation LP | Netlib Kennington set (EMPS-compressed; expand with Netlib's `emps`); `gen_lp 1000 1000` |

Reference optima are in `bench_data/robustness/references.json`: HiGHS 1.15.1
on the same file (a `--highs` run adds any that are missing), the original
model's optimum for the rescaled ones, and MIPLIB 3's published optima for
mas74, mas76, noswot and qiu (which HiGHS does not prove within its limit).
