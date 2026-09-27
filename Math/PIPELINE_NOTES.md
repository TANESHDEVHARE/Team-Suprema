# Corrections to the pipeline documents (as implemented)

These apply to `LP_Solver_Pipeline updated.pdf` (Rev 2), `MILP_Solver_Pipeline-1.pdf`
(Rev 3) and `QP_Solver_Pipeline.pdf` (Rev 6). The code follows these notes where
they differ from the PDFs.

1. **Presolve before scaling (LP Step 1/2, MILP Step 1/2).** Presolve runs on the
   original data; scaling is applied afterwards, only for the continuous solves.
   Presolving the scaled matrix breaks MILP: Ruiz scaling makes integer
   coefficients non-integer (GCD tightening can never fire) and column scaling
   makes integer variables non-integer (branching and integrality checks would
   happen in the wrong coordinates). For MILP, integer columns must not be
   column-scaled (or only by powers of two). Postsolve order is unchanged in
   spirit: unscale first, then undo presolve (last-in, first-out).

2. **Dual simplex needs a dual-feasible start (LP Steps 4-5).** A crossover basis
   from a 1e-4-accurate PDLP point is in general neither primal nor dual
   feasible. The implementation obtains dual feasibility by (a) bound flips for
   boxed variables, (b) dual phase 1 on the artificial-bounding auxiliary problem
   for a cold start, or (c) cost shifting for a warm start, and finishes with a
   primal simplex cleanup once shifts/perturbations are removed.

3. **Cost perturbation, not bound perturbation, in the dual simplex (LP Step 6.1).**
   Degeneracy in the dual simplex is dual degeneracy (zero reduced costs), so
   costs are perturbed (deterministically, by a hash of the column index).
   Bound perturbation is the primal-simplex counterpart. Bland's rule is kept as
   the rare deterministic fallback, triggered after 1000 pivots without
   objective progress; on the full Netlib set it never fired (longest stall
   observed: 225 pivots).

4. **No METIS (QP Step 5).** METIS is an external library, which the problem
   statement rules out. Fill-reducing ordering for the LDL^T factorization will
   be implemented here (approximate minimum degree first, nested dissection
   later).

5. **Harris ratio test is combined with bound flipping (LP Step 5.2).** The
   dual ratio test passes breakpoints of boxed variables while the leaving
   variable stays infeasible (bound-flipping ratio test); within the final
   group Harris' two-pass rule picks the largest |alpha|.

6. **Refactorization frequency (LP Step 5.3).** Every 100 Forrest-Tomlin updates
   (not 500-1000): at 500+ the row-eta file dominates FTRAN/BTRAN cost with the
   current dense-vector solves, and 100 is also the common choice in production
   simplex codes.

7. **QP: primal and dual take the same step (QP Step 5).** The dual residual
   Qx + c - A'y - z moves with both steps, so it only converges when they are
   equal; LP keeps separate primal/dual steps. Gondzio centrality correctors
   are used on top of Mehrotra. The ADMM warm start (QP Step 3) is not
   implemented: the IPM uses Mehrotra's starting point (least-squares solve +
   interior shift), which the document allows as the cold start.

8. **Crossover (LP Step 4)** is implemented as a pivoting crash (structurals
   enter the slack basis in basicness order, replacing only logicals, so the
   basis is nonsingular by construction), not as the full Megiddo primal/dual
   push. The simplex then finishes, with cost shifting for small dual
   infeasibilities and dual phase 1 for large ones.

9. **PDLP (LP Step 3)** uses the adaptive step size, primal weight and
   KKT-based restarts of Applegate et al., written once
   (`include/pdlp_algo.hpp`) over a CPU and a CUDA backend. GPU reductions
   are deterministic (per-block partials summed in fixed order).

## Implementation status by document step

| Document | Step | Status |
|---|---|---|
| LP | 1 parsing, condition screening, Ruiz + Pock-Chambolle | done (condition screening flag not separate) |
| LP | 2 presolve | partial: singleton rows, empty rows/cols, fixed columns; no doubletons, DBT or duplicate hashing |
| LP | 3 PDLP on GPU, native kernels | done |
| LP | 4 crossover | done (pivoting crash, see note 8) |
| LP | 5 dual simplex, Markowitz LU, DSE, Harris, Forrest-Tomlin | done |
| LP | 6 degeneracy (perturbation + Bland), refinement | done |
| LP | 7 postsolve, projection, unscale, KKT verify | done |
| MILP | 1-2 integrality tagging, integer-aware presolve | partial: bound rounding; no probing, cliques, GCD, symmetry |
| MILP | 3-6 root PDLP / crossover / warm simplex | root uses the dual simplex (PDLP root optional through --auto for LP) |
| MILP | 7 root cuts: GMI, MIR, cover | done (+ VUB substitution for flow structure); no lifted/clique cuts; no GPU cut scoring |
| MILP | 8 reliability branching | done |
| MILP | 9 node selection, parallel tree, pruning, reduced-cost fixing | done (best bound + plunging, work pool) |
| MILP | 10 heuristics: diving, feasibility pump, RINS/RENS | done |
| MILP | 11-12 gap, limits, verification | done |
| QP | 1 convexity filter, symmetric scaling | scaling done; Gershgorin filter not separate (inertia control catches non-convexity) |
| QP | 2 presolve | LP rules made Q-aware |
| QP | 3 GPU ADMM warm start | not done (note 7) |
| QP | 4-6 IPM, LDL (own ordering), inertia control, refinement | done |
| QP | 7 postsolve + KKT verification | done |
