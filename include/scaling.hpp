// scaling.hpp -- port of scaling.py: Ruiz equilibration + Pock-Chambolle.
#pragma once
#include "ranged_lp.hpp"

struct ScalingResult {
    RangedLP scaled;
    std::vector<double> Dr, Dc;
};

// Columns with ranged.integer[j] set are never column-scaled (Dc = 1): a
// scaled integer variable x/Dc would not be integer-valued, breaking
// branching and integrality checks (Math/PIPELINE_NOTES.md, item 1).
ScalingResult scale(const RangedLP& ranged, int ruiz_passes = 10);
