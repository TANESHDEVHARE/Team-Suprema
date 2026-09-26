// scaling.hpp -- port of scaling.py: Ruiz equilibration + Pock-Chambolle.
#pragma once
#include "ranged_lp.hpp"

struct ScalingResult {
    RangedLP scaled;
    std::vector<double> Dr, Dc;
};

ScalingResult scale(const RangedLP& ranged, int ruiz_passes = 10);
