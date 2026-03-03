#pragma once

#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/surface_model.hpp>

#include <functional>
#include <vector>

namespace opencalibration
{

void densifyMesh(const MeasurementGraph &graph, std::vector<surface_model> &surfaces,
                 std::function<void(float)> progress_cb = {});

} // namespace opencalibration
