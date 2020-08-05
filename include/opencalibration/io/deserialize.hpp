#pragma once

#include <opencalibration/types/measurement_graph.hpp>

namespace opencalibration
{
bool deserialize(const std::string &json, MeasurementGraph &graph);
}
