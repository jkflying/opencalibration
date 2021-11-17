#pragma once

#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/mesh_graph.hpp>

namespace opencalibration
{
bool deserialize(const std::string &json, MeasurementGraph &graph);
bool deserialize(std::istream &ply, MeshGraph &graph);
} // namespace opencalibration
