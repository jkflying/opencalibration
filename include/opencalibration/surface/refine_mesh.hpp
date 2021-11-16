#pragma once

#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/mesh_graph.hpp>

namespace opencalibration
{

void refineMesh(const MeasurementGraph &measurementGraph, MeshGraph &meshGraph);

}
