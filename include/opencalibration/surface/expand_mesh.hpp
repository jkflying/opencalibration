#pragma once

#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/mesh_graph.hpp>
#include <opencalibration/types/point_cloud.hpp>

namespace opencalibration
{

MeshGraph rebuildMesh(const point_cloud &cameraLocations, const MeshGraph &meshGraph);

}
