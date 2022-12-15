#pragma once

#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/point_cloud.hpp>
#include <opencalibration/types/surface_model.hpp>

namespace opencalibration
{

MeshGraph rebuildMesh(const point_cloud &cameraLocations, const std::vector<surface_model> &previousSurfaces);

}
