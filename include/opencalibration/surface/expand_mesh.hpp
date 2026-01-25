#pragma once

#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/point_cloud.hpp>
#include <opencalibration/types/surface_model.hpp>

namespace opencalibration
{

MeshGraph rebuildMesh(const point_cloud &cameraLocations, const std::vector<surface_model> &previousSurfaces);

/**
 * @brief Build a minimal 2-triangle mesh covering the camera bounding box
 *
 * Creates a simple square mesh with 4 vertices at the corners of the camera
 * bounding box (with margin), split into 2 triangles by a diagonal edge.
 *
 * @param cameraLocations Camera positions to determine bounds
 * @param previousSurfaces Previous surfaces for height estimation
 * @return MeshGraph with 4 nodes, 5 edges forming 2 triangles
 */
MeshGraph buildMinimalMesh(const point_cloud &cameraLocations, const std::vector<surface_model> &previousSurfaces);

} // namespace opencalibration
