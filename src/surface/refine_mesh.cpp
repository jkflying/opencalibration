#include <opencalibration/surface/refine_mesh.hpp>

namespace opencalibration
{
void refineMesh(const MeasurementGraph &measurementGraph, MeshGraph &meshGraph)
{
    // find polygons on the mesh with a high number of measurements in them and split the polygon by adding new edges
    // and/or vertexes
}
} // namespace opencalibration
