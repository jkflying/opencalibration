#pragma once

#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/mesh_graph.hpp>

namespace opencalibration
{
bool deserialize(const std::string &json, MeasurementGraph &graph);
bool deserialize(std::istream &json, MeasurementGraph &graph);
bool deserialize(std::istream &ply, MeshGraph &graph);

class GraphRowReader
{
  public:
    bool addNode(MeasurementGraph &graph, size_t node_id, const std::string &json);
    bool addEdge(MeasurementGraph &graph, size_t edge_id, const std::string &json);

  private:
    ankerl::unordered_dense::map<size_t, std::shared_ptr<CameraModel>> _camera_models;
};
} // namespace opencalibration
