#pragma once

#include <opencalibration/relax/graph.hpp>

#include <opencalibration/types/camera_relations.hpp>
#include <opencalibration/types/image.hpp>

namespace opencalibration
{

    using MeasurementGraph = DirectedGraph<image, camera_relations>;

    void relaxSingle(size_t node_id, MeasurementGraph& graph);
    void relaxSubset(const std::vector<size_t>& node_id, MeasurementGraph& graph);
}
