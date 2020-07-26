#pragma once

#include <opencalibration/relax/graph.hpp>

#include <opencalibration/types/camera_relations.hpp>
#include <opencalibration/types/image.hpp>

#include <mutex>

namespace opencalibration
{

using MeasurementGraph = DirectedGraph<image, camera_relations>;

void relaxSubset(const std::vector<size_t> &node_ids, MeasurementGraph &graph, std::mutex &graph_mutex);
} // namespace opencalibration
