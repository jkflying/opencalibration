#pragma once

#include <opencalibration/types/measurement_graph.hpp>

#include <mutex>

namespace opencalibration
{

void relaxSubset(const std::vector<size_t> &node_ids, MeasurementGraph &graph, std::mutex &graph_mutex);
} // namespace opencalibration
