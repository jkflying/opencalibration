#pragma once

#include <opencalibration/types/graph.hpp>
#include <opencalibration/types/camera_relations.hpp>
#include <opencalibration/types/image.hpp>


namespace opencalibration
{

using MeasurementGraph = DirectedGraph<image, camera_relations>;

} // namespace opencalibration

