#include <opencalibration/dense/dense_stereo.hpp>

#include <spdlog/spdlog.h>

namespace opencalibration
{

void densifyMesh(const MeasurementGraph & /*graph*/, std::vector<surface_model> & /*surfaces*/,
                 std::function<void(float)> progress_cb)
{
    spdlog::info("Dense mesh: TODO");

    if (progress_cb)
        progress_cb(1.f);
}

} // namespace opencalibration
