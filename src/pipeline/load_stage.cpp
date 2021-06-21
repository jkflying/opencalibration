#include <opencalibration/pipeline/load_stage.hpp>

#include <opencalibration/extract/extract_features.hpp>
#include <opencalibration/extract/extract_metadata.hpp>
#include <opencalibration/performance/performance.hpp>

#include <spdlog/spdlog.h>

namespace
{
std::array<double, 3> to_array(const Eigen::Vector3d &v)
{
    return {v.x(), v.y(), v.z()};
}
} // namespace

namespace opencalibration
{

void LoadStage::init(const std::vector<std::string> &paths_to_load)
{
    PerformanceMeasure p("Load init");
    spdlog::info("Queueing {} image paths for loading", paths_to_load.size());
    _paths_to_load = paths_to_load;
    _images.clear();
    _images.reserve(_paths_to_load.size());
}

std::vector<std::function<void()>> LoadStage::get_runners()
{
    PerformanceMeasure p("Load get_runners");
    std::vector<std::function<void()>> funcs;
    funcs.reserve(_paths_to_load.size());
    for (size_t i = 0; i < _paths_to_load.size(); i++)
    {
        auto run_func = [&, i]() {
            PerformanceMeasure p2("Load runner");
            const std::string &path = _paths_to_load[i];
            image img;
            img.path = path;
            img.features = extract_features(img.path);
            if (img.features.size() == 0)
            {
                return;
            }

            img.metadata = extract_metadata(img.path);
            img.model.focal_length_pixels = img.metadata.focal_length_px;
            img.model.pixels_cols = img.metadata.width_px;
            img.model.pixels_rows = img.metadata.height_px;
            img.model.principle_point = Eigen::Vector2d(img.model.pixels_cols, img.model.pixels_rows) / 2;

            spdlog::debug("camera model: dims: {}x{} focal: {}", img.model.pixels_cols, img.model.pixels_rows,
                          img.model.focal_length_pixels);

            std::lock_guard<std::mutex> lock(_images_mutex);
            _images.emplace_back(i, std::move(img));
        };
        funcs.push_back(run_func);
    }

    return funcs;
}

std::vector<size_t> LoadStage::finalize(GeoCoord &coordinate_system, MeasurementGraph &graph,
                                        jk::tree::KDTree<size_t, 3> &imageGPSLocations)
{
    PerformanceMeasure p("Load finalize");
    // put the images back in order after the parallel processing
    std::sort(_images.begin(), _images.end(),
              [](const std::pair<size_t, image> &img1, const std::pair<size_t, image> &img2) -> int {
                  return img1.first < img2.first;
              });

    std::vector<size_t> node_ids;
    node_ids.reserve(_paths_to_load.size());

    for (auto &p : _images)
    {
        auto &img = p.second;
        if (!coordinate_system.isInitialized())
        {
            coordinate_system.setOrigin(img.metadata.latitude, img.metadata.longitude);
        }

        Eigen::Vector3d local_pos = img.position =
            coordinate_system.toLocalCS(img.metadata.latitude, img.metadata.longitude, img.metadata.altitude);
        size_t node_id = graph.addNode(std::move(img));
        imageGPSLocations.addPoint(to_array(local_pos), node_id);
        node_ids.push_back(node_id);
    }

    return node_ids;
}

} // namespace opencalibration
