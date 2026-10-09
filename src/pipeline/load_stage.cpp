#include "load_stage.hpp"

#include <opencalibration/extract/extract_image.hpp>
#include <opencalibration/performance/performance.hpp>

#include <spdlog/spdlog.h>

#include <cmath>

namespace opencalibration
{

namespace
{

const char *unusableReason(const image &img)
{
    const auto &capture = img.metadata.capture_info;
    if (!std::isfinite(capture.latitude) || !std::isfinite(capture.longitude) || !std::isfinite(capture.altitude))
        return "no GPS position";
    if (!(img.model->focal_length_pixels > 0))
        return "unknown focal length, add the camera to the camera database";
    return nullptr;
}

} // namespace

void LoadStage::init(const MeasurementGraph &graph, const std::vector<std::string> &paths_to_load)
{
    PerformanceMeasure p("Load init");
    spdlog::info("Queueing {} image paths for loading", paths_to_load.size());
    _paths_to_load = paths_to_load;
    _images.clear();
    _images.reserve(_paths_to_load.size());

    // initialize camera models map if the graph was deserialized from elsewhere
    if (_camera_models.size() == 0 && graph.size_nodes() > 0)
    {
        for (auto niter = graph.cnodebegin(); niter != graph.cnodeend(); ++niter)
        {
            const auto &payload = niter->second.payload;

            auto miter = _camera_models.find(payload.model->id);
            if (miter == _camera_models.end())
            {
                _camera_models.emplace(payload.model->id, std::make_pair(payload.metadata.camera_info, payload.model));
            }
        }
    }
}

std::vector<std::function<void()>> LoadStage::get_runners()
{
    std::vector<std::function<void()>> funcs;
    funcs.reserve(_paths_to_load.size());
    for (size_t i = 0; i < _paths_to_load.size(); i++)
    {
        auto run_func = [&, i]() {
            PerformanceMeasure p("Load store read");
            std::optional<image> img = store ? store->loadImage(_paths_to_load[i]) : std::nullopt;
            p.reset("");
            if (img == std::nullopt)
            {
                img = extract_image(_paths_to_load[i]);
                p.reset("Load store write");
                if (img != std::nullopt && store && !store->saveImage(*img))
                    spdlog::warn("Failed to cache features of {}", _paths_to_load[i]);
                p.reset("");
            }
            if (img != std::nullopt)
            {
                if (on_loaded)
                    on_loaded(img->path, img->metadata.capture_info.latitude, img->metadata.capture_info.longitude);
                std::lock_guard<std::mutex> lock(_images_mutex);
                _images.emplace_back(i, std::move(*img));
            }
        };
        funcs.push_back(run_func);
    }

    return funcs;
}

std::vector<size_t> LoadStage::finalize(GeoCoord &coordinate_system, MeasurementGraph &graph,
                                        jk::tree::KDTree<size_t, 2> &imageGPSLocations)
{
    PerformanceMeasure p("Load finalize");
    // put the images back in order after the parallel processing
    std::sort(_images.begin(), _images.end(),
              [](const std::pair<size_t, image> &img1, const std::pair<size_t, image> &img2) -> int {
                  return img1.first < img2.first;
              });

    std::vector<size_t> node_ids;
    node_ids.reserve(_paths_to_load.size());
    size_t stationary = 0;

    for (auto &p : _images)
    {
        auto &img = p.second;
        if (const char *reason = unusableReason(img))
        {
            spdlog::warn("Skipping {}: {}", img.path, reason);
            continue;
        }
        if (!coordinate_system.isInitialized())
        {
            coordinate_system.setOrigin(img.metadata.capture_info.latitude, img.metadata.capture_info.longitude);
        }

        // figure out if we've already loaded a camera with the same lens
        bool already_added = false;
        for (const auto &id_model : _camera_models)
        {
            if (id_model.second.first == img.metadata.camera_info)
            {
                img.model = id_model.second.second;
                already_added = true;
                break;
            }
        }
        if (!already_added)
        {
            do
            {
                img.model->id = distribution(generator);
            } while (_camera_models.find(img.model->id) != _camera_models.end());
            _camera_models.emplace(img.model->id, std::make_pair(img.metadata.camera_info, img.model));

            spdlog::debug("camera model: dims: {}x{} focal: {}", img.model->pixels_cols, img.model->pixels_rows,
                          img.model->focal_length_pixels);
        }

        Eigen::Vector3d local_pos = img.position = img.gps_position =
            coordinate_system.toLocalCS(img.metadata.capture_info.latitude, img.metadata.capture_info.longitude,
                                        img.metadata.capture_info.altitude);

        const bool adds_no_baseline =
            _last_kept_xy && (local_pos.head<2>() - *_last_kept_xy).norm() < MIN_HORIZONTAL_MOVE_METERS;
        if (adds_no_baseline)
        {
            stationary++;
            continue;
        }
        _last_kept_xy = local_pos.head<2>();

        size_t node_id = graph.addNode(std::move(img));
        imageGPSLocations.addPoint({local_pos.x(), local_pos.y()}, node_id);
        node_ids.push_back(node_id);
    }

    if (stationary > 0)
        spdlog::info("Skipped {} images within {}m of the previous image", stationary, MIN_HORIZONTAL_MOVE_METERS);
    return node_ids;
}

} // namespace opencalibration
