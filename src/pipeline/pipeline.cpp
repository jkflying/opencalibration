#include <opencalibration/pipeline/pipeline.hpp>

#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/extract/extract_features.hpp>
#include <opencalibration/extract/extract_metadata.hpp>
#include <opencalibration/match/match_features.hpp>
#include <opencalibration/model_inliers/ransac.hpp>

#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace opencalibration
{
Pipeline::Pipeline(size_t batch_size)
{
    _keep_running = true;
    auto get_paths = [this, batch_size](std::vector<std::string> &paths, ThreadStatus &status) -> bool {
        while (_add_queue.size() > 0 && paths.size() < batch_size)
        {
            paths.emplace_back(std::move(_add_queue.front()));
            _add_queue.pop_front();
        }
        if (paths.size() > 0)
        {
            status = ThreadStatus::BUSY;
            return true;
        }
        else
        {
            status = ThreadStatus::IDLE;
            return false;
        }
    };

    _runner.thread.reset(new std::thread([this, batch_size, get_paths]() {
        std::mutex sleep_mutex;
        std::vector<std::string> paths;
        paths.reserve(batch_size);
        while (_keep_running)
        {
            paths.clear();
            if (get_paths(paths, _runner.status))
            {
                process_images(paths);
            }
            else
            {
                std::unique_lock<std::mutex> lck(sleep_mutex);
                _queue_condition_variable.wait_for(lck, 1s);
            }
        }
    }));
}

Pipeline::~Pipeline()
{
    _keep_running = false;
    _queue_condition_variable.notify_all();
    if (_runner.thread != nullptr)
    {
        _runner.thread->join();
    }
}

void Pipeline::process_images(const std::vector<std::string> &paths)
{
    auto to_array = [](Eigen::Vector3d v) -> std::array<double, 3> { return {v.x(), v.y(), v.z()}; };

    std::vector<size_t> node_ids;
    node_ids.reserve(paths.size());

#pragma omp parallel for
    for (int i = 0; i < (int)paths.size(); i++)
    {
        const std::string& path = paths[i];
        image img;
        img.path = path;
        img.features = extract_features(img.path);
        if (img.features.size() == 0)
        {
            continue;
        }

        img.metadata = extract_metadata(img.path);
        img.model.focal_length_pixels = img.metadata.focal_length_px;
        img.model.pixels_cols = img.metadata.width_px;
        img.model.pixels_rows = img.metadata.height_px;
        img.model.principle_point = Eigen::Vector2d(img.model.pixels_cols, img.model.pixels_rows) / 2;

        std::lock_guard<std::mutex> graph_lock(_graph_structure_mutex);
        if (!_coordinate_system.isInitialized())
        {
            _coordinate_system.setOrigin(img.metadata.latitude, img.metadata.longitude);
        }
        Eigen::Vector3d local_pos = img.position =
            _coordinate_system.toLocalCS(img.metadata.latitude, img.metadata.longitude, img.metadata.altitude);
        size_t node_id = _graph.addNode(std::move(img));

        _imageGPSLocations.addPoint(to_array(local_pos), node_id);
        node_ids.push_back(node_id);
    }

    // TODO: sort node_ids by the path, since they may be out of order now

    std::sort(node_ids.begin(), node_ids.end(), [this](size_t node_id_1, size_t node_id_2) -> int {
        const auto &path1 = _graph.getNode(node_id_1)->payload.path;
        const auto &path2 = _graph.getNode(node_id_2)->payload.path;
        return path1 < path2;
    });

    // build nearest in serial, since it keeps graph structure deterministic
    std::vector<std::pair<size_t, std::vector<size_t>>> batch_nearest;
    batch_nearest.reserve(node_ids.size());
    for (size_t node_id : node_ids)
    {
        const auto &img = _graph.getNode(node_id)->payload;
        std::vector<size_t> nearest;
        auto knn = _imageGPSLocations.searchKnn(to_array(img.position), 10);
        nearest.reserve(knn.size());
        for (const auto &nn : knn)
        {
            nearest.push_back(nn.payload);
        }
        batch_nearest.emplace_back(node_id, std::move(nearest));
    }

#pragma omp parallel for
    for (int i = 0; i < (int)batch_nearest.size(); i++)
    {
        const auto &node_nearest = batch_nearest[i];
        size_t node_id = node_nearest.first;
        const auto &img = _graph.getNode(node_id)->payload;
        const auto &nearest = node_nearest.second;

        // match & distort
        std::vector<std::tuple<size_t, CameraModel, std::vector<feature_2d>>> nearest_descriptors;
        nearest_descriptors.reserve(nearest.size());
        for (size_t match_node_id : nearest)
        {
            const auto *node = _graph.getNode(match_node_id);
            if (node != nullptr)
            {
                nearest_descriptors.emplace_back(match_node_id, node->payload.model, node->payload.features);
            }
        }
        std::vector<std::pair<size_t, camera_relations>> inlier_measurements;
        inlier_measurements.reserve(nearest_descriptors.size());
        for (const auto &node_descriptors : nearest_descriptors)
        {
            // match
            auto matches = match_features(img.features, std::get<2>(node_descriptors));

            // distort
            std::vector<correspondence> correspondences = distort_keypoints(
                img.features, std::get<2>(node_descriptors), matches, img.model, std::get<1>(node_descriptors));

            // ransac
            homography_model h;
            std::vector<bool> inliers;
            ransac(correspondences, h, inliers);

            camera_relations relations;
            relations.ransac_relation = h.homography;
            relations.relationType = camera_relations::RelationType::HOMOGRAPHY;

            bool can_decompose =
                h.decompose(correspondences, inliers, relations.relative_rotation, relations.relative_translation);

            if (can_decompose)
            {
                relations.inlier_matches.reserve(std::count(inliers.begin(), inliers.end(), true));
                for (size_t i = 0; i < inliers.size(); i++)
                {
                    if (inliers[i])
                    {
                        feature_match_denormalized fmd;
                        fmd.pixel_1 = img.features[matches[i].feature_index_1].location;
                        fmd.pixel_2 = std::get<2>(node_descriptors)[matches[i].feature_index_2].location;
                        relations.inlier_matches.push_back(fmd);
                    }
                }
                inlier_measurements.emplace_back(std::get<0>(node_descriptors), std::move(relations));
            }
        }

        std::lock_guard<std::mutex> graph_lock(_graph_structure_mutex);
        for (auto &node_measurements : inlier_measurements)
        {
            _graph.addEdge(std::move(node_measurements.second), node_id, node_measurements.first);
        }
    }

    // in serial to keep graph optimization deterministic
    for (size_t node_id : node_ids)
    {
        initializeOrientation({node_id}, _graph);
    }

    // TODO:
    // relaxSubset(node_ids, _graph);
}

void Pipeline::add(const std::vector<std::string> &paths)
{
    {
        std::lock_guard<std::mutex> guard(_queue_mutex);
        _add_queue.insert(_add_queue.end(), paths.begin(), paths.end());
        _queue_condition_variable.notify_all();
    }
}

Pipeline::Status Pipeline::getStatus()
{
    bool queue_empty;
    {
        std::lock_guard<std::mutex> guard(_queue_mutex);
        queue_empty = _add_queue.size() == 0;
    }
    return queue_empty && _runner.status == ThreadStatus::IDLE ? Status::COMPLETE : Status::PROCESSING;
}

const MeasurementGraph &Pipeline::getGraph()
{
    return _graph;
}
} // namespace opencalibration
