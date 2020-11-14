#include <opencalibration/pipeline/pipeline.hpp>

#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/extract/extract_features.hpp>
#include <opencalibration/extract/extract_metadata.hpp>
#include <opencalibration/match/match_features.hpp>
#include <opencalibration/model_inliers/ransac.hpp>

#include <omp.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace
{
std::array<double, 3> to_array(const Eigen::Vector3d &v)
{
    return {v.x(), v.y(), v.z()};
}
} // namespace

namespace opencalibration
{
Pipeline::Pipeline(size_t batch_size)
{
    _keep_running = true;
    omp_set_num_threads(batch_size);
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
        std::vector<size_t> loaded_ids, next_ids;
        paths.reserve(batch_size);
        loaded_ids.reserve(batch_size);
        next_ids.reserve(batch_size);
        while (_keep_running)
        {
            paths.clear();
            next_ids.clear();
            if (get_paths(paths, _runner.status) || loaded_ids.size() > 0)
            {
                process_images(loaded_ids, paths, next_ids);
                std::swap(loaded_ids, next_ids);
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

void Pipeline::process_images(const std::vector<size_t> &loaded_ids, const std::vector<std::string> &paths_to_load,
                              std::vector<size_t> &next_ids)
{
    spdlog::info("Building links");
    const std::vector<NodeLinks> links = find_links(loaded_ids);

    std::vector<std::function<void()>> funcs;
    funcs.reserve(paths_to_load.size() + links.size());

    std::vector<std::tuple<size_t, size_t, camera_relations>> all_inlier_measurements;
    all_inlier_measurements.reserve(links.size() * 10);
    std::mutex measurement_mutex;

    {
        for (const auto &link : links)
        {
            spdlog::debug("Node: {}", link.node_id);
            spdlog::debug("Links to:");
            for (size_t neighbor : link.link_ids)
                spdlog::debug("{:>30}", neighbor);
        }
        //             process_links(links);
        for (size_t i = 0; i < links.size(); i++)
        {
            auto run_func = [&, i]() {
                const auto &node_nearest = links[i];
                size_t node_id = node_nearest.node_id;
                const auto &img = _graph.getNode(node_id)->payload;
                const auto &nearest = node_nearest.link_ids;

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

                    bool can_decompose = h.decompose(correspondences, inliers, relations.relative_rotation,
                                                     relations.relative_translation);

                    size_t num_inliers = std::count(inliers.begin(), inliers.end(), true);

                    spdlog::debug("Matches: {}  inliers: {}  can_decompose: {}", matches.size(), num_inliers,
                                  can_decompose);
                    if (can_decompose && num_inliers > h.MINIMUM_POINTS * 1.5)
                    {
                        relations.inlier_matches.reserve(num_inliers);
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
                        std::lock_guard<std::mutex> lock(measurement_mutex);
                        all_inlier_measurements.emplace_back(node_id, std::get<0>(node_descriptors),
                                                             std::move(relations));
                    }
                }
            };
            funcs.push_back(run_func);
        }
    }
    {
        spdlog::info("Loading batch of {}", paths_to_load.size());
        for (const auto &path : paths_to_load)
            spdlog::debug(path);
        //             next_ids = build_nodes(paths_to_load);
        std::vector<size_t> node_ids;
        node_ids.reserve(paths_to_load.size());

        for (size_t i = 0; i < paths_to_load.size(); i++)
        {
            auto run_func = [&, i]() {
                const std::string &path = paths_to_load[i];
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
            };
            funcs.push_back(run_func);
        }

        // alternate between feature extraction and matching
        //         std::random_shuffle(funcs.begin(), funcs.end());
        for (size_t i = 0; i * 2 < funcs.size(); i += 2)
        {
            std::swap(funcs[i], funcs[i * 2]);
        }

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < (int)funcs.size(); i++)
        {
            funcs[i]();
        }

        std::lock_guard<std::mutex> graph_lock(_graph_structure_mutex);
        // sort node_ids by the path, since they may be out of order now
        std::sort(node_ids.begin(), node_ids.end(), [this](size_t node_id_1, size_t node_id_2) -> int {
            const auto &path1 = _graph.getNode(node_id_1)->payload.path;
            const auto &path2 = _graph.getNode(node_id_2)->payload.path;
            return path1 < path2;
        });
        next_ids = node_ids;

        for (auto &node_measurements : all_inlier_measurements)
        {
            _graph.addEdge(std::move(std::get<2>(node_measurements)), std::get<0>(node_measurements),
                           std::get<1>(node_measurements));
        }
    }

    initializeOrientation(loaded_ids, _graph);
    // TODO:
    // relaxSubset(node_ids, _graph, _graph_structure_mutex);
}

std::vector<Pipeline::NodeLinks> Pipeline::find_links(const std::vector<size_t> &node_ids)
{
    // build nearest in serial, since it is really fast
    std::vector<NodeLinks> batch_nearest;
    batch_nearest.reserve(node_ids.size());

    std::lock_guard<std::mutex> graph_lock(_graph_structure_mutex);
    for (size_t node_id : node_ids)
    {
        const auto &img = _graph.getNode(node_id)->payload;

        auto knn = _imageGPSLocations.searchKnn(to_array(img.position), 10);
        NodeLinks link;
        link.node_id = node_id;
        link.link_ids.reserve(knn.size());
        for (const auto &nn : knn)
        {
            if (nn.payload != node_id)
            {
                link.link_ids.push_back(nn.payload);
            }
        }
        batch_nearest.emplace_back(std::move(link));
    }
    return batch_nearest;
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
