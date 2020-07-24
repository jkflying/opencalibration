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
Pipeline::Pipeline(size_t threads)
{
    _keep_running = true;
    auto get_path = [this](std::string &path, ThreadStatus &status) -> bool {
        std::lock_guard<std::mutex> guard(_queue_mutex);
        if (_add_queue.size() > 0)
        {
            path = _add_queue.front();
            _add_queue.pop_front();
            status = ThreadStatus::BUSY;
            return true;
        }
        else
        {
            status = ThreadStatus::IDLE;
            return false;
        }
    };
    _runners.resize(threads);
    for (size_t i = 0; i < threads; i++)
    {
        _runners[i].thread.reset(new std::thread(
            [this, get_path](size_t index) {
                std::mutex sleep_mutex;
                while (_keep_running)
                {
                    std::string path;
                    if (get_path(path, _runners[index].status))
                    {
                        process_image(path);
                    }
                    else
                    {
                        std::unique_lock<std::mutex> lck(sleep_mutex);
                        _queue_condition_variable.wait_for(lck, 1s);
                    }
                }
            },
            i));
    }
}

Pipeline::~Pipeline()
{
    _keep_running = false;
    _queue_condition_variable.notify_all();
    for (auto &runner : _runners)
    {
        runner.thread->join();
    }
}

bool Pipeline::process_image(const std::string &path)
{

    image img;
    img.path = path;
    img.metadata = extract_metadata(img.path);
    img.descriptors = extract_features(img.path);

    // find N nearest
    std::vector<size_t> nearest;
    {
        std::lock_guard<std::mutex> kdtree_lock(_kdtree_mutex);
        auto knn =
            _imageGPSLocations.searchKnn({img.metadata.latitude, img.metadata.longitude, img.metadata.altitude}, 10);
        nearest.reserve(knn.size());
        for (const auto &nn : knn)
        {
            nearest.push_back(nn.payload);
        }
    }

    // match & distort
    std::vector<std::tuple<size_t, CameraModel, std::vector<feature_2d>>> nearest_descriptors;
    nearest_descriptors.reserve(nearest.size());
    {
        std::lock_guard<std::mutex> graph_lock(_graph_structure_mutex);
        for (size_t node_id : nearest)
        {
            const auto *node = _graph.getNode(node_id);
            if (node != nullptr)
            {
                nearest_descriptors.emplace_back(node_id, node->payload.model, node->payload.descriptors);
            }
        }
    }
    std::vector<std::pair<size_t, camera_relations>> inlier_measurements;
    inlier_measurements.reserve(nearest_descriptors.size());
    for (const auto &node_descriptors : nearest_descriptors)
    {
        // match
        auto matches = match_features(img.descriptors, std::get<2>(node_descriptors));

        // distort
        auto correspondences = distort_keypoints(img.descriptors, std::get<2>(node_descriptors), matches, img.model,
                                                 std::get<1>(node_descriptors));

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
                    fmd.pixel_1 = img.descriptors[matches[i].feature_index_1].location;
                    fmd.pixel_2 = std::get<2>(node_descriptors)[matches[i].feature_index_2].location;
                    relations.inlier_matches.push_back(fmd);
                }
            }
            inlier_measurements.emplace_back(std::get<0>(node_descriptors), std::move(relations));
        }
    }

    size_t node_id;
    {
        std::lock_guard<std::mutex> graph_lock(_graph_structure_mutex);
        node_id = _graph.addNode(std::move(img));

        for (auto &node_measurements : inlier_measurements)
        {
            _graph.addEdge(std::move(node_measurements.second), node_id, node_measurements.first);
        }
    }
    {
        std::lock_guard<std::mutex> kdtree_lock(_kdtree_mutex);
        _imageGPSLocations.addPoint({img.metadata.latitude, img.metadata.longitude, img.metadata.altitude}, node_id);
    }

    return true;
}

void Pipeline::add(const std::string &path)
{
    {
        std::lock_guard<std::mutex> guard(_queue_mutex);
        _add_queue.push_back(path);
        _queue_condition_variable.notify_one();
    }
}

Pipeline::Status Pipeline::getStatus()
{
    bool queue_empty;
    {
        std::lock_guard<std::mutex> guard(_queue_mutex);
        queue_empty = _add_queue.size() == 0;
    }
    return queue_empty && std::all_of(_runners.begin(), _runners.end(),
                                      [](const Runner &runner) -> bool { return runner.status == ThreadStatus::IDLE; })
               ? Status::COMPLETE
               : Status::PROCESSING;
}
} // namespace opencalibration
