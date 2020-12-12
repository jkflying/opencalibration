#include <opencalibration/pipeline/link_stage.hpp>

#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/match/match_features.hpp>
#include <opencalibration/model_inliers/ransac.hpp>

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

void BuildLinksStage::init(const MeasurementGraph &graph, const jk::tree::KDTree<size_t, 3> &imageGPSLocations,
                           const std::vector<size_t> &node_ids)
{
    spdlog::info("Queueing {} image nodes for link building", node_ids.size());
    _links.clear();
    _links.reserve(node_ids.size());

    // build nearest in serial, since it is really fast
    for (size_t node_id : node_ids)
    {
        const auto &img = graph.getNode(node_id)->payload;

        auto knn = imageGPSLocations.searchKnn(to_array(img.position), 10);
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
        _links.emplace_back(std::move(link));
    }
}

std::vector<std::function<void()>> BuildLinksStage::get_runners(const MeasurementGraph &graph)
{
    std::vector<std::function<void()>> funcs;
    funcs.reserve(_links.size());
    _all_inlier_measurements.reserve(10 * _links.size());
    for (size_t i = 0; i < _links.size(); i++)
    {
        auto run_func = [&, i]() {
            const auto &node_nearest = _links[i];
            size_t node_id = node_nearest.node_id;
            const auto &img = graph.getNode(node_id)->payload;
            const auto &nearest = node_nearest.link_ids;

            // match & distort
            std::vector<std::tuple<size_t, std::reference_wrapper<const image>>> nearest_images;
            nearest_images.reserve(nearest.size());
            for (size_t match_node_id : nearest)
            {
                const auto *node = graph.getNode(match_node_id);
                if (node != nullptr)
                {
                    nearest_images.emplace_back(match_node_id, node->payload);
                }
            }

            for (const auto &near_image : nearest_images)
            {
                // match
                auto matches = match_features(img.features, std::get<1>(near_image).get().features);

                // distort
                std::vector<correspondence> correspondences =
                    distort_keypoints(img.features, std::get<1>(near_image).get().features, matches, img.model,
                                      std::get<1>(near_image).get().model);

                // ransac
                homography_model h;
                std::vector<bool> inliers;
                ransac(correspondences, h, inliers);

                camera_relations relations;
                relations.ransac_relation = h.homography;
                relations.relationType = camera_relations::RelationType::HOMOGRAPHY;

                bool can_decompose =
                    h.decompose(correspondences, inliers, relations.relative_rotation, relations.relative_translation);

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
                            fmd.pixel_2 = std::get<1>(near_image).get().features[matches[i].feature_index_2].location;
                            relations.inlier_matches.push_back(fmd);
                        }
                    }
                    std::lock_guard<std::mutex> lock(_measurement_mutex);
                    _all_inlier_measurements.emplace_back(
                        inlier_measurement{node_id, std::get<0>(near_image), std::move(relations)});
                }
            }
        };
        funcs.push_back(run_func);
    }
    return funcs;
}

std::vector<size_t> BuildLinksStage::finalize(MeasurementGraph &graph)
{
    for (auto &measurements : _all_inlier_measurements)
    {
        graph.addEdge(std::move(measurements.relations), measurements.node_id, measurements.match_node_id);
    }
    _all_inlier_measurements.clear();

    std::vector<size_t> node_ids;
    node_ids.reserve(_links.size());
    for (const auto &link : _links)
    {
        node_ids.push_back(link.node_id);
    }
    _links.clear();

    return node_ids;
}

} // namespace opencalibration
