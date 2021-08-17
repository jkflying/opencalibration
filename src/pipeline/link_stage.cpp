#include <opencalibration/pipeline/link_stage.hpp>

#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/match/match_features.hpp>
#include <opencalibration/model_inliers/ransac.hpp>
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

void LinkStage::init(const MeasurementGraph &graph, const jk::tree::KDTree<size_t, 3> &imageGPSLocations,
                     const std::vector<size_t> &node_ids)
{
    PerformanceMeasure p("Link init");
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

std::vector<std::function<void()>> LinkStage::get_runners(const MeasurementGraph &graph)
{
    std::vector<std::function<void()>> funcs;

    size_t funcs_required = 0;
    for (const auto &l : _links)
    {
        funcs_required += l.link_ids.size();
    }

    funcs.reserve(funcs_required);
    _all_inlier_measurements.reserve(10 * _links.size());
    for (size_t i = 0; i < _links.size(); i++)
    {
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

        auto &mtx = _measurement_mutex;
        auto &meas = _all_inlier_measurements;

        for (const auto near_image : nearest_images)
        {
            auto run_func = [i, node_id, near_image, &img, &mtx, &meas]() {
                PerformanceMeasure p("Link runner match");
                camera_relations relations;

                // match
                relations.matches = match_features(img.features, std::get<1>(near_image).get().features);

                // distort
                p.reset("Link runner undistort");
                std::vector<correspondence> correspondences =
                    distort_keypoints(img.features, std::get<1>(near_image).get().features, relations.matches,
                                      *img.model, *std::get<1>(near_image).get().model);

                // ransac
                p.reset("Link runner ransac");
                homography_model h;
                std::vector<bool> inliers;
                ransac(correspondences, h, inliers);

                p.reset("Link runner decompose");
                relations.ransac_relation = h.homography;
                relations.relationType = camera_relations::RelationType::HOMOGRAPHY;

                bool can_decompose = h.decompose(correspondences, inliers, relations.relative_poses);

                size_t num_inliers = std::count(inliers.begin(), inliers.end(), true);

                spdlog::trace("Matches: {}  inliers: {}  can_decompose: {}", relations.matches.size(), num_inliers,
                              can_decompose);

                if (can_decompose && num_inliers > h.MINIMUM_POINTS * 1.5)
                {
                    relations.inlier_matches.reserve(num_inliers);
                    for (size_t i = 0; i < relations.matches.size(); i++)
                    {
                        if (inliers[i])
                        {
                            feature_match_denormalized fmd;
                            fmd.pixel_1 = img.features[relations.matches[i].feature_index_1].location;
                            fmd.pixel_2 =
                                std::get<1>(near_image).get().features[relations.matches[i].feature_index_2].location;
                            fmd.feature_index_1 = relations.matches[i].feature_index_1;
                            fmd.feature_index_2 = relations.matches[i].feature_index_2;
                            relations.inlier_matches.push_back(fmd);
                        }
                    }
                }
                std::lock_guard<std::mutex> lock(mtx);

                meas.emplace_back(edge_payload{i, node_id, std::get<0>(near_image), std::move(relations)});
            };
            funcs.push_back(run_func);
        }
    }
    return funcs;
}

std::vector<size_t> LinkStage::finalize(MeasurementGraph &graph)
{
    PerformanceMeasure p("Link finalize");
    // put them back in the order they would have been if they were calculated serially
    std::sort(_all_inlier_measurements.begin(), _all_inlier_measurements.end(), [](const auto &a, const auto &b) {
        return std::make_tuple(a.loop_index, a.node_id, a.match_node_id) <
               std::make_tuple(b.loop_index, b.node_id, b.match_node_id);
    });

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
