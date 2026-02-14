#include <opencalibration/pipeline/link_stage.hpp>

#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/match/match_features.hpp>
#include <opencalibration/model_inliers/fundamental_matrix_model.hpp>
#include <opencalibration/model_inliers/ransac.hpp>
#include <opencalibration/performance/performance.hpp>

#include <spdlog/spdlog.h>

namespace opencalibration
{

void LinkStage::init(const MeasurementGraph &graph, const jk::tree::KDTree<size_t, 2> &imageGPSLocations,
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

        auto knn = imageGPSLocations.searchKnn({img.position.x(), img.position.y()}, 10);
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
        auto &mtx = _measurement_mutex;
        auto &meas = _all_inlier_measurements;

        for (size_t match_node_id : nearest)
        {
            const auto *node = graph.getNode(match_node_id);
            if (node == nullptr)
            {
                continue;
            }
            const image &near_image = node->payload;
            auto run_func = [i, node_id, near_image, match_node_id, &img, &mtx, &meas]() {
                PerformanceMeasure p("Link runner coarse match");
                camera_relations relations;

                const double coarse_spacing_pixels = 40.0;
                std::vector<size_t> coarse_indices_1 =
                    spatially_subsample_feature_indices(img.features, coarse_spacing_pixels);
                std::vector<size_t> coarse_indices_2 =
                    spatially_subsample_feature_indices(near_image.features, coarse_spacing_pixels);

                std::vector<feature_match> coarse_matches =
                    match_features_subset(img.features, near_image.features, coarse_indices_1, coarse_indices_2);

                p.reset("Link runner coarse undistort");
                std::vector<correspondence> coarse_correspondences =
                    distort_keypoints(img.features, near_image.features, coarse_matches, *img.model, *near_image.model);

                p.reset("Link runner coarse ransac");
                homography_model h;
                std::vector<bool> coarse_inliers;
                ransac(coarse_correspondences, h, coarse_inliers);

                relations.ransac_relation = h.homography;
                relations.relationType = camera_relations::RelationType::HOMOGRAPHY;

                bool can_decompose = h.decompose(coarse_correspondences, coarse_inliers, relations.relative_poses);
                size_t num_coarse_inliers = std::count(coarse_inliers.begin(), coarse_inliers.end(), true);

                spdlog::trace("Coarse matches: {}  inliers: {}  can_decompose: {}", coarse_matches.size(),
                              num_coarse_inliers, can_decompose);

                if (can_decompose && num_coarse_inliers > h.MINIMUM_POINTS * 1.5)
                {
                    p.reset("Link runner fine match");

                    // Compute pixel-space fundamental matrix from coarse inliers for epipolar filtering
                    std::vector<correspondence> pixel_correspondences(coarse_matches.size());
                    for (size_t k = 0; k < coarse_matches.size(); k++)
                    {
                        const auto &loc1 = img.features[coarse_matches[k].feature_index_1].location;
                        const auto &loc2 = near_image.features[coarse_matches[k].feature_index_2].location;
                        pixel_correspondences[k].measurement1 = loc1.homogeneous();
                        pixel_correspondences[k].measurement2 = loc2.homogeneous();
                    }

                    fundamental_matrix_model fm;
                    fm.inlier_threshold = 15.0;
                    std::vector<bool> fm_inliers;
                    ransac(pixel_correspondences, fm, fm_inliers);
                    fm.fitInliers(pixel_correspondences, fm_inliers);

                    const double search_radius_pixels = 150.0;
                    const double epipolar_threshold_pixels = 12.0;
                    std::vector<feature_match> fine_matches = match_features_local_guided(
                        img.features, near_image.features, h.homography, search_radius_pixels, &fm.fundamental_matrix,
                        epipolar_threshold_pixels);

                    std::vector<feature_match> all_matches = coarse_matches;
                    all_matches.insert(all_matches.end(), fine_matches.begin(), fine_matches.end());

                    p.reset("Link runner fine undistort");
                    std::vector<correspondence> all_correspondences =
                        distort_keypoints(img.features, near_image.features, all_matches, *img.model, *near_image.model);

                    p.reset("Link runner fine ransac");
                    std::vector<bool> all_inliers;
                    ransac(all_correspondences, h, all_inliers);

                    relations.ransac_relation = h.homography;
                    size_t num_all_inliers = std::count(all_inliers.begin(), all_inliers.end(), true);

                    spdlog::trace("All matches: {}  inliers: {}", all_matches.size(), num_all_inliers);

                    if (num_all_inliers > h.MINIMUM_POINTS * 1.5)
                    {
                        relations.matches = all_matches;
                        assembleInliers(relations.matches, all_inliers, img.features, near_image.features,
                                        relations.inlier_matches);
                    }
                    else
                    {
                        relations.matches = coarse_matches;
                        assembleInliers(relations.matches, coarse_inliers, img.features, near_image.features,
                                        relations.inlier_matches);
                    }
                }
                std::lock_guard<std::mutex> lock(mtx);
                meas.emplace_back(edge_payload{i, node_id, match_node_id, std::move(relations)});
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
