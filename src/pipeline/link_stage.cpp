#include "link_stage.hpp"

#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/match/match_features.hpp>
#include <opencalibration/model_inliers/ransac.hpp>
#include <opencalibration/performance/performance.hpp>

#include <ankerl/unordered_dense.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <memory>

namespace opencalibration
{

namespace
{
struct LinkFeatures
{
    std::vector<feature_2d> features;
    std::vector<size_t> coarse_indices;
};
using WorkingSet = ankerl::unordered_dense::map<size_t, LinkFeatures>;

WorkingSet loadWorkingSet(const MeasurementGraph &graph, const std::vector<NodeLinks> &links)
{
    PerformanceMeasure p("Link load features");
    WorkingSet working_set;
    for (const auto &link : links)
    {
        working_set.try_emplace(link.node_id);
        for (size_t link_id : link.link_ids)
            if (graph.getNode(link_id) != nullptr)
                working_set.try_emplace(link_id);
    }

    std::vector<size_t> node_ids;
    node_ids.reserve(working_set.size());
    for (const auto &entry : working_set)
        node_ids.push_back(entry.first);

    const auto start = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < node_ids.size(); i++) // NOLINT(modernize-loop-convert)
    {
        const image &img = graph.getNode(node_ids[i])->payload;
        LinkFeatures &entry = working_set.find(node_ids[i])->second;
        const double coarse_spacing_pixels = 40.0;
        entry.features = img.features.load();
        entry.coarse_indices =
            spatially_subsample_feature_indices(entry.features, coarse_spacing_pixels, img.num_sparse_features);
    }
    spdlog::info("Link: loaded features of {} images for {} new in {:.2f}s", working_set.size(), links.size(),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    return working_set;
}
} // namespace

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
    _all_inlier_measurements.reserve(funcs_required);
    auto working_set = std::make_shared<const WorkingSet>(loadWorkingSet(graph, _links));
    for (size_t i = 0; i < _links.size(); i++)
    {
        const auto &node_nearest = _links[i];
        size_t node_id = node_nearest.node_id;
        const auto &img = graph.getNode(node_id)->payload;
        const auto &nearest = node_nearest.link_ids;

        auto &mtx = _measurement_mutex;
        auto &meas = _all_inlier_measurements;
        const auto &on_linked = this->on_linked;

        for (size_t match_node_id : nearest)
        {
            const auto *node = graph.getNode(match_node_id);
            if (node == nullptr)
            {
                continue;
            }
            const image &near_image = node->payload;
            auto run_func = [working_set, i, node_id, &near_image, match_node_id, &img, &mtx, &meas, &on_linked]() {
                PerformanceMeasure p("Link runner coarse match");
                camera_relations relations;
                const LinkFeatures &source = working_set->at(node_id);
                const LinkFeatures &dest = working_set->at(match_node_id);
                const auto &features = source.features;
                const auto &near_features = dest.features;

                std::vector<feature_match> coarse_matches =
                    match_features_subset(features, near_features, source.coarse_indices, dest.coarse_indices);

                p.reset("Link runner coarse undistort");
                std::vector<correspondence> coarse_correspondences =
                    distort_keypoints(features, near_features, coarse_matches, *img.model, *near_image.model);

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
                    relations.matches = coarse_matches;
                    assembleInliers(relations.matches, coarse_inliers, features, near_features,
                                    relations.inlier_matches);
                }
                if (on_linked && !relations.inlier_matches.empty())
                    on_linked(node_id, match_node_id);
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
