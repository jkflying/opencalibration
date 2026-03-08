#include <opencalibration/dense/dense_stereo.hpp>

#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/geometry/intersection.hpp>
#include <opencalibration/surface/intersect.hpp>
#include <opencalibration/types/feature_2d.hpp>

#include <jk/KDTree.h>
#include <spdlog/spdlog.h>

#include <omp.h>

#include <algorithm>
#include <mutex>
#include <numeric>

namespace
{

// Hilbert curve distance for spatial locality when walking the mesh
uint32_t xy2d(int order, int x, int y)
{
    uint32_t d = 0;
    for (int s = order / 2; s > 0; s /= 2)
    {
        int rx = (x & s) > 0 ? 1 : 0;
        int ry = (y & s) > 0 ? 1 : 0;
        d += s * s * ((3 * rx) ^ ry);
        if (ry == 0)
        {
            if (rx == 1)
            {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            std::swap(x, y);
        }
    }
    return d;
}

std::vector<size_t> hilbertFeatureOrder(const std::vector<opencalibration::feature_2d> &features, int image_width,
                                        int image_height)
{
    int max_dim = std::max(image_width, image_height);
    int order = 1;
    while (order < max_dim)
        order *= 2;

    std::vector<std::pair<uint32_t, size_t>> indexed;
    indexed.reserve(features.size());
    for (size_t i = 0; i < features.size(); i++)
    {
        int x = std::clamp(static_cast<int>(features[i].location.x()), 0, image_width - 1);
        int y = std::clamp(static_cast<int>(features[i].location.y()), 0, image_height - 1);
        indexed.push_back({xy2d(order, x, y), i});
    }
    std::sort(indexed.begin(), indexed.end());

    std::vector<size_t> result;
    result.reserve(indexed.size());
    for (auto &p : indexed)
    {
        result.push_back(p.second);
    }
    return result;
}

constexpr double SEARCH_RADIUS_PIXELS = 20.0;
constexpr double RATIO_THRESHOLD = 0.8;
constexpr int MAX_CANDIDATE_IMAGES = 10;
constexpr double MAX_ABSOLUTE_DESCRIPTOR_DISTANCE = 0.3;

double descriptor_distance(const opencalibration::feature_2d &f1, const opencalibration::feature_2d &f2)
{
    return (f1.descriptor ^ f2.descriptor).count() * (1.0 / opencalibration::feature_2d::DESCRIPTOR_BITS);
}

} // namespace

namespace opencalibration
{

void densifyMesh(const MeasurementGraph &graph, std::vector<surface_model> &surfaces,
                 std::function<void(float)> progress_cb)
{
    if (surfaces.empty())
    {
        spdlog::warn("Dense: no surfaces to densify");
        if (progress_cb)
            progress_cb(1.f);
        return;
    }

    // Collect all image node IDs that have dense features and a valid camera
    std::vector<size_t> node_ids;
    for (auto it = graph.cnodebegin(); it != graph.cnodeend(); ++it)
    {
        const auto &img = it->second.payload;
        if (!img.dense_features.empty() && img.model && !img.position.hasNaN() &&
            !img.orientation.coeffs().hasNaN())
        {
            node_ids.push_back(it->first);
        }
    }

    if (node_ids.empty())
    {
        spdlog::info("Dense: no images with dense features");
        if (progress_cb)
            progress_cb(1.f);
        return;
    }

    spdlog::info("Dense: {} images with dense features, {} surfaces", node_ids.size(), surfaces.size());

    // Build KDTree of camera positions for finding candidate images
    jk::tree::KDTree<size_t, 3, 8> camera_tree;
    for (size_t nid : node_ids)
    {
        const auto &pos = graph.getNode(nid)->payload.position;
        camera_tree.addPoint({pos.x(), pos.y(), pos.z()}, nid);
    }

    // Build per-image KDTrees of dense features (indexed by node_id)
    struct ImageFeatureTree
    {
        jk::tree::KDTree<size_t, 2, 8> tree;
    };

    std::unordered_map<size_t, ImageFeatureTree> feature_trees;
    for (size_t nid : node_ids)
    {
        const auto &img = graph.getNode(nid)->payload;
        auto &ft = feature_trees[nid];
        for (size_t i = 0; i < img.dense_features.size(); i++)
        {
            const auto &loc = img.dense_features[i].location;
            ft.tree.addPoint({loc.x(), loc.y()}, i);
        }
    }

    // Use first surface's mesh for intersection
    auto &surface = surfaces[0];
    const auto &mesh = surface.mesh;

    std::atomic<size_t> images_done{0};
    std::mutex cloud_mutex;

    const int num_nodes = static_cast<int>(node_ids.size());
#pragma omp parallel for schedule(dynamic) // NOLINT(modernize-loop-convert)
    for (int ni = 0; ni < num_nodes; ni++)
    {
        const size_t src_nid = node_ids[ni];
        const auto *src_node = graph.getNode(src_nid);
        const auto &src_img = src_node->payload;
        const auto &src_model = *src_img.model;
        const auto &src_pos = src_img.position;
        const auto &src_ori = src_img.orientation;

        MeshIntersectionSearcher searcher;
        if (!searcher.init(mesh))
        {
            images_done++;
            continue;
        }

        auto order = hilbertFeatureOrder(src_img.dense_features, static_cast<int>(src_model.pixels_cols),
                                         static_cast<int>(src_model.pixels_rows));

        point_cloud local_points;

        auto camera_searcher = camera_tree.searcher();

        for (size_t fi : order)
        {
            const auto &feat = src_img.dense_features[fi];

            ray_d r = image_to_3d(feat.location, src_model, src_pos, src_ori);
            const auto &info = searcher.triangleIntersect(r);

            if (info.type != MeshIntersectionSearcher::IntersectionInfo::INTERSECTION)
                continue;

            const Eigen::Vector3d &pt3d = info.intersectionLocation;

            auto candidates =
                camera_searcher.search({pt3d.x(), pt3d.y(), pt3d.z()},
                                       std::numeric_limits<double>::max(), MAX_CANDIDATE_IMAGES + 1);

            for (const auto &candidate : candidates)
            {
                size_t cand_nid = candidate.payload;
                if (cand_nid == src_nid)
                    continue;

                const auto *cand_node = graph.getNode(cand_nid);
                const auto &cand_img = cand_node->payload;
                const auto &cand_model = *cand_img.model;
                const auto &cand_pos = cand_img.position;
                const auto &cand_ori = cand_img.orientation;

                Eigen::Vector2d predicted =
                    image_from_3d(pt3d, cand_model, cand_pos, cand_ori);

                if (predicted.x() < 0 || predicted.x() >= cand_model.pixels_cols || predicted.y() < 0 ||
                    predicted.y() >= cand_model.pixels_rows)
                {
                    continue;
                }

                // Search candidate's dense features near the predicted location
                auto ft_it = feature_trees.find(cand_nid);
                if (ft_it == feature_trees.end())
                    continue;

                auto ft_searcher = ft_it->second.tree.searcher();
                const auto &nearby = ft_searcher.search({predicted.x(), predicted.y()},
                                                        SEARCH_RADIUS_PIXELS * SEARCH_RADIUS_PIXELS,
                                                        std::numeric_limits<size_t>::max());

                if (nearby.empty())
                    continue;

                double best_dist = std::numeric_limits<double>::infinity();
                double second_best_dist = std::numeric_limits<double>::infinity();
                size_t best_feat_idx = 0;

                for (const auto &n : nearby)
                {
                    double d = descriptor_distance(feat, cand_img.dense_features[n.payload]);
                    if (d < second_best_dist)
                    {
                        if (d < best_dist)
                        {
                            second_best_dist = best_dist;
                            best_dist = d;
                            best_feat_idx = n.payload;
                        }
                        else
                        {
                            second_best_dist = d;
                        }
                    }
                }

                bool good_match = nearby.size() >= 2
                                     ? best_dist < RATIO_THRESHOLD * second_best_dist
                                     : best_dist < MAX_ABSOLUTE_DESCRIPTOR_DISTANCE;
                if (good_match)
                {
                    ray_d cand_ray = image_to_3d(cand_img.dense_features[best_feat_idx].location,
                                                 cand_model, cand_pos, cand_ori);
                    auto triangulated = rayIntersection(r, cand_ray);
                    if (triangulated.first.allFinite() && triangulated.second >= 0)
                        local_points.push_back(triangulated.first);
                    else
                        local_points.push_back(pt3d);
                    break;
                }
            }
        }

        if (!local_points.empty())
        {
            std::lock_guard<std::mutex> lock(cloud_mutex);
            surface.cloud.push_back(std::move(local_points));
        }

        size_t done = ++images_done;
        if (progress_cb && done % 10 == 0)
        {
            progress_cb(static_cast<float>(done) / static_cast<float>(num_nodes));
        }
    }

    size_t total_points = 0;
    for (const auto &cloud : surface.cloud)
    {
        total_points += cloud.size();
    }
    spdlog::info("Dense: added {} 3D points from {} images", total_points, node_ids.size());

    if (progress_cb)
        progress_cb(1.f);
}

} // namespace opencalibration
