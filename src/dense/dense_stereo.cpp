#include <opencalibration/dense/dense_stereo.hpp>

#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/geometry/intersection.hpp>
#include <opencalibration/surface/intersect.hpp>
#include <opencalibration/types/feature_2d.hpp>
#include <opencalibration/types/union_find.hpp>

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
        if (!img.dense_features.empty() && img.model && !img.position.hasNaN() && !img.orientation.coeffs().hasNaN())
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

    struct Measurement
    {
        size_t node_id, feat_idx;
    };
    std::vector<Measurement> id_to_measurement;
    ankerl::unordered_dense::map<size_t, size_t> node_id_to_offset;
    {
        size_t total_features = 0;
        for (size_t nid : node_ids)
        {
            node_id_to_offset[nid] = total_features;
            total_features += graph.getNode(nid)->payload.dense_features.size();
        }
        id_to_measurement.resize(total_features);
        for (size_t nid : node_ids)
        {
            size_t offset = node_id_to_offset[nid];
            size_t count = graph.getNode(nid)->payload.dense_features.size();
            for (size_t i = 0; i < count; i++)
            {
                id_to_measurement[offset + i] = {nid, i};
            }
        }
    }

    auto measurementId = [&](size_t nid, size_t feat_idx) -> size_t { return node_id_to_offset[nid] + feat_idx; };

    std::atomic<size_t> images_done{0};
    std::mutex uf_mutex;
    UnionFind uf(id_to_measurement.size());
    std::vector<Eigen::Vector3d> mesh_fallbacks(id_to_measurement.size(), Eigen::Vector3d(NAN, NAN, NAN));

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

        struct LocalMatch
        {
            size_t src_id, dst_id;
        };
        std::vector<LocalMatch> local_matches;
        std::vector<std::pair<size_t, Eigen::Vector3d>> local_fallbacks;

        auto camera_searcher = camera_tree.searcher();

        for (size_t fi : order)
        {
            const auto &feat = src_img.dense_features[fi];

            ray_d r = image_to_3d(feat.location, src_model, src_pos, src_ori);
            const auto &info = searcher.triangleIntersect(r);

            if (info.type != MeshIntersectionSearcher::IntersectionInfo::INTERSECTION)
                continue;

            const Eigen::Vector3d &pt3d = info.intersectionLocation;
            size_t src_id = measurementId(src_nid, fi);
            bool has_match = false;

            auto candidates = camera_searcher.search({pt3d.x(), pt3d.y(), pt3d.z()}, std::numeric_limits<double>::max(),
                                                     MAX_CANDIDATE_IMAGES + 1);

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

                Eigen::Vector2d predicted = image_from_3d(pt3d, cand_model, cand_pos, cand_ori);

                if (predicted.x() < 0 || predicted.x() >= cand_model.pixels_cols || predicted.y() < 0 ||
                    predicted.y() >= cand_model.pixels_rows)
                {
                    continue;
                }

                auto ft_it = feature_trees.find(cand_nid);
                if (ft_it == feature_trees.end())
                    continue;

                auto ft_searcher = ft_it->second.tree.searcher();
                const auto &nearby =
                    ft_searcher.search({predicted.x(), predicted.y()}, SEARCH_RADIUS_PIXELS * SEARCH_RADIUS_PIXELS,
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

                bool good_match = nearby.size() >= 2 ? best_dist < RATIO_THRESHOLD * second_best_dist
                                                     : best_dist < MAX_ABSOLUTE_DESCRIPTOR_DISTANCE;
                if (good_match)
                {
                    local_matches.push_back({src_id, measurementId(cand_nid, best_feat_idx)});
                    has_match = true;
                }
            }

            if (has_match)
            {
                local_fallbacks.push_back({src_id, pt3d});
            }
        }

        if (!local_matches.empty())
        {
            std::lock_guard<std::mutex> lock(uf_mutex);
            for (const auto &lm : local_matches)
            {
                uf.unite(lm.src_id, lm.dst_id);
            }
            for (const auto &[id, pt] : local_fallbacks)
            {
                mesh_fallbacks[id] = pt;
            }
        }

        size_t done = ++images_done;
        if (progress_cb && done % 10 == 0)
        {
            progress_cb(static_cast<float>(done) / static_cast<float>(num_nodes));
        }
    }

    ankerl::unordered_dense::map<size_t, std::vector<size_t>> track_ids;
    ankerl::unordered_dense::map<size_t, Eigen::Vector3d> track_fallbacks;

    for (size_t i = 0; i < id_to_measurement.size(); i++)
    {
        size_t root = uf.find(i);
        if (root == i && !mesh_fallbacks[i].allFinite())
            continue;

        track_ids[root].push_back(i);
        if (mesh_fallbacks[i].allFinite())
            track_fallbacks.try_emplace(root, mesh_fallbacks[i]);
    }

    point_cloud merged_points;
    merged_points.reserve(track_ids.size());

    for (const auto &[root, ids] : track_ids)
    {
        std::vector<ray_d> rays;
        rays.reserve(ids.size());

        for (size_t id : ids)
        {
            const auto &m = id_to_measurement[id];
            const auto &img = graph.getNode(m.node_id)->payload;
            rays.push_back(
                image_to_3d(img.dense_features[m.feat_idx].location, *img.model, img.position, img.orientation));
        }

        if (rays.size() >= 2)
        {
            auto triangulated = rayIntersection(rays);
            if (triangulated.first.allFinite() && triangulated.second >= 0)
            {
                merged_points.push_back(triangulated.first);
                continue;
            }
        }

        auto fb = track_fallbacks.find(root);
        if (fb != track_fallbacks.end())
            merged_points.push_back(fb->second);
    }

    if (!merged_points.empty())
    {
        surface.cloud.push_back(std::move(merged_points));
    }

    size_t total_points = 0;
    for (const auto &cloud : surface.cloud)
    {
        total_points += cloud.size();
    }
    spdlog::info("Dense: {} 3D points from {} tracks, {} images", total_points, track_ids.size(), node_ids.size());

    if (progress_cb)
        progress_cb(1.f);
}

} // namespace opencalibration
