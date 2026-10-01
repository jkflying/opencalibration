#include <opencalibration/dense/densify_multiview.hpp>

#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/geometry/intersection.hpp>
#include <opencalibration/performance/performance.hpp>
#include <opencalibration/surface/intersect.hpp>
#include <opencalibration/surface/refine_mesh.hpp>
#include <opencalibration/types/feature_2d.hpp>
#include <opencalibration/types/hilbert.hpp>
#include <opencalibration/types/union_find.hpp>

#include <jk/KDTree.h>
#include <spdlog/spdlog.h>

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <mutex>
#include <numeric>

namespace
{

std::vector<size_t> hilbertFeatureOrder(const std::vector<opencalibration::feature_2d> &features, size_t start_index,
                                        int image_width, int image_height)
{
    int max_dim = std::max(image_width, image_height);
    int order = 1;
    while (order < max_dim)
        order *= 2;

    std::vector<std::pair<uint32_t, size_t>> indexed;
    indexed.reserve(features.size() - start_index);
    for (size_t i = start_index; i < features.size(); i++)
    {
        int x = std::clamp(static_cast<int>(features[i].location.x()), 0, image_width - 1);
        int y = std::clamp(static_cast<int>(features[i].location.y()), 0, image_height - 1);
        indexed.push_back({opencalibration::xy2d(order, x, y), i - start_index});
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

constexpr double SEARCH_RADIUS_PIXELS = 150.0;
constexpr double RATIO_THRESHOLD = 0.85;
constexpr int MAX_CANDIDATE_IMAGES = 10;
constexpr double MAX_ABSOLUTE_DESCRIPTOR_DISTANCE = 0.35;
constexpr double MAX_REPROJECTION_ERROR_PIXELS = 8.0;

constexpr int TRIANGULATION_REFINE_ITERATIONS = 10;

using Descriptor = std::bitset<opencalibration::feature_2d::DESCRIPTOR_BITS>;

size_t hammingDistance(const Descriptor &a, const Descriptor &b)
{
    return (a ^ b).count();
}

class CellSortedFeatures
{
  public:
    CellSortedFeatures() = default;

    explicit CellSortedFeatures(const opencalibration::image &img)
    {
        const size_t first = img.num_sparse_features;
        const size_t count = img.features.size() - first;
        if (count == 0)
            return;

        _origin = img.features[first].location;
        Eigen::Vector2d max = _origin;
        for (size_t i = first; i < img.features.size(); i++)
        {
            _origin = _origin.cwiseMin(img.features[i].location);
            max = max.cwiseMax(img.features[i].location);
        }
        _cols = cellOf(max.x() - _origin.x()) + 1;
        _rows = cellOf(max.y() - _origin.y()) + 1;

        std::vector<uint32_t> featureCell(count);
        _cellBegin.assign(static_cast<size_t>(_cols) * _rows + 1, 0);
        for (size_t i = 0; i < count; i++)
        {
            const Eigen::Vector2d &loc = img.features[first + i].location;
            featureCell[i] = cellIndex(cellOf(loc.y() - _origin.y()), cellOf(loc.x() - _origin.x()));
            _cellBegin[featureCell[i] + 1]++;
        }
        std::partial_sum(_cellBegin.begin(), _cellBegin.end(), _cellBegin.begin());

        _locations.resize(count);
        _descriptors.resize(count);
        _imageFeatureIndex.resize(count);
        std::vector<uint32_t> nextSlotInCell(_cellBegin.begin(), _cellBegin.end() - 1);
        for (size_t i = 0; i < count; i++)
        {
            const uint32_t slot = nextSlotInCell[featureCell[i]]++;
            _locations[slot] = img.features[first + i].location;
            _descriptors[slot] = img.features[first + i].descriptor;
            _imageFeatureIndex[slot] = first + i;
        }
    }

    static constexpr size_t NO_MATCH = std::numeric_limits<size_t>::max();

    [[nodiscard]] size_t ratioTestMatchNear(const opencalibration::feature_2d &query,
                                            const Eigen::Vector2d &center) const
    {
        size_t nearby = 0;
        double best_dist = std::numeric_limits<double>::infinity();
        double second_best_dist = std::numeric_limits<double>::infinity();
        size_t best_slot = 0;
        forEachSlotWithinSearchRadius(center, [&](uint32_t slot) {
            nearby++;
            const double d = hammingDistance(query.descriptor, _descriptors[slot]) *
                             (1.0 / opencalibration::feature_2d::DESCRIPTOR_BITS);
            if (d < second_best_dist)
            {
                if (d < best_dist)
                {
                    second_best_dist = best_dist;
                    best_dist = d;
                    best_slot = slot;
                }
                else
                {
                    second_best_dist = d;
                }
            }
        });

        if (nearby == 0)
            return NO_MATCH;
        bool good_match =
            nearby >= 2 ? best_dist < RATIO_THRESHOLD * second_best_dist : best_dist < MAX_ABSOLUTE_DESCRIPTOR_DISTANCE;
        if (!good_match)
            return NO_MATCH;
        return _imageFeatureIndex[best_slot];
    }

  private:
    static constexpr double CELL_SIZE_PIXELS = SEARCH_RADIUS_PIXELS / 4;

    static int cellOf(double offset)
    {
        return static_cast<int>(std::floor(offset * (1 / CELL_SIZE_PIXELS)));
    }

    [[nodiscard]] size_t cellIndex(int row, int col) const
    {
        return static_cast<size_t>(row) * _cols + col;
    }

    template <typename Visit> void forEachSlotWithinSearchRadius(const Eigen::Vector2d &center, Visit &&visit) const
    {
        if (_locations.empty() || !center.allFinite())
            return;

        const Eigen::Vector2d lo = center - _origin - Eigen::Vector2d::Constant(SEARCH_RADIUS_PIXELS);
        const Eigen::Vector2d hi = center - _origin + Eigen::Vector2d::Constant(SEARCH_RADIUS_PIXELS);
        const int firstCol = std::max(0, cellOf(lo.x())), lastCol = std::min(_cols - 1, cellOf(hi.x()));
        const int firstRow = std::max(0, cellOf(lo.y())), lastRow = std::min(_rows - 1, cellOf(hi.y()));
        if (firstCol > lastCol)
            return;

        constexpr double radiusSquared = SEARCH_RADIUS_PIXELS * SEARCH_RADIUS_PIXELS;
        for (int row = firstRow; row <= lastRow; row++)
        {
            const uint32_t rowSpanEnd = _cellBegin[cellIndex(row, lastCol) + 1];
            for (uint32_t slot = _cellBegin[cellIndex(row, firstCol)]; slot < rowSpanEnd; slot++)
            {
                if ((_locations[slot] - center).squaredNorm() < radiusSquared)
                    visit(slot);
            }
        }
    }

    Eigen::Vector2d _origin{0, 0};
    int _cols = 0, _rows = 0;
    std::vector<uint32_t> _cellBegin;
    std::vector<Eigen::Vector2d> _locations;
    std::vector<Descriptor> _descriptors;
    std::vector<size_t> _imageFeatureIndex;
};

struct RayMeasurement
{
    opencalibration::ray_d ray;
    Eigen::Vector2d pixel;
    const opencalibration::CameraModel *model;
    const Eigen::Vector3d *position;
    const Eigen::Quaterniond *orientation;
};

Eigen::Vector2d reprojectionResidual(const Eigen::Vector3d &point, const RayMeasurement &m)
{
    return opencalibration::image_from_3d(point, *m.model, *m.position, *m.orientation) - m.pixel;
}

// Gauss-Newton on pixel reprojection error, starting from the algebraic ray-midpoint solution.
// Jacobians are central differences; a step is only taken if it lowers the cost.
Eigen::Vector3d refineTriangulation(Eigen::Vector3d point, const std::vector<RayMeasurement> &measurements,
                                    const std::vector<size_t> &use)
{
    auto cost = [&](const Eigen::Vector3d &p) {
        double c = 0;
        for (size_t i : use)
            c += reprojectionResidual(p, measurements[i]).squaredNorm();
        return c;
    };

    double current_cost = cost(point);
    for (int iter = 0; iter < TRIANGULATION_REFINE_ITERATIONS; iter++)
    {
        Eigen::Matrix3d JtJ = Eigen::Matrix3d::Zero();
        Eigen::Vector3d Jtr = Eigen::Vector3d::Zero();
        for (size_t i : use)
        {
            const auto &m = measurements[i];
            const double h = 1e-6 * std::max(1.0, (point - *m.position).norm());
            Eigen::Matrix<double, 2, 3> J;
            for (int k = 0; k < 3; k++)
            {
                Eigen::Vector3d dp = Eigen::Vector3d::Zero();
                dp[k] = h;
                J.col(k) = (reprojectionResidual(point + dp, m) - reprojectionResidual(point - dp, m)) / (2 * h);
            }
            const Eigen::Vector2d r = reprojectionResidual(point, m);
            JtJ += J.transpose() * J;
            Jtr += J.transpose() * r;
        }

        const Eigen::Vector3d step = JtJ.ldlt().solve(-Jtr);
        if (!step.allFinite())
            break;
        const Eigen::Vector3d candidate = point + step;
        const double candidate_cost = cost(candidate);
        if (!(candidate_cost < current_cost))
            break;
        const double improvement = current_cost - candidate_cost;
        point = candidate;
        current_cost = candidate_cost;
        if (improvement < 1e-6 * current_cost)
            break;
    }
    return point;
}

bool inFrontOfCameras(const Eigen::Vector3d &point, const std::vector<RayMeasurement> &measurements,
                      const std::vector<size_t> &use)
{
    for (size_t i : use)
    {
        const auto &m = measurements[i];
        if ((m.orientation->inverse() * (point - *m.position)).z() <= 0)
            return false;
    }
    return true;
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

    std::vector<size_t> node_ids;
    for (auto it = graph.cnodebegin(); it != graph.cnodeend(); ++it)
    {
        const auto &img = it->second.payload;
        if (img.features.size() > img.num_sparse_features && img.model && !img.position.hasNaN() &&
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

    auto phase_start = std::chrono::steady_clock::now();
    auto lapSeconds = [&phase_start] {
        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(now - phase_start).count();
        phase_start = now;
        return seconds;
    };
    PerformanceMeasure p("Dense feature sort");

    jk::tree::KDTree<size_t, 3, 8> camera_tree;
    for (size_t nid : node_ids)
    {
        const auto &pos = graph.getNode(nid)->payload.position;
        camera_tree.addPoint({pos.x(), pos.y(), pos.z()}, nid);
    }

    std::vector<CellSortedFeatures> cell_sorted_features(node_ids.size());
    const int num_nodes_ft = static_cast<int>(node_ids.size());
#pragma omp parallel for schedule(dynamic) // NOLINT(modernize-loop-convert)
    for (int ni = 0; ni < num_nodes_ft; ni++)
    {
        cell_sorted_features[ni] = CellSortedFeatures(graph.getNode(node_ids[ni])->payload);
    }
    ankerl::unordered_dense::map<size_t, const CellSortedFeatures *> features_by_node;
    for (size_t ni = 0; ni < node_ids.size(); ni++)
    {
        features_by_node[node_ids[ni]] = &cell_sorted_features[ni];
    }

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
            const auto &img = graph.getNode(nid)->payload;
            size_t dense_count = img.features.size() - img.num_sparse_features;
            node_id_to_offset[nid] = total_features;
            total_features += dense_count;
        }
        id_to_measurement.resize(total_features);
        for (size_t nid : node_ids)
        {
            const auto &img = graph.getNode(nid)->payload;
            size_t offset = node_id_to_offset[nid];
            size_t dense_count = img.features.size() - img.num_sparse_features;
            for (size_t i = 0; i < dense_count; i++)
            {
                id_to_measurement[offset + i] = {nid, img.num_sparse_features + i};
            }
        }
    }

    auto measurementId = [&](size_t nid, size_t feat_idx) -> size_t {
        return node_id_to_offset.at(nid) + feat_idx - graph.getNode(nid)->payload.num_sparse_features;
    };

    spdlog::info("Dense: sorted features of {} images into cells, {} measurements in {:.1f}s", node_ids.size(),
                 id_to_measurement.size(), lapSeconds());
    p.reset("Dense match");

    std::atomic<size_t> images_done{0};
    std::mutex uf_mutex;
    std::mutex progress_mutex;
    UnionFind uf(id_to_measurement.size());
    constexpr size_t NO_SURFACE = std::numeric_limits<size_t>::max();
    std::vector<size_t> measurement_surface(id_to_measurement.size(), NO_SURFACE);

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

        std::vector<MeshIntersectionSearcher> searchers(surfaces.size());
        for (size_t si = 0; si < surfaces.size(); si++)
            if (!searchers[si].init(surfaces[si].mesh))
                searchers[si] = MeshIntersectionSearcher();

        auto order =
            hilbertFeatureOrder(src_img.features, src_img.num_sparse_features, static_cast<int>(src_model.pixels_cols),
                                static_cast<int>(src_model.pixels_rows));

        struct LocalMatch
        {
            size_t src_id, dst_id;
        };
        std::vector<LocalMatch> local_matches;

        auto camera_searcher = camera_tree.searcher();
        const CellSortedFeatures &src_features = *features_by_node.at(src_nid);

        for (size_t fi : order)
        {
            const size_t global_fi = src_img.num_sparse_features + fi;
            const auto &feat = src_img.features[global_fi];

            ray_d r = image_to_3d(feat.location, src_model, src_pos, src_ori);
            size_t surface_index = 0;
            while (surface_index < searchers.size() && searchers[surface_index].triangleIntersect(r).type !=
                                                           MeshIntersectionSearcher::IntersectionInfo::INTERSECTION)
                surface_index++;
            if (surface_index == searchers.size())
                continue;

            const Eigen::Vector3d pt3d = searchers[surface_index].lastResult().intersectionLocation;
            size_t src_id = measurementId(src_nid, global_fi);
            measurement_surface[src_id] = surface_index;

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

                auto cand_features = features_by_node.find(cand_nid);
                if (cand_features == features_by_node.end())
                    continue;

                const size_t forward = cand_features->second->ratioTestMatchNear(feat, predicted);
                if (forward == CellSortedFeatures::NO_MATCH)
                    continue;

                // Mutual check: the candidate's best match back in the source image must be this feature. Centre the
                // reverse search where the candidate maps to, assuming the local src->cand offset is a translation.
                const auto &cand_feat = cand_img.features[forward];
                const Eigen::Vector2d reverse_center = feat.location + (cand_feat.location - predicted);
                if (src_features.ratioTestMatchNear(cand_feat, reverse_center) == global_fi)
                {
                    local_matches.push_back({src_id, measurementId(cand_nid, forward)});
                }
            }
        }

        if (!local_matches.empty())
        {
            std::lock_guard<std::mutex> lock(uf_mutex);
            for (const auto &lm : local_matches)
            {
                uf.unite(lm.src_id, lm.dst_id);
            }
        }

        size_t done = ++images_done;
        if (progress_cb && done % 10 == 0)
        {
            std::lock_guard<std::mutex> lock(progress_mutex);
            progress_cb(static_cast<float>(done) / static_cast<float>(num_nodes));
        }
    }

    spdlog::info("Dense: matched {} images in {:.1f}s", node_ids.size(), lapSeconds());
    p.reset("Dense track build");

    ankerl::unordered_dense::map<size_t, std::vector<size_t>> track_ids;

    for (size_t i = 0; i < id_to_measurement.size(); i++)
    {
        if (uf.is_singleton(i))
            continue;
        size_t root = uf.find(i);
        track_ids[root].push_back(i);
    }

    const double max_reproj_err_sq = MAX_REPROJECTION_ERROR_PIXELS * MAX_REPROJECTION_ERROR_PIXELS;

    auto hasMultipleFeaturesFromOneImage = [&id_to_measurement](const std::vector<size_t> &ids) {
        ankerl::unordered_dense::set<size_t> track_nodes;
        for (size_t id : ids)
        {
            if (!track_nodes.insert(id_to_measurement[id].node_id).second)
                return true;
        }
        return false;
    };

    std::vector<std::vector<size_t>> multi_tracks;
    multi_tracks.reserve(track_ids.size());
    for (auto &[root, ids] : track_ids)
    {
        if (ids.size() >= 2 && !hasMultipleFeaturesFromOneImage(ids))
            multi_tracks.push_back(std::move(ids));
    }

    spdlog::info("Dense: built {} tracks, {} multi-view in {:.1f}s", track_ids.size(), multi_tracks.size(),
                 lapSeconds());
    p.reset("Dense triangulate");

    std::vector<Eigen::Vector3d> track_results(multi_tracks.size());
    std::vector<char> track_valid(multi_tracks.size(), 0);

    const int num_tracks = static_cast<int>(multi_tracks.size());
#pragma omp parallel for schedule(dynamic) // NOLINT(modernize-loop-convert)
    for (int ti = 0; ti < num_tracks; ti++)
    {
        const auto &ids = multi_tracks[ti];

        std::vector<RayMeasurement> measurements;
        for (size_t id : ids)
        {
            const auto &m = id_to_measurement[id];
            const auto &img = graph.getNode(m.node_id)->payload;
            const auto &pixel = img.features[m.feat_idx].location;
            measurements.push_back({image_to_3d(pixel, *img.model, img.position, img.orientation), pixel,
                                    img.model.get(), &img.position, &img.orientation});
        }

        std::vector<ray_d> rays;
        for (const auto &rm : measurements)
            rays.push_back(rm.ray);

        auto triangulated = rayIntersection(rays);
        if (!triangulated.first.allFinite() || triangulated.second < 0)
            continue;

        std::vector<size_t> all_indices(measurements.size());
        std::iota(all_indices.begin(), all_indices.end(), 0);
        Eigen::Vector3d point = refineTriangulation(triangulated.first, measurements, all_indices);

        std::vector<size_t> inlier_indices;
        for (size_t i = 0; i < measurements.size(); i++)
        {
            if (reprojectionResidual(point, measurements[i]).squaredNorm() <= max_reproj_err_sq)
                inlier_indices.push_back(i);
        }

        if (inlier_indices.size() < 2)
            continue;

        if (inlier_indices.size() < measurements.size())
        {
            // Restart from the inliers' algebraic solution so outliers don't bias the initial guess
            rays.clear();
            for (size_t i : inlier_indices)
                rays.push_back(measurements[i].ray);
            triangulated = rayIntersection(rays);
            if (!triangulated.first.allFinite() || triangulated.second < 0)
                continue;
            point = refineTriangulation(triangulated.first, measurements, inlier_indices);

            bool all_inliers = std::all_of(inlier_indices.begin(), inlier_indices.end(), [&](size_t i) {
                return reprojectionResidual(point, measurements[i]).squaredNorm() <= max_reproj_err_sq;
            });
            if (!all_inliers)
                continue;
        }

        if (!point.allFinite() || !inFrontOfCameras(point, measurements, inlier_indices))
            continue;

        track_results[ti] = point;
        track_valid[ti] = true;
    }

    spdlog::info("Dense: triangulated {} tracks in {:.1f}s", multi_tracks.size(), lapSeconds());
    p.reset("Dense assign points");

    std::vector<point_cloud> surface_points(surfaces.size());
    for (int ti = 0; ti < num_tracks; ti++)
    {
        if (!track_valid[ti])
            continue;
        const auto &ids = multi_tracks[ti];
        const auto with_surface =
            std::find_if(ids.begin(), ids.end(), [&](size_t id) { return measurement_surface[id] != NO_SURFACE; });
        if (with_surface != ids.end())
            surface_points[measurement_surface[*with_surface]].push_back(track_results[ti]);
    }

    size_t total_points = 0;
    for (size_t si = 0; si < surfaces.size(); si++)
    {
        if (surface_points[si].empty())
            continue;
        total_points += surface_points[si].size();
        surfaces[si].cloud.push_back(std::move(surface_points[si]));
    }
    spdlog::info("Dense: {} 3D points from {} tracks, {} images", total_points, track_ids.size(), node_ids.size());

    if (progress_cb)
        progress_cb(1.f);
}

} // namespace opencalibration
