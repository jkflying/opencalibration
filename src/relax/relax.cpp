#include <opencalibration/relax/relax.hpp>

#include <opencalibration/distort/distort_keypoints.hpp>

#include <ceres/ceres.h>

#include <unordered_set>

namespace
{
using namespace opencalibration;

struct RayIntersectionAngleError
{
    RayIntersectionAngleError(const feature_match_denormalized &feature_match, const image &img1, const image &img2)
        : _feature_match(feature_match), _img1(img1), _img2(img2)
    {
    }

    template <typename T> bool operator()(const T *const q1, T *residual) const
    {
        Eigen::Quaternion<T> q2 = _img2.orientation.cast<T>();
        return operator()(q1, &q2.coeffs()[0], residual);
    }

    template <typename T> bool operator()(const T *const q1, const T *const q2, T *r) const
    {
        using Matrix3T = Eigen::Matrix<T, 3, 1>;
        const Eigen::Quaternion<T> ori1(q1), ori2(q2);
        const Matrix3T pos1(_img1.position.cast<T>()), pos2(_img2.position.cast<T>());

        const Matrix3T ray1 = ori1 * image_to_3d(_feature_match.pixel_1, _img1.model).cast<T>();
        const Matrix3T ray2 = ori2 * image_to_3d(_feature_match.pixel_2, _img2.model).cast<T>();

        const Matrix3T n = ray1.cross(ray2).normalized(), n2 = ray2.cross(n);

        const Matrix3T ray1_nearest_point = pos1 + (pos2 - pos1).dot(n2) / ray1.dot(n2) * ray1;
        const Matrix3T ray2_nearest_point = pos2 + (pos1 - pos2).dot(n) / ray2.dot(n) * ray2;

        const Matrix3T ray_angle_error = (ray1_nearest_point - ray2_nearest_point) /
                                         ((ray1_nearest_point - pos1).norm() + (ray2_nearest_point - pos2).norm());

        Eigen::Map<Matrix3T> residual(r);
        residual = ray_angle_error;

        return true;
    }

    const feature_match_denormalized _feature_match;
    const image &_img1, &_img2;
};

} // namespace

namespace opencalibration
{

void initializeOrientation(const std::vector<size_t> &node_ids, MeasurementGraph &graph)
{
    for (size_t node_id : node_ids)
    {
        MeasurementGraph::Node *node1 = graph.getNode(node_id);
        if (node1 == nullptr)
            return;

        // TODO: get connections, use relative orientations from RANSAC to initialize orientations
        node1->payload.orientation = Eigen::Quaterniond::Identity();
    }
}

void relaxSubset(const std::vector<size_t> &node_ids, MeasurementGraph &graph, std::mutex &graph_mutex)
{
    ceres::Problem problem;

    {
        std::lock_guard<std::mutex> lock(graph_mutex);

        std::unordered_set<size_t> added_nodes;
        for (size_t node_id : node_ids)
        {
            MeasurementGraph::Node *node1 = graph.getNode(node_id);

            if (node1 == nullptr)
                continue;

            for (size_t edge_id : node1->getEdges())
            {
                MeasurementGraph::Edge *edge = graph.getEdge(edge_id);

                if (edge == nullptr)
                    continue;

                const size_t node_id2 = edge->getSource() == node_id ? edge->getDest() : edge->getSource();
                MeasurementGraph::Node *node2 = graph.getNode(node_id2);

                if (node2 == nullptr)
                    continue;

                bool added = false;
                for (const auto &match : edge->payload.inlier_matches)
                {
                    added = true;
                    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<RayIntersectionAngleError, 3, 4, 4>(
                                                 new RayIntersectionAngleError(match, node1->payload, node2->payload)),
                                             nullptr, &(node1->payload.orientation.coeffs()[0]),
                                             &(node2->payload.orientation.coeffs()[0]));
                }
                if (edge->payload.inlier_matches.size() > 0)
                {
                    added_nodes.insert(node_id);
                    added_nodes.insert(node_id2);
                }
            }
        }
        ceres::EigenQuaternionParameterization q_parameterization;
        for (size_t node_id : added_nodes)
        {
            problem.SetParameterization(&graph.getNode(node_id)->payload.orientation.coeffs()[0], &q_parameterization);
        }
        // set nodes not to optimize as constant
        for (size_t node_id : node_ids)
        {
            added_nodes.erase(node_id);
        }
        for (size_t node_id : added_nodes)
        {
            problem.SetParameterBlockConstant(&graph.getNode(node_id)->payload.orientation.coeffs()[0]);
        }
        ceres::Solver::Summary summary;
        ceres::Solver::Options options;
        ceres::Solve(options, &problem, &summary);
    }
}
} // namespace opencalibration
