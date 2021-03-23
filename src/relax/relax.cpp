#include <opencalibration/relax/relax.hpp>

#include <opencalibration/distort/distort_keypoints.hpp>

#include <ceres/ceres.h>
#include <spdlog/spdlog.h>

#include <unordered_set>

namespace opencalibration
{

static const Eigen::Quaterniond DOWN_ORIENTED_NORTH(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));

void initializeOrientation(const MeasurementGraph &graph, std::vector<NodePose> &nodes)
{
    std::vector<std::pair<double, Eigen::Quaterniond>> hypotheses;
    for (NodePose &node_pose : nodes)
    {
        const MeasurementGraph::Node *node = graph.getNode(node_pose.node_id);
        if (node == nullptr)
        {
            spdlog::error("Null node referenced from optimization list");
            continue;
        }

        // get connections
        hypotheses.clear();
        hypotheses.reserve(node->getEdges().size());

        for (size_t edge_id : node->getEdges())
        {
            const MeasurementGraph::Edge *edge = graph.getEdge(edge_id);
            size_t other_node_id = edge->getDest() == node_pose.node_id ? edge->getSource() : edge->getDest();
            const MeasurementGraph::Node *other_node = graph.getNode(other_node_id);

            Eigen::Quaterniond transform = edge->getDest() == node_pose.node_id
                                               ? edge->payload.relative_rotation.inverse()
                                               : edge->payload.relative_rotation;

            // TODO: use relative positions as well when NaN
            bool other_ori_nan = other_node->payload.orientation.coeffs().hasNaN();
            Eigen::Quaterniond other_orientation =
                other_ori_nan ? DOWN_ORIENTED_NORTH : other_node->payload.orientation;
            double weight = other_ori_nan ? 0.1 : 1;
            Eigen::Quaterniond hypothesis = transform * other_orientation;

            hypotheses.emplace_back(weight, hypothesis);
        }

        // for now just make a dumb weighted average
        double weight_sum = 0;
        Eigen::Vector4d vec_sum;
        for (const auto &h : hypotheses)
        {
            weight_sum += h.first;
            vec_sum += h.first * h.second.coeffs();
        }

        node_pose.orientation.coeffs() = weight_sum > 0 ? vec_sum : DOWN_ORIENTED_NORTH.coeffs();
        node_pose.orientation.normalize();
    }
}

struct PointsDownwardsPrior
{
    template <typename T> bool operator()(const T *rotation1, T *residuals) const
    {
        using QuaterionT = Eigen::Quaternion<T>;
        using Vector3T = Eigen::Matrix<T, 3, 1>;
        using QuaterionTCM = Eigen::Map<const QuaterionT>;

        const QuaterionTCM rotation_em(rotation1);

        const Eigen::Vector3d cam_center(0, 0, 1);
        const Eigen::Vector3d down(0, 0, -1);

        Vector3T rotated_cam_center = rotation_em * cam_center.cast<T>();

        residuals[0] = acos(T(0.99999) * rotated_cam_center.dot(down.cast<T>()));
        return true;
    }
};

// cost functions for rotations relative to positions
struct DecomposedRotationCost
{
    DecomposedRotationCost(const camera_relations &relations, const Eigen::Vector3d *translation1,
                           const Eigen::Vector3d *translation2)
        : _relations(relations), _translation1(translation1), _translation2(translation2)
    {
    }

    template <typename T> bool operator()(const T *rotation1, const T *rotation2, T *residuals) const
    {
        using QuaterionT = Eigen::Quaternion<T>;
        using Vector3T = Eigen::Matrix<T, 3, 1>;
        using QuaterionTCM = Eigen::Map<const QuaterionT>;

        const QuaterionTCM rotation1_em(rotation1);
        const QuaterionTCM rotation2_em(rotation2);

        // angle from camera1 -> camera2
        const Vector3T rotated_translation2_1 = rotation1_em.inverse() * (*_translation2 - *_translation1).cast<T>();
        residuals[0] = acos(T(0.99999) * rotated_translation2_1.dot(_relations.relative_translation) /
                            sqrt(rotated_translation2_1.squaredNorm() * _relations.relative_translation.squaredNorm()));

        // angle from camera2 -> camera1
        const Vector3T rotated_translation1_2 =
            rotation2_em.inverse() * (_relations.relative_rotation * (*_translation1 - *_translation2)).cast<T>();
        residuals[1] = acos(T(0.99999) * rotated_translation1_2.dot(-_relations.relative_translation) /
                            sqrt(rotated_translation1_2.squaredNorm() * _relations.relative_translation.squaredNorm()));

        // relative orientation of camera1 and camera2
        const QuaterionT rotation2_1 = rotation1_em.inverse() * rotation2_em;
        residuals[2] = Eigen::AngleAxis<T>(_relations.relative_rotation.cast<T>() * rotation2_1).angle();

        return true;
    }

  private:
    const camera_relations &_relations;
    const Eigen::Vector3d *_translation1, *_translation2;
};

void relaxDecompositions(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                         const std::unordered_set<size_t> &edges_to_optimize)
{
    ceres::Problem::Options problemOptions;
    problemOptions.cost_function_ownership = ceres::TAKE_OWNERSHIP;
    problemOptions.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problemOptions.local_parameterization_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;

    ceres::EigenQuaternionParameterization quat_parameterization;
    ceres::HuberLoss huber_loss(M_PI_2);

    ceres::Problem problem(problemOptions);

    std::unordered_map<size_t, NodePose *> nodes_to_optimize;
    nodes_to_optimize.reserve(nodes.size());

    for (NodePose &n : nodes)
    {
        n.orientation.normalize();
        nodes_to_optimize.emplace(n.node_id, &n);
    }

    std::unordered_set<size_t> edges_used, external_nodes_used;
    for (NodePose &n : nodes)
    {
        const auto *node = graph.getNode(n.node_id);
        if (node == nullptr)
        {
            spdlog::error("Null node referenced from optimization list");
            continue;
        }
        const auto &edgesIds = node->getEdges();
        for (size_t edge_id : edgesIds)
        {
            // skip edges not whitelisted
            if (edges_to_optimize.find(edge_id) == edges_to_optimize.end())
            {
                continue;
            }

            const auto *edge = graph.getEdge(edge_id);
            if (edge == nullptr)
            {
                spdlog::error("Null edge referenced from node");
                continue;
            }

            // skip unitialized edges
            if (edge->payload.relative_rotation.coeffs().hasNaN() || edge->payload.relative_translation.hasNaN())
            {
                continue;
            }

            bool n_is_source = edge->getSource() == n.node_id;

            size_t other_id = n_is_source ? edge->getDest() : edge->getSource();

            bool other_also_optimized = nodes_to_optimize.find(other_id) != nodes_to_optimize.end();

            Eigen::Vector3d *n_loc_ptr = &n.position, *other_loc_ptr;
            Eigen::Quaterniond *n_rot_ptr = &n.orientation, *other_rot_ptr;

            if (other_also_optimized)
            {
                NodePose *other = nodes_to_optimize.find(other_id)->second;
                other_loc_ptr = &other->position;
                other_rot_ptr = &other->orientation;
            }
            else
            {
                const auto *other = graph.getNode(other_id);
                if (other == nullptr)
                {
                    spdlog::error("Null node referenced from edge list");
                    continue;
                }

                if (other->payload.orientation.coeffs().hasNaN() || other->payload.position.hasNaN())
                {
                    continue;
                }

                // rather set the parameter block const once we've added it
                // otherwise we'd need a combinatorial list of const/mutable cost functions
                // although that would technically lead to faster jacobian evaluation
                other_loc_ptr = const_cast<Eigen::Vector3d *>(&other->payload.position);
                other_rot_ptr = const_cast<Eigen::Quaterniond *>(&other->payload.orientation);
            }

            Eigen::Vector3d *source_loc_ptr, *dest_loc_ptr;
            Eigen::Quaterniond *source_rot_ptr, *dest_rot_ptr;

            if (n_is_source)
            {
                source_loc_ptr = n_loc_ptr;
                source_rot_ptr = n_rot_ptr;
                dest_loc_ptr = other_loc_ptr;
                dest_rot_ptr = other_rot_ptr;
            }
            else
            {
                source_loc_ptr = other_loc_ptr;
                source_rot_ptr = other_rot_ptr;
                dest_loc_ptr = n_loc_ptr;
                dest_rot_ptr = n_rot_ptr;
            }

            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<DecomposedRotationCost, 3, 4, 4>(
                                         new DecomposedRotationCost(edge->payload, source_loc_ptr, dest_loc_ptr)),
                                     &huber_loss, source_rot_ptr->coeffs().data(), dest_rot_ptr->coeffs().data());

            if (!other_also_optimized)
            {
                problem.SetParameterBlockConstant(other_rot_ptr->coeffs().data());
            }

            edges_used.emplace(edge_id);
            if (!other_also_optimized)
            {
                external_nodes_used.emplace(other_id);
            }
        }
    }

    for (NodePose &n : nodes)
    {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<PointsDownwardsPrior, 1, 4>(new PointsDownwardsPrior()), nullptr,
            n.orientation.coeffs().data());
        problem.SetParameterization(n.orientation.coeffs().data(), &quat_parameterization);
    }

    ceres::Solver::Options solverOptions;
    solverOptions.num_threads = 1;
    solverOptions.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    solverOptions.max_num_iterations = 150;

    spdlog::debug("Start rotation relax: {} active nodes, {} edges, {} inactive nodes", nodes.size(), edges_used.size(),
                  external_nodes_used.size());

    ceres::Solver::Summary summary;
    ceres::Solver solver;
    solver.Solve(solverOptions, &problem, &summary);
    spdlog::debug(summary.BriefReport());

    for (NodePose &n : nodes)
    {
        n.orientation.normalize();
    }
}
} // namespace opencalibration
