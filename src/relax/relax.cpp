#include <opencalibration/relax/relax.hpp>

#include <opencalibration/distort/distort_keypoints.hpp>

#include <ceres/ceres.h>
#include <spdlog/spdlog.h>

#include <unordered_set>

namespace
{
using namespace opencalibration;

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

        const Eigen::Vector3d distance = *_translation2 - *_translation1;
        if (distance.squaredNorm() > 1e-9)
        {
            // angle from camera1 -> camera2
            const Vector3T rotated_translation2_1 = rotation1_em.inverse() * distance.cast<T>();
            residuals[0] =
                acos(T(0.99999) * rotated_translation2_1.dot(_relations.relative_translation) /
                     sqrt(rotated_translation2_1.squaredNorm() * _relations.relative_translation.squaredNorm()));

            // angle from camera2 -> camera1
            const Vector3T rotated_translation1_2 =
                rotation2_em.inverse() * (_relations.relative_rotation * -distance).cast<T>();
            residuals[1] =
                acos(T(0.99999) * rotated_translation1_2.dot(-_relations.relative_translation) /
                     sqrt(rotated_translation1_2.squaredNorm() * _relations.relative_translation.squaredNorm()));
        }
        else
        {
            residuals[0] = residuals[1] = T(0);
        }
        // relative orientation of camera1 and camera2
        const QuaterionT rotation2_1 = rotation1_em.inverse() * rotation2_em;
        residuals[2] = Eigen::AngleAxis<T>(_relations.relative_rotation.cast<T>() * rotation2_1).angle();

        return true;
    }

  private:
    const camera_relations &_relations;
    const Eigen::Vector3d *_translation1, *_translation2;
};

struct OptimizationPackage
{
    const camera_relations *relations;

    struct PoseOpt
    {
        Eigen::Vector3d *loc_ptr;
        Eigen::Quaterniond *rot_ptr;
        bool optimize = true;
        size_t node_id;
    } source, dest;
};

struct RelaxProblem
{
    ceres::Problem::Options problemOptions;
    ceres::EigenQuaternionParameterization quat_parameterization;
    ceres::HuberLoss huber_loss;
    std::unique_ptr<ceres::Problem> problem;

    ceres::Solver::Options solverOptions;
    ceres::Solver::Summary summary;
    ceres::Solver solver;

    std::unordered_map<size_t, NodePose *> nodes_to_optimize;
    std::unordered_set<size_t> edges_used, constant_nodes;

    RelaxProblem() : huber_loss(M_PI_2)
    {
        problemOptions.cost_function_ownership = ceres::TAKE_OWNERSHIP;
        problemOptions.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        problemOptions.local_parameterization_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        problem.reset(new ceres::Problem(problemOptions));

        solverOptions.num_threads = 1;
        solverOptions.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        solverOptions.max_num_iterations = 150;
    }

    void initialize(std::vector<NodePose> &nodes)
    {
        nodes_to_optimize.reserve(nodes.size());

        for (NodePose &n : nodes)
        {
            n.orientation.normalize();
            nodes_to_optimize.emplace(n.node_id, &n);
        }
    }

    bool shouldOptimizeEdge(const std::unordered_set<size_t> &edges_to_optimize, size_t edge_id,
                            const MeasurementGraph::Edge &edge)
    {
        // skip edges not whitelisted
        if (edges_to_optimize.find(edge_id) == edges_to_optimize.end())
        {
            return false;
        }

        // skip edges already used in this optimization problem
        if (edges_used.find(edge_id) != edges_used.end())
        {
            return false;
        }

        // skip unitialized edges
        if (edge.payload.relative_rotation.coeffs().hasNaN() || edge.payload.relative_translation.hasNaN())
        {
            return false;
        }

        return true;
    }

    OptimizationPackage::PoseOpt nodeid2poseopt(const MeasurementGraph &graph, size_t node_id)
    {
        OptimizationPackage::PoseOpt po;
        po.node_id = node_id;
        auto opt_iter = nodes_to_optimize.find(node_id);
        if (opt_iter != nodes_to_optimize.end())
        {
            NodePose *other = opt_iter->second;
            po.loc_ptr = &other->position;
            po.rot_ptr = &other->orientation;
            po.optimize = true;
        }
        else
        {
            const MeasurementGraph::Node *other = graph.getNode(node_id);
            if (other == nullptr)
            {
                spdlog::error("Null node referenced from edge list");
                po.loc_ptr = nullptr;
                po.rot_ptr = nullptr;
                po.optimize = false;
            }
            else if (other->payload.orientation.coeffs().hasNaN() || other->payload.position.hasNaN())
            {
                po.loc_ptr = nullptr;
                po.rot_ptr = nullptr;
                po.optimize = false;
            }
            else
            {
                po.loc_ptr = const_cast<Eigen::Vector3d *>(&other->payload.position);
                po.rot_ptr = const_cast<Eigen::Quaterniond *>(&other->payload.orientation);
                po.optimize = false;
            }
        }
        return po;
    }

    void addRelationCost(const MeasurementGraph &graph, size_t edge_id, const MeasurementGraph::Edge &edge)
    {
        OptimizationPackage pkg;
        pkg.relations = &edge.payload;

        pkg.source = nodeid2poseopt(graph, edge.getSource());
        if (pkg.source.loc_ptr == nullptr)
            return;

        pkg.dest = nodeid2poseopt(graph, edge.getDest());
        if (pkg.dest.loc_ptr == nullptr)
            return;

        // skip edges that trigger a singularity in the math
        if ((*pkg.source.loc_ptr - *pkg.dest.loc_ptr).squaredNorm() < 1e-9)
            return;

        using CostFunction = ceres::AutoDiffCostFunction<DecomposedRotationCost, 3, 4, 4>;
        std::unique_ptr<CostFunction> func(
            new CostFunction(new DecomposedRotationCost(*pkg.relations, pkg.source.loc_ptr, pkg.dest.loc_ptr)));

        // test-evaluate the cost function to make sure no weird data gets into the relaxation
        Eigen::Matrix<double, 4, 3> jac[2];
        jac[0].setConstant(NAN);
        jac[1].setConstant(NAN);
        double *jacdata[2] = {jac[0].data(), jac[1].data()};
        Eigen::Vector3d res;
        res.setConstant(NAN);
        double *datas[2] = {pkg.source.rot_ptr->coeffs().data(), pkg.dest.rot_ptr->coeffs().data()};
        bool success = func->Evaluate(datas, res.data(), jacdata);

        if (!success || !res.allFinite() || !jac[0].allFinite() || !jac[1].allFinite())
        {
            std::stringstream ss;
            ss << std::endl;
            ss << "source pos: " << pkg.source.loc_ptr->transpose()
               << "  rot: " << pkg.source.rot_ptr->coeffs().transpose() << std::endl;
            ss << "dest   pos: " << pkg.dest.loc_ptr->transpose() << "  rot: " << pkg.dest.rot_ptr->coeffs().transpose()
               << std::endl;
            ss << "res : " << res.transpose() << std::endl;
            ss << "jac[0]: " << std::endl << jac[0] << std::endl;
            ss << "jac[1]: " << std::endl << jac[1] << std::endl;
            spdlog::warn("Bad camera relation prevented from entering minimization: edge {} more info: {}", edge_id,
                         ss.str());
            return;
        }

        problem->AddResidualBlock(func.release(), &huber_loss, datas[0], datas[1]);

        if (!pkg.source.optimize)
        {
            constant_nodes.emplace(pkg.source.node_id);
        }
        if (!pkg.dest.optimize)
        {
            constant_nodes.emplace(pkg.dest.node_id);
        }

        edges_used.emplace(edge_id);
    }

    void addDownwardsPrior(std::vector<NodePose> &nodes)
    {
        for (NodePose &n : nodes)
        {
            problem->AddResidualBlock(
                new ceres::AutoDiffCostFunction<PointsDownwardsPrior, 1, 4>(new PointsDownwardsPrior()), nullptr,
                n.orientation.coeffs().data());
            problem->SetParameterization(n.orientation.coeffs().data(), &quat_parameterization);
        }
    }

    void setConstantBlocks(const MeasurementGraph &graph)
    {

        for (size_t const_node_id : constant_nodes)
        {
            problem->SetParameterBlockConstant(
                const_cast<double *>(graph.getNode(const_node_id)->payload.orientation.coeffs().data()));
        }
    }

    void solve(std::vector<NodePose> &nodes)
    {
        spdlog::debug("Start rotation relax: {} active nodes, {} edges, {} inactive nodes", nodes.size(),
                      edges_used.size(), constant_nodes.size());

        solver.Solve(solverOptions, problem.get(), &summary);
        spdlog::debug(summary.BriefReport());

        for (NodePose &n : nodes)
        {
            n.orientation.normalize();
        }
    }
};

} // namespace

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

void relaxDecompositions(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                         const std::unordered_set<size_t> &edges_to_optimize)
{
    RelaxProblem rp;
    rp.initialize(nodes);

    for (auto iter = graph.edgebegin(); iter != graph.edgeend(); ++iter)
    {
        size_t edge_id = iter->first;
        const MeasurementGraph::Edge &edge = iter->second;

        if (rp.shouldOptimizeEdge(edges_to_optimize, edge_id, edge))
        {
            rp.addRelationCost(graph, edge_id, edge);
        }
    }

    rp.addDownwardsPrior(nodes);
    rp.setConstantBlocks(graph);

    rp.solve(nodes);
}
} // namespace opencalibration
