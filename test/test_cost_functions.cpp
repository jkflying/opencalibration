#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/relax/autodiff_cost_function.hpp>
#include <opencalibration/relax/padded_autodiff_cost_function.hpp>
#include <opencalibration/relax/relax_cost_function.hpp>

#include <ceres/ceres.h>
#include <gtest/gtest.h>

using namespace opencalibration;

TEST(cost_functions, difference_cost)
{
    DifferenceCost cost(2.0);
    double v1 = 5.0, v2 = 3.0;
    double residual = 0;
    EXPECT_TRUE(cost(&v1, &v2, &residual));
    EXPECT_DOUBLE_EQ(4.0, residual);
}

TEST(cost_functions, difference_cost_equal)
{
    DifferenceCost cost(1.0);
    double v = 7.0;
    double residual = 0;
    EXPECT_TRUE(cost(&v, &v, &residual));
    EXPECT_DOUBLE_EQ(0.0, residual);
}

TEST(cost_functions, value_prior)
{
    // GIVEN: a prior pulling towards 3 with weight 2
    ValuePrior cost(3.0, 2.0);
    double v = 5.0;
    double residual = 0;

    // WHEN: evaluated at 5
    EXPECT_TRUE(cost(&v, &residual));

    // THEN: residual is weighted offset from target
    EXPECT_DOUBLE_EQ(4.0, residual);
}

TEST(cost_functions, distortion_monotonicity_zero_distortion)
{
    DistortionMonotonicityCost cost(1.0, 1.0);
    double radial[3] = {0, 0, 0};
    double residuals[10] = {};
    EXPECT_TRUE(cost(radial, residuals));
    for (int i = 0; i < 10; i++)
    {
        EXPECT_DOUBLE_EQ(0.0, residuals[i]) << "residual " << i;
    }
}

TEST(cost_functions, distortion_monotonicity_negative_k1)
{
    DistortionMonotonicityCost cost(1.0, 1.0);
    double radial[3] = {-10.0, 0, 0};
    double residuals[10] = {};
    EXPECT_TRUE(cost(radial, residuals));

    bool any_nonzero = false;
    for (int i = 0; i < 10; i++)
    {
        EXPECT_GE(residuals[i], 0.0);
        if (residuals[i] > 0)
            any_nonzero = true;
    }
    EXPECT_TRUE(any_nonzero);
}

TEST(cost_functions, adjacent_triangle_coplanar)
{
    Eigen::Vector2d A(0, 0), B(1, 0), C(0.5, 1), D(0.5, -1);
    AdjacentTriangleNormalCost cost(A, B, C, D, 1.0);

    double zA = 0, zB = 0, zC = 0, zD = 0;
    double residual = 0;
    EXPECT_TRUE(cost(&zA, &zB, &zC, &zD, &residual));
    EXPECT_NEAR(0.0, residual, 1e-5);
}

TEST(cost_functions, adjacent_triangle_noncoplanar)
{
    Eigen::Vector2d A(0, 0), B(1, 0), C(0.5, 1), D(0.5, -1);
    AdjacentTriangleNormalCost cost(A, B, C, D, 1.0);

    double zA = 0, zB = 0, zC = 0, zD = 5.0;
    double residual = 0;
    EXPECT_TRUE(cost(&zA, &zB, &zC, &zD, &residual));
    EXPECT_GT(std::abs(residual), 0.1);
}

TEST(cost_functions, robust_centroid_no_outlier)
{
    Eigen::Vector3d points[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    Eigen::Vector3d result = robustCentroid(points, 3, 10.0);
    Eigen::Vector3d expected(1.0 / 3, 1.0 / 3, 1.0 / 3);
    EXPECT_NEAR(expected.x(), result.x(), 1e-6);
    EXPECT_NEAR(expected.y(), result.y(), 1e-6);
    EXPECT_NEAR(expected.z(), result.z(), 1e-6);
}

TEST(cost_functions, robust_centroid_with_outlier)
{
    Eigen::Vector3d points[4] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {100, 100, 100}};
    Eigen::Vector3d result = robustCentroid(points, 4, 1.0);

    Eigen::Vector3d inlier_center(1.0 / 3, 1.0 / 3, 0);
    EXPECT_LT((result - inlier_center).norm(), 1.0);
}

TEST(cost_functions, angle_between_unit_vectors)
{
    Eigen::Vector3d a(1, 0, 0), b(0, 1, 0);
    double angle = angleBetweenUnitVectors(a, b);
    EXPECT_NEAR(M_PI / 2, angle, 1e-10);

    double zero_angle = angleBetweenUnitVectors(a, a);
    EXPECT_NEAR(0.0, zero_angle, 1e-5);
}

TEST(cost_functions, signed_dihedral_angle)
{
    const Eigen::Vector3d A(0, 0, 0), B(2, 0, 0), C(1, 1, 0);
    EXPECT_NEAR(0.0, signedDihedralAngle<double>(A, B, C, Eigen::Vector3d(1, -1, 0)), 1e-12);
    EXPECT_NEAR(-M_PI / 4, signedDihedralAngle<double>(A, B, C, Eigen::Vector3d(1, -1, 1)), 1e-12);
    EXPECT_NEAR(M_PI / 4, signedDihedralAngle<double>(A, B, C, Eigen::Vector3d(1, -1, -1)), 1e-12);
    EXPECT_NEAR(M_PI / 4, signedDihedralAngle<double>(A, B, C, Eigen::Vector3d(1, -3, -3)), 1e-12);
}

TEST(cost_functions, plane_intersection_focal_radial_residual_not_reduced_by_larger_focal)
{
    InverseDifferentiableCameraModel<double> model;
    model.pixels_cols = 4000;
    model.pixels_rows = 3000;
    model.focal_length_pixels = 3000;
    model.principle_point << 2000, 1500;

    const Eigen::Quaterniond down(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
    const Eigen::Vector3d cam0(0, 0, 100), cam1(20, 0, 100);
    std::array<double, 7> pose0{down.x(), down.y(), down.z(), down.w(), cam0.x(), cam0.y(), cam0.z()};
    std::array<double, 7> pose1{down.x(), down.y(), down.z(), down.w(), cam1.x(), cam1.y(), cam1.z()};
    const double z = 0;

    const Eigen::Vector2d pixel_error(5, 0);
    const auto residualNormForPixelError = [&](double focal, const Eigen::Vector3d &ground) {
        InverseDifferentiableCameraModel<double> scene_model = model;
        scene_model.focal_length_pixels = focal;
        const Eigen::Vector2d pixel0 = image_from_3d(down.inverse() * (ground - cam0), scene_model);
        const Eigen::Vector2d pixel1 = image_from_3d(down.inverse() * (ground - cam1), scene_model) + pixel_error;
        PlaneIntersectionAngleCost_OrientationFocalRadial_SharedModel cost(
            pixel0, pixel1, Eigen::Vector2d(-100, -100), Eigen::Vector2d(100, -100), Eigen::Vector2d(0, 100), model);
        std::array<double, 4> residuals;
        EXPECT_TRUE(cost(pose0.data(), pose1.data(), &z, &z, &z, &focal, model.principle_point.data(),
                         model.radial_distortion.data(), residuals.data()));
        return Eigen::Map<Eigen::Matrix<double, 4, 1>>(residuals.data()).norm();
    };

    const Eigen::Vector3d center(10, 5, 0), off_axis(150, 80, 0);
    EXPECT_GT(residualNormForPixelError(3000, center), 0);
    EXPECT_NEAR(residualNormForPixelError(6000, center) / residualNormForPixelError(3000, center), 1.0, 0.05);
    EXPECT_NEAR(residualNormForPixelError(6000, off_axis) / residualNormForPixelError(3000, off_axis), 1.0, 0.05);
    EXPECT_NEAR(residualNormForPixelError(3000, off_axis) / residualNormForPixelError(3000, center), 1.0, 0.1);
}

TEST(cost_functions, plane_intersection_focal_radial_residual_continuous_at_wide_angles)
{
    InverseDifferentiableCameraModel<double> model;
    model.focal_length_pixels = 3000;
    model.principle_point << 2000, 1500;

    const Eigen::Quaterniond down(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
    const Eigen::Vector3d cam0(0, 0, 100), cam1(20, 0, 100);
    std::array<double, 7> pose0{down.x(), down.y(), down.z(), down.w(), cam0.x(), cam0.y(), cam0.z()};
    std::array<double, 7> pose1{down.x(), down.y(), down.z(), down.w(), cam1.x(), cam1.y(), cam1.z()};
    const double z = 0;

    const auto residualNorm = [&](const Eigen::Vector3d &ground) {
        const Eigen::Vector2d pixel0 = image_from_3d(down.inverse() * (ground - cam0), model);
        const Eigen::Vector2d pixel1 = image_from_3d(down.inverse() * (ground - cam1), model) + Eigen::Vector2d(5, 0);
        PlaneIntersectionAngleCost_OrientationFocalRadial_SharedModel cost(
            pixel0, pixel1, Eigen::Vector2d(-500, -500), Eigen::Vector2d(500, -500), Eigen::Vector2d(0, 500), model);
        std::array<double, 4> residuals;
        EXPECT_TRUE(cost(pose0.data(), pose1.data(), &z, &z, &z, &model.focal_length_pixels,
                         model.principle_point.data(), model.radial_distortion.data(), residuals.data()));
        return Eigen::Map<Eigen::Matrix<double, 4, 1>>(residuals.data()).norm();
    };

    // GIVEN: a ground point seen by camera 0 on either side of 60 degrees off axis
    const double sixty_degrees_x = 100 * std::tan(M_PI / 3);

    // WHEN: we evaluate the residual just either side
    const double inside = residualNorm({sixty_degrees_x - 0.1, 0, 0});
    const double outside = residualNorm({sixty_degrees_x + 0.1, 0, 0});

    // THEN: the residual barely changes
    EXPECT_NEAR(outside / inside, 1.0, 0.01);
}

TEST(cost_functions, triangulated_reprojection_zero_at_true_pose_for_any_terrain_height)
{
    const Eigen::Quaterniond down(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
    const Eigen::Vector3d cam0(0, 0, 100), cam1(20, 0, 100);
    std::array<double, 7> pose0{down.x(), down.y(), down.z(), down.w(), cam0.x(), cam0.y(), cam0.z()};
    std::array<double, 7> pose1{down.x(), down.y(), down.z(), down.w(), cam1.x(), cam1.y(), cam1.z()};

    const auto residualNorm = [&](const Eigen::Vector3d &ground, const std::array<double, 7> &p1) {
        TriangulatedReprojectionCost<2> cost({down.inverse() * (ground - cam0), down.inverse() * (ground - cam1)});
        std::array<double, 6> residuals;
        EXPECT_TRUE(cost(pose0.data(), p1.data(), residuals.data()));
        return Eigen::Map<Eigen::Matrix<double, 6, 1>>(residuals.data()).norm();
    };

    EXPECT_NEAR(residualNorm({10, 5, 0}, pose1), 0, 1e-9);
    EXPECT_NEAR(residualNorm({-30, 40, 45}, pose1), 0, 1e-9);

    const Eigen::Quaterniond tilted =
        down * Eigen::Quaterniond(Eigen::AngleAxisd(0.1 * M_PI / 180, Eigen::Vector3d::UnitX()));
    std::array<double, 7> tilted_pose1{tilted.x(), tilted.y(), tilted.z(), tilted.w(), cam1.x(), cam1.y(), cam1.z()};
    EXPECT_GT(residualNorm({10, 5, 0}, tilted_pose1), 1e-4);
}

template <int N> void checkTriangulatedReprojectionNView()
{
    const Eigen::Quaterniond down(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
    const std::array<Eigen::Vector3d, 5> all_cams{Eigen::Vector3d(0, 0, 100), Eigen::Vector3d(20, 0, 100),
                                                  Eigen::Vector3d(0, 20, 100), Eigen::Vector3d(20, 20, 95),
                                                  Eigen::Vector3d(10, -15, 105)};
    const Eigen::Vector3d ground(-30, 40, 45);

    std::array<Eigen::Vector3d, N> rays;
    std::array<std::array<double, 7>, N> poses;
    for (int i = 0; i < N; i++)
    {
        rays[i] = down.inverse() * (ground - all_cams[i]);
        poses[i] = {down.x(), down.y(), down.z(), down.w(), all_cams[i].x(), all_cams[i].y(), all_cams[i].z()};
    }
    TriangulatedReprojectionCost<N> cost(rays);

    const auto residualNorm = [&](const std::array<std::array<double, 7>, N> &p) {
        std::array<const double *, N> pose_ptrs;
        for (int i = 0; i < N; i++)
            pose_ptrs[i] = p[i].data();
        std::array<double, N * 3> residuals;
        EXPECT_TRUE(cost.eval(pose_ptrs.data(), residuals.data()));
        return Eigen::Map<Eigen::Matrix<double, N * 3, 1>>(residuals.data()).norm();
    };

    EXPECT_NEAR(residualNorm(poses), 0, 1e-9) << N;

    for (const Eigen::Vector3d axis : {Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY()})
    {
        auto tilted_poses = poses;
        const Eigen::Quaterniond tilted = down * Eigen::Quaterniond(Eigen::AngleAxisd(0.1 * M_PI / 180, axis));
        tilted_poses[N - 1] = {tilted.x(),          tilted.y(),          tilted.z(),         tilted.w(),
                               all_cams[N - 1].x(), all_cams[N - 1].y(), all_cams[N - 1].z()};
        EXPECT_GT(residualNorm(tilted_poses), 1e-4) << N << " " << axis.transpose();
    }
}

TEST(cost_functions, triangulated_reprojection_n_view_zero_at_true_pose_and_observes_rotation)
{
    checkTriangulatedReprojectionNView<3>();
    checkTriangulatedReprojectionNView<4>();
    checkTriangulatedReprojectionNView<5>();
}

TEST(cost_functions, triangulated_reprojection_evaluates_with_all_cameras_flipped)
{
    const Eigen::Quaterniond down(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
    const std::array<Eigen::Vector3d, 3> cams{Eigen::Vector3d(0, 0, 100), Eigen::Vector3d(20, 0, 100),
                                              Eigen::Vector3d(0, 20, 100)};
    const Eigen::Vector3d ground(5, 7, 0);

    std::array<Eigen::Vector3d, 3> rays;
    std::array<std::array<double, 7>, 3> poses;
    for (int i = 0; i < 3; i++)
    {
        rays[i] = down.inverse() * (ground - cams[i]);
        const Eigen::Vector3d to_ground = ground - cams[i];
        const Eigen::Quaterniond reversed =
            Eigen::AngleAxisd(M_PI, to_ground.cross(Eigen::Vector3d::UnitX()).normalized()) * down;
        poses[i] = {reversed.x(), reversed.y(), reversed.z(), reversed.w(), cams[i].x(), cams[i].y(), cams[i].z()};
    }
    TriangulatedReprojectionCost<3> cost(rays);
    const std::array<const double *, 3> pose_ptrs{poses[0].data(), poses[1].data(), poses[2].data()};
    std::array<double, 9> residuals;
    ASSERT_TRUE(cost.eval(pose_ptrs.data(), residuals.data()));
    EXPECT_TRUE((Eigen::Map<Eigen::Matrix<double, 9, 1>>(residuals.data()).allFinite()));
}

TEST(cost_functions, triangulated_reprojection_fixed_position_matches_full_pose)
{
    const Eigen::Quaterniond down(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
    const std::vector<Eigen::Vector3d> cams{Eigen::Vector3d(0, 0, 100), Eigen::Vector3d(20, 0, 100),
                                            Eigen::Vector3d(0, 20, 100)};
    const Eigen::Vector3d ground(5, 7, 0);

    const Eigen::Quaterniond outlier_orientation = down * Eigen::AngleAxisd(M_PI / 6, Eigen::Vector3d::UnitY());

    std::vector<Eigen::Vector3d> rays;
    std::array<std::array<double, 7>, 3> poses;
    for (int i = 0; i < 3; i++)
    {
        rays.push_back(down.inverse() * (ground - cams[i]));
        const Eigen::Quaterniond q = i == 0 ? outlier_orientation : down;
        poses[i] = {q.x(), q.y(), q.z(), q.w(), cams[i].x(), cams[i].y(), cams[i].z()};
    }
    std::unique_ptr<ceres::CostFunction> full(newAutoDiffTriangulatedReprojectionCost(rays));
    std::unique_ptr<ceres::CostFunction> fixed(newAutoDiffTriangulatedReprojectionCost_FixedPositions(rays, cams));
    EXPECT_EQ(full->parameter_block_sizes(), std::vector<int32_t>(3, 7));
    EXPECT_EQ(fixed->parameter_block_sizes(), std::vector<int32_t>(3, 4));

    const std::array<const double *, 3> pose_ptrs{poses[0].data(), poses[1].data(), poses[2].data()};
    Eigen::Matrix<double, 9, 1> full_res, fixed_res;
    ASSERT_TRUE(full->Evaluate(pose_ptrs.data(), full_res.data(), nullptr));
    ASSERT_TRUE(fixed->Evaluate(pose_ptrs.data(), fixed_res.data(), nullptr));
    EXPECT_GT(full_res.norm(), 0.1);
    EXPECT_LT((full_res - fixed_res).norm(), 1e-12);
}

TEST(cost_functions, triangulated_reprojection_unflips_camera)
{
    const Eigen::Quaterniond down(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
    const std::array<Eigen::Vector3d, 3> cams{Eigen::Vector3d(0, 0, 100), Eigen::Vector3d(20, 0, 100),
                                              Eigen::Vector3d(0, 20, 100)};

    for (const Eigen::Vector3d &axis : {Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0),
                                        Eigen::Vector3d(1, 1, 0).normalized(), Eigen::Vector3d(1, -2, 1).normalized()})
    {
        for (int deg = 90; deg <= 180; deg += 15)
        {
            std::array<std::array<double, 7>, 3> poses;
            for (int i = 0; i < 3; i++)
                poses[i] = {down.x(), down.y(), down.z(), down.w(), cams[i].x(), cams[i].y(), cams[i].z()};
            const Eigen::Quaterniond flipped = down * Eigen::AngleAxisd(deg * M_PI / 180, axis);
            poses[0] = {flipped.x(), flipped.y(), flipped.z(), flipped.w(), cams[0].x(), cams[0].y(), cams[0].z()};

            ceres::Problem::Options problem_options;
            problem_options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
            problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
            ceres::Problem problem(problem_options);
            ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::SubsetManifold> manifold(
                ceres::EigenQuaternionManifold{}, ceres::SubsetManifold(3, {0, 1, 2}));
            ceres::HuberLoss loss(0.2 * M_PI / 180);
            for (int x = -40; x <= 60; x += 20)
            {
                for (int y = -40; y <= 60; y += 20)
                {
                    const Eigen::Vector3d ground(x, y, (x * y) % 7);
                    std::array<Eigen::Vector3d, 3> rays;
                    for (int i = 0; i < 3; i++)
                        rays[i] = down.inverse() * (ground - cams[i]);
                    problem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<TriangulatedReprojectionCost<3>, 9, 7, 7, 7>(
                            new TriangulatedReprojectionCost<3>(rays)),
                        &loss, poses[0].data(), poses[1].data(), poses[2].data());
                }
            }
            problem.SetManifold(poses[0].data(), &manifold);
            problem.SetParameterBlockConstant(poses[1].data());
            problem.SetParameterBlockConstant(poses[2].data());

            ceres::Solver::Options solver_options;
            solver_options.max_num_iterations = 200;
            ceres::Solver::Summary summary;
            ceres::Solve(solver_options, &problem, &summary);

            const Eigen::Quaterniond result(poses[0][3], poses[0][0], poses[0][1], poses[0][2]);
            EXPECT_LT(result.angularDistance(down) * 180 / M_PI, 1e-3)
                << "flipped " << deg << " deg about " << axis.transpose();
        }
    }
}

TEST(cost_functions, triangulated_reprojection_residuals_scale_with_inverse_sigma)
{
    const Eigen::Quaterniond down(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
    const std::vector<Eigen::Vector3d> cams{Eigen::Vector3d(0, 0, 100), Eigen::Vector3d(20, 0, 100),
                                            Eigen::Vector3d(0, 20, 100)};
    const Eigen::Vector3d ground(5, 7, 0);

    std::vector<Eigen::Vector3d> rays;
    std::array<std::array<double, 7>, 3> poses;
    for (int i = 0; i < 3; i++)
    {
        rays.push_back(down.inverse() * (ground - cams[i]));
        const Eigen::Quaterniond q = down * Eigen::AngleAxisd(0.01 * (i + 1), Eigen::Vector3d::UnitY());
        poses[i] = {q.x(), q.y(), q.z(), q.w(), cams[i].x(), cams[i].y(), cams[i].z()};
    }
    std::unique_ptr<ceres::CostFunction> unit(newAutoDiffTriangulatedReprojectionCost(rays));
    std::unique_ptr<ceres::CostFunction> whitened(newAutoDiffTriangulatedReprojectionCost(rays, {2, 3, 5}));

    const std::array<const double *, 3> pose_ptrs{poses[0].data(), poses[1].data(), poses[2].data()};
    Eigen::Matrix<double, 9, 1> unit_res, whitened_res;
    ASSERT_TRUE(unit->Evaluate(pose_ptrs.data(), unit_res.data(), nullptr));
    ASSERT_TRUE(whitened->Evaluate(pose_ptrs.data(), whitened_res.data(), nullptr));
    EXPECT_GT(unit_res.norm(), 1e-3);
    const std::array<double, 3> inverse_sigmas{2, 3, 5};
    for (int i = 0; i < 3; i++)
        EXPECT_LT((whitened_res.segment<3>(3 * i) - inverse_sigmas[i] * unit_res.segment<3>(3 * i)).norm(), 1e-12);
}

TEST(cost_functions, pixel_from_projected_ray_jet_matches_finite_differences)
{
    // GIVEN: a distorted inverse camera model and a projected ray, with derivatives on focal, principal, k1 and the ray
    using JetT = ceres::Jet<double, 6>;
    InverseDifferentiableCameraModel<double> model;
    model.focal_length_pixels = 2800;
    model.principle_point = Eigen::Vector2d(2010, 1490);
    model.radial_distortion = Eigen::Vector3d(-0.08, 0.02, -0.005);
    model.tangential_distortion = Eigen::Vector2d(0.001, -0.0005);
    const Eigen::Vector2d projected_ray(0.55, -0.4);
    const Eigen::Vector2d initial_pixel = projected_ray * model.focal_length_pixels + model.principle_point;

    InverseDifferentiableCameraModel<JetT> jet_model = model.cast<JetT>();
    jet_model.focal_length_pixels.v[0] = 1;
    jet_model.principle_point[0].v[1] = 1;
    jet_model.principle_point[1].v[2] = 1;
    jet_model.radial_distortion[0].v[3] = 1;
    Eigen::Matrix<JetT, 2, 1> jet_ray = projected_ray.cast<JetT>();
    jet_ray[0].v[4] = 1;
    jet_ray[1].v[5] = 1;

    // WHEN: we solve for the pixel with the implicit function theorem derivative
    Eigen::Matrix<JetT, 2, 1> jet_pixel;
    ASSERT_TRUE(pixelFromProjectedRay(jet_ray, jet_model, initial_pixel, jet_pixel));

    // THEN: the value round-trips through the closed-form unprojection
    const Eigen::Vector2d pixel(jet_pixel[0].a, jet_pixel[1].a);
    EXPECT_LT((projectedRayFromPixel<double>(pixel, model) - projected_ray).norm(), 1e-10);

    // AND: the derivatives match central finite differences
    const auto solveWith = [&](int parameter, double delta) {
        InverseDifferentiableCameraModel<double> perturbed = model;
        Eigen::Vector2d ray = projected_ray;
        double *targets[6]{&perturbed.focal_length_pixels,
                           &perturbed.principle_point[0],
                           &perturbed.principle_point[1],
                           &perturbed.radial_distortion[0],
                           &ray[0],
                           &ray[1]};
        *targets[parameter] += delta;
        Eigen::Vector2d result;
        EXPECT_TRUE(pixelFromProjectedRay(ray, perturbed, initial_pixel, result));
        return result;
    };
    const double steps[6]{1e-3, 1e-3, 1e-3, 1e-7, 1e-7, 1e-7};
    for (int parameter = 0; parameter < 6; parameter++)
    {
        const Eigen::Vector2d numeric =
            (solveWith(parameter, steps[parameter]) - solveWith(parameter, -steps[parameter])) / (2 * steps[parameter]);
        for (int i = 0; i < 2; i++)
            EXPECT_NEAR(jet_pixel[i].v[parameter], numeric[i], 1e-4 * (1 + std::abs(numeric[i])))
                << "parameter " << parameter << " pixel axis " << i;
    }
}

namespace
{
struct MixedBlocksCost
{
    template <typename T> bool operator()(const T *a, const T *b, T *residuals) const
    {
        residuals[0] = a[0] * b[0] + a[2] * b[1];
        residuals[1] = a[1] * a[1] - b[0] * b[1];
        return true;
    }
};
} // namespace

TEST(cost_functions, padded_autodiff_matches_plain_autodiff)
{
    // GIVEN: parameter blocks of 3 and 2 doubles, whose Jet width of 5 needs padding to a whole SIMD packet
    using Plain = ceres::AutoDiffCostFunction<MixedBlocksCost, 2, 3, 2>;
    using Padded = PaddedAutoDiffCostFunction<MixedBlocksCost, 2, 3, 2>;
    static_assert(!std::is_same_v<Plain, Padded>);
    Plain plain(new MixedBlocksCost);
    Padded padded(new MixedBlocksCost);
    const double a[3] = {1.5, -2.0, 0.5};
    const double b[2] = {3.0, -1.0};
    const double *parameters[2] = {a, b};

    // WHEN: both are evaluated with jacobians
    double plain_residuals[2], padded_residuals[2];
    double plain_jacobian_a[6], plain_jacobian_b[4], padded_jacobian_a[6], padded_jacobian_b[4];
    double *plain_jacobians[2] = {plain_jacobian_a, plain_jacobian_b};
    double *padded_jacobians[2] = {padded_jacobian_a, padded_jacobian_b};
    ASSERT_TRUE(plain.Evaluate(parameters, plain_residuals, plain_jacobians));
    ASSERT_TRUE(padded.Evaluate(parameters, padded_residuals, padded_jacobians));

    // THEN: residuals and jacobians are identical, and residuals alone can also be evaluated
    for (int i = 0; i < 2; i++)
        EXPECT_DOUBLE_EQ(plain_residuals[i], padded_residuals[i]);
    for (int i = 0; i < 6; i++)
        EXPECT_DOUBLE_EQ(plain_jacobian_a[i], padded_jacobian_a[i]);
    for (int i = 0; i < 4; i++)
        EXPECT_DOUBLE_EQ(plain_jacobian_b[i], padded_jacobian_b[i]);
    double residuals_only[2];
    ASSERT_TRUE(padded.Evaluate(parameters, residuals_only, nullptr));
    EXPECT_DOUBLE_EQ(plain_residuals[0], residuals_only[0]);
}

TEST(cost_functions, autodiff_is_not_padded_when_jets_already_fill_packets_or_are_narrow)
{
    // GIVEN/WHEN: parameter widths of 4 (already a whole packet) and 2 (too narrow to vectorize)
    // THEN: the plain ceres cost function is used
    static_assert(std::is_same_v<PaddedAutoDiffCostFunction<MixedBlocksCost, 2, 3, 1>,
                                 ceres::AutoDiffCostFunction<MixedBlocksCost, 2, 3, 1>>);
    static_assert(std::is_same_v<PaddedAutoDiffCostFunction<MixedBlocksCost, 2, 1, 1>,
                                 ceres::AutoDiffCostFunction<MixedBlocksCost, 2, 1, 1>>);
}
