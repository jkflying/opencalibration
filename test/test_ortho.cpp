#include <jk/KDTree.h>
#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/relax/relax.hpp>
#include <opencalibration/relax/relax_cost_function.hpp>
#include <opencalibration/relax/relax_problem.hpp>
#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_pose.hpp>
#include <opencalibration/types/point_cloud.hpp>
#include <opencalibration/ortho/ortho.hpp>
#include <opencalibration/surface/expand_mesh.hpp>

#include <gtest/gtest.h>

#include <chrono>

using namespace opencalibration;
using namespace opencalibration::orthomosaic;
using namespace std::chrono_literals;

struct ortho : public ::testing::Test
{
    size_t id[3];
    MeasurementGraph graph;
    std::vector<NodePose> nodePoses;
    std::unordered_map<size_t, CameraModel> cam_models;
    std::shared_ptr<CameraModel> model;
    Eigen::Quaterniond ground_ori[3];
    Eigen::Vector3d ground_pos[3];
    size_t edge_id[3];

    void init_cameras()
    {
        auto down = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX());
        ground_ori[0] = Eigen::Quaterniond(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ()) * down);
        ground_ori[1] = Eigen::Quaterniond(Eigen::AngleAxisd(-0.3, Eigen::Vector3d::UnitY()) * down);
        ground_ori[2] = Eigen::Quaterniond(Eigen::AngleAxisd(-0.3, Eigen::Vector3d::UnitX()) * down);
        ground_pos[0] = Eigen::Vector3d(9, 9, 9);
        ground_pos[1] = Eigen::Vector3d(11, 9, 9);
        ground_pos[2] = Eigen::Vector3d(11, 11, 9);

        model = std::make_shared<CameraModel>();
        model->focal_length_pixels = 600;
        model->principle_point << 400, 300;
        model->pixels_cols = 800;
        model->pixels_rows = 600;
        model->projection_type = opencalibration::ProjectionType::PLANAR;
        model->id = 42;

        cam_models[model->id] = *model;

        for (int i = 0; i < 3; i++)
        {
            image img;
            img.orientation = ground_ori[i];
            img.position = ground_pos[i];
            img.model = model;
            img.metadata.camera_info.height_px = model->pixels_rows;
            img.metadata.camera_info.width_px = model->pixels_cols;
            img.metadata.camera_info.focal_length_px = model->focal_length_pixels;
            img.thumbnail = RGBRaster(100, 100, 3);
            img.thumbnail.layers[0].band = Band::RED;
            img.thumbnail.layers[1].band = Band::GREEN;
            img.thumbnail.layers[2].band = Band::BLUE;
            for (int j = 0; j < 3; j++)
            {
                img.thumbnail.layers[j].pixels.fill(i * 3 + j);
            }
            id[i] = graph.addNode(std::move(img));
            nodePoses.emplace_back(NodePose{id[i], ground_ori[i], ground_pos[i]});
        }
    }

    point_cloud generate_planar_points()
    {
        point_cloud vec3d;
        vec3d.reserve(100);
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                vec3d.emplace_back(i + 5, j + 5, -10 + 1e-3 * i + 1e-2 * j);
            }
        }
        return vec3d;
    }

    point_cloud generate_3d_points()
    {
        point_cloud vec3d;
        vec3d.reserve(100);
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                vec3d.emplace_back(i + 5, j + 5, -10 + (i + j) % 2);
            }
        }
        return vec3d;
    }
};


TEST_F(ortho, measurement_3_images_points)
{
    // GIVEN: a graph with 3 images and a 3d point based surface model
    init_cameras();

    surface_model points_surface;
    points_surface.cloud.push_back(generate_planar_points());

    point_cloud camera_locations; // TODO: get camera locations
    for (const auto& nodePose : nodePoses)
    {
        camera_locations.push_back(nodePose.position);
    }
    surface_model mesh_surface;
    mesh_surface.mesh = rebuildMesh(camera_locations, {points_surface});


    // WHEN: we generate an orthomosaic
    OrthoMosaic result = generateOrthomosaic({mesh_surface}, graph);

    // THEN: it should have the right colours in the right locations

}
/*
TEST_F(ortho_, measurement_3_images_plane)
{
    // GIVEN: a graph, 3 images with edges between them all, then with their rotation disturbed
    init_cameras();
    add_point_measurements(generate_planar_points());
    add_ori_noise({-0.1, 0.1, 0.1});

    // WHEN: we relax them with relative orientation
    std::unordered_set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    relax(graph, np, cam_models, edges, {Option::ORIENTATION, Option::GROUND_PLANE}, {});
    // and again to re-init the inliers
    relax(graph, np, cam_models, edges, {Option::ORIENTATION, Option::GROUND_PLANE}, {});

    // THEN: it should put them back into the original orientation
    for (int i = 0; i < 3; i++)
        EXPECT_LT(Eigen::AngleAxisd(np[i].orientation.inverse() * ground_ori[i]).angle(), 1e-3)
            << i << ": " << np[i].orientation.coeffs().transpose() << std::endl
            << "g: " << ground_ori[i].coeffs().transpose();
}

TEST_F(ortho_, measurement_3_images_mesh_radial)
{
    // GIVEN: a graph, 3 images with edges between them all, then with their rotation disturbed
    init_cameras();
    cam_models[model->id].radial_distortion << 0.1, -0.1, 0.1;

    for (int i = 0; i < 10; i++)
        // a few times...
        relax(graph, np, cam_models, edges, options, {});

    // THEN: it should put them back into the original orientation
    for (int i = 0; i < 3; i++)
    {
        EXPECT_LT(Eigen::AngleAxisd(np[i].orientation.inverse() * ground_ori[i]).angle(), 1e-3)
            << i << ": " << np[i].orientation.coeffs().transpose() << std::endl
            << "g: " << ground_ori[i].coeffs().transpose();
    }

    EXPECT_LT((cam_models[model->id].radial_distortion - model->radial_distortion).norm(), 1e-4)
        << cam_models[model->id].radial_distortion;
    EXPECT_NEAR(cam_models[model->id].focal_length_pixels, model->focal_length_pixels, 1e-9);
}
*/
