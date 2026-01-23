#include <jk/KDTree.h>
#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/geo_coord/geo_coord.hpp>
#include <opencalibration/ortho/image_cache.hpp>
#include <opencalibration/ortho/ortho.hpp>
#include <opencalibration/relax/relax.hpp>
#include <opencalibration/relax/relax_cost_function.hpp>
#include <opencalibration/relax/relax_problem.hpp>
#include <opencalibration/surface/expand_mesh.hpp>
#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_pose.hpp>
#include <opencalibration/types/point_cloud.hpp>

#include <gdal/gdal_priv.h>
#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <filesystem>

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

TEST_F(ortho, calculate_bounds_cloud)
{
    // GIVEN: a surface model with point clouds
    surface_model s;
    point_cloud cloud;
    cloud.emplace_back(0, 0, 10);
    cloud.emplace_back(10, 20, 30);
    s.cloud.push_back(cloud);

    // WHEN: we calculate the bounds
    auto bounds = calculateBoundsAndMeanZ({s});

    // THEN: they should match the cloud
    EXPECT_DOUBLE_EQ(bounds.min_x, 0);
    EXPECT_DOUBLE_EQ(bounds.max_x, 10);
    EXPECT_DOUBLE_EQ(bounds.min_y, 0);
    EXPECT_DOUBLE_EQ(bounds.max_y, 20);
    EXPECT_DOUBLE_EQ(bounds.mean_surface_z, 20);
}

TEST_F(ortho, calculate_bounds_mesh)
{
    // GIVEN: a surface model with a mesh
    surface_model s;
    MeshNode n;
    n.location = {1, 2, 3};
    s.mesh.addNode(std::move(n));
    n.location = {5, 6, 7};
    s.mesh.addNode(std::move(n));

    // WHEN: we calculate the bounds
    auto bounds = calculateBoundsAndMeanZ({s});

    // THEN: they should match the mesh
    EXPECT_DOUBLE_EQ(bounds.min_x, 1);
    EXPECT_DOUBLE_EQ(bounds.max_x, 5);
    EXPECT_DOUBLE_EQ(bounds.min_y, 2);
    EXPECT_DOUBLE_EQ(bounds.max_y, 6);
    EXPECT_DOUBLE_EQ(bounds.mean_surface_z, 5);
}

TEST_F(ortho, calculate_gsd)
{
    // GIVEN: a graph with 1 image
    init_cameras();
    MeasurementGraph single_image_graph;
    image img = graph.getNode(id[0])->payload;
    single_image_graph.addNode(std::move(img));

    // h = 0.001
    // ray1 = {0,0,1}, ray2 = {0.001, 0, 1}
    // focal = 600
    // pixel1 = {400, 300}, pixel2 = {400 + 0.001*600, 300} = {400.6, 300}
    // dist = 0.6
    // arc_pixel = 0.001 / 0.6 = 1/600
    // thumb_scale = 100 / 600 = 1/6
    // thumb_arc_pixel = (1/600) / (1/6) = 6/600 = 0.01
    // camera_z = 9, surface_z = 0
    // elevation = 9
    // gsd = 9 * 0.01 = 0.09

    // WHEN: we calculate the GSD
    double gsd = calculateGSD(single_image_graph, {id[0]}, 0);

    // THEN: it should be 0.09
    EXPECT_NEAR(gsd, 0.09, 1e-7);
}

TEST_F(ortho, prepare_context)
{
    // GIVEN: a scene with images and surface
    init_cameras();
    surface_model s;
    point_cloud cloud;
    cloud.emplace_back(5, 5, -10);
    cloud.emplace_back(10, 10, -5);
    s.cloud.push_back(cloud);

    // WHEN: we prepare the orthomosaic context
    OrthoMosaicContext context = prepareOrthoMosaicContext({s}, graph);

    // THEN: it should have correct bounds
    EXPECT_DOUBLE_EQ(context.bounds.min_x, 5);
    EXPECT_DOUBLE_EQ(context.bounds.max_x, 10);
    EXPECT_DOUBLE_EQ(context.bounds.min_y, 5);
    EXPECT_DOUBLE_EQ(context.bounds.max_y, 10);
    EXPECT_DOUBLE_EQ(context.bounds.mean_surface_z, -7.5);

    // AND: it should have involved nodes
    EXPECT_EQ(context.involved_nodes.size(), 3);
    EXPECT_TRUE(context.involved_nodes.count(id[0]));
    EXPECT_TRUE(context.involved_nodes.count(id[1]));
    EXPECT_TRUE(context.involved_nodes.count(id[2]));

    // AND: it should have calculated GSD
    EXPECT_GT(context.gsd, 0);

    // AND: it should have built KDTree
    auto nearest = context.imageGPSLocations.searchKnn({10, 10, 9}, 1);
    EXPECT_EQ(nearest.size(), 1);

    // AND: it should have calculated mean camera z
    EXPECT_DOUBLE_EQ(context.mean_camera_z, 9);

    // AND: it should have calculated average camera elevation (9 - (-7.5) = 16.5)
    EXPECT_DOUBLE_EQ(context.average_camera_elevation, 16.5);
}

TEST_F(ortho, ray_trace_height)
{
    // GIVEN: a surface model with a mesh
    surface_model s;
    point_cloud camera_locations = {{0, 0, 10}, {10, 0, 10}};
    point_cloud cloud;
    cloud.emplace_back(5, 5, -10);
    cloud.emplace_back(10, 10, -5);
    cloud.emplace_back(5, 10, -7.5);
    s.cloud.push_back(cloud);
    s.mesh = rebuildMesh(camera_locations, {s});

    // WHEN: we ray-trace at a point
    double z = rayTraceHeight(7.5, 7.5, 10, {s});

    // THEN: it should return a valid height
    EXPECT_FALSE(std::isnan(z));
    EXPECT_LT(z, 0);   // Surface is below z=0
    EXPECT_GT(z, -10); // Surface is above z=-10
}

TEST_F(ortho, ray_trace_height_miss)
{
    // GIVEN: a surface model with a small mesh
    surface_model s;
    point_cloud camera_locations = {{0, 0, 10}};
    point_cloud cloud;
    cloud.emplace_back(5, 5, -10);
    cloud.emplace_back(6, 5, -10);
    cloud.emplace_back(5, 6, -10);
    s.cloud.push_back(cloud);
    s.mesh = rebuildMesh(camera_locations, {s});

    // WHEN: we ray-trace outside the mesh
    double z = rayTraceHeight(100, 100, 10, {s});

    // THEN: it should return NAN
    EXPECT_TRUE(std::isnan(z));
}

TEST_F(ortho, calculate_gsd_multi)
{
    // GIVEN: a graph with 2 images at different heights
    init_cameras();
    MeasurementGraph multi_image_graph;

    // Image 1: camera_z = 9, surface_z = 0, elevation = 9, thumb_arc_pixel = 0.01 -> GSD = 0.09
    image img1 = graph.getNode(id[0])->payload;
    multi_image_graph.addNode(std::move(img1));

    // Image 2: height = 19, surface_z = 0, elevation = 19, thumb_arc_pixel = 0.01 -> GSD = 0.19
    image img2 = graph.getNode(id[1])->payload;
    img2.position.z() = 19;
    multi_image_graph.addNode(std::move(img2));

    // mean_camera_z = (9 + 19) / 2 = 14
    // elevation = 14 - 0 = 14
    // mean_gsd = 14 * 0.01 = 0.14

    // WHEN: we calculate the GSD
    double gsd = calculateGSD(multi_image_graph, {id[0], id[1]}, 0);

    // THEN: it should be 0.14
    EXPECT_NEAR(gsd, 0.14, 1e-7);
}

TEST_F(ortho, functional_ortho_scene)
{
    // GIVEN: A scene with two images and a mesh surface
    // Image 0 at (0, 0, 10), Color Red
    // Image 1 at (10, 0, 10), Color Blue
    // Surface is a rectangle from (-2, -2, 0) to (12, 2, 0)

    MeasurementGraph functional_graph;
    std::vector<NodePose> functional_nodePoses;

    auto model = std::make_shared<CameraModel>();
    model->focal_length_pixels = 500;
    model->principle_point << 50, 50;
    model->pixels_cols = 100;
    model->pixels_rows = 100;
    model->projection_type = opencalibration::ProjectionType::PLANAR;
    model->id = 100;

    auto down = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX());

    // Image 0 (Red)
    image img0;
    img0.orientation = Eigen::Quaterniond::Identity() * down;
    img0.position = {0, 0, 10};
    img0.model = model;
    img0.thumbnail = RGBRaster(100, 100, 3);
    img0.thumbnail.layers[0].pixels.fill(255); // Red
    img0.thumbnail.layers[1].pixels.fill(0);
    img0.thumbnail.layers[2].pixels.fill(0);
    size_t id0 = functional_graph.addNode(std::move(img0));

    // Image 1 (Blue)
    image img1;
    img1.orientation = Eigen::Quaterniond::Identity() * down;
    img1.position = {10, 0, 10};
    img1.model = model;
    img1.thumbnail = RGBRaster(100, 100, 3);
    img1.thumbnail.layers[0].pixels.fill(0);
    img1.thumbnail.layers[1].pixels.fill(0);
    img1.thumbnail.layers[2].pixels.fill(255); // Blue
    size_t id1 = functional_graph.addNode(std::move(img1));

    // Mesh Surface: Use rebuildMesh to create a valid mesh from points
    surface_model points_surface;
    point_cloud cloud;
    cloud.emplace_back(-2, -2, 0);
    cloud.emplace_back(12, -2, 0);
    cloud.emplace_back(12, 2, 0);
    cloud.emplace_back(-2, 2, 0);
    cloud.emplace_back(5, 0, 0); // Add a middle point to ensure triangulation
    points_surface.cloud.push_back(cloud);

    point_cloud camera_locations = {{0, 0, 10}, {10, 0, 10}};

    surface_model functional_surface;
    functional_surface.mesh = rebuildMesh(camera_locations, {points_surface});

    // WHEN: we generate the orthomosaic
    OrthoMosaic result = generateOrthomosaic({functional_surface}, functional_graph);

    // THEN: it should have the correct GSD
    EXPECT_NEAR(result.gsd, 0.02, 1e-7);

    const auto &pixels = std::get<MultiLayerRaster<uint8_t>>(result.pixelValues);

    // Image 0 center at world (0, 0)
    EXPECT_EQ((int)pixels.layers[0].pixels(1000, 1000), 255); // R
    EXPECT_EQ((int)pixels.layers[1].pixels(1000, 1000), 0);   // G
    EXPECT_EQ((int)pixels.layers[2].pixels(1000, 1000), 0);   // B
    EXPECT_EQ(result.cameraUUID.pixels(1000, 1000), static_cast<uint32_t>(id0 & 0xFFFFFFFF));

    // Image 1 center at world (10, 0)
    EXPECT_EQ((int)pixels.layers[0].pixels(1000, 1500), 0);   // R
    EXPECT_EQ((int)pixels.layers[1].pixels(1000, 1500), 0);   // G
    EXPECT_EQ((int)pixels.layers[2].pixels(1000, 1500), 255); // B
    EXPECT_EQ(result.cameraUUID.pixels(1000, 1500), static_cast<uint32_t>(id1 & 0xFFFFFFFF));
}

TEST_F(ortho, measurement_3_images_points)
{
    // GIVEN: a graph with 3 images and a 3d point based surface model
    init_cameras();

    surface_model points_surface;
    points_surface.cloud.push_back(generate_planar_points());

    point_cloud camera_locations; // TODO: get camera locations
    for (const auto &nodePose : nodePoses)
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
