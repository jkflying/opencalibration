#include <jk/KDTree.h>
#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/geo_coord/geo_coord.hpp>
#include <opencalibration/ortho/image_cache.hpp>
#include <opencalibration/ortho/ortho.hpp>
#include <opencalibration/ortho/patch_sampler.hpp>
#include <opencalibration/relax/relax.hpp>
#include <opencalibration/relax/relax_cost_function.hpp>
#include <opencalibration/relax/relax_problem.hpp>
#include <opencalibration/surface/expand_mesh.hpp>
#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_pose.hpp>
#include <opencalibration/types/point_cloud.hpp>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <filesystem>
#include <random>
#include <set>

using namespace opencalibration;
using namespace opencalibration::orthomosaic;
using namespace std::chrono_literals;

struct ortho : public ::testing::Test
{
    size_t id[3];
    MeasurementGraph graph;
    std::vector<NodePose> nodePoses;
    ankerl::unordered_dense::map<size_t, CameraModel> cam_models;
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
    MeshNode n1;
    n1.location = {1, 2, 3};
    s.mesh.addNode(n1);
    MeshNode n2;
    n2.location = {5, 6, 7};
    s.mesh.addNode(n2);

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
    auto nearest = context.imageGPSLocations.searchKnn({10, 10}, 1);
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
    double z = RayTraceContext({s}).traceHeight(7.5, 7.5, 10);

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
    double z = RayTraceContext({s}).traceHeight(100, 100, 10);

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
    img0.metadata.camera_info.width_px = model->pixels_cols;
    img0.metadata.camera_info.height_px = model->pixels_rows;
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
    img1.metadata.camera_info.width_px = model->pixels_cols;
    img1.metadata.camera_info.height_px = model->pixels_rows;
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

    // THEN: GSD should be positive (clamped from 0.02 to fit within input pixel budget)
    EXPECT_GT(result.gsd, 0.0);

    const auto &pixels = std::get<MultiLayerRaster<uint8_t>>(result.pixelValues);

    // Mesh bounds: cameras at (0,0) and (10,0) at height 10, border = height*2 = 20
    // → min_x=-20, max_y=20. With clamping (2 * 100x100 inputs vs ~5MP natural output),
    // GSD ≈ 0.316m and raster ≈ 126×158. Pixel indices:
    //   world(0,  0): row = (20-0)/gsd ≈ 63, col = (0+20)/gsd ≈ 63
    //   world(10, 0): row ≈ 63,              col = (10+20)/gsd ≈ 94
    constexpr double scene_min_x = -20.0, scene_max_y = 20.0;
    const int row_y0 = static_cast<int>(scene_max_y / result.gsd);
    const int col_x0 = static_cast<int>((0.0 - scene_min_x) / result.gsd);
    const int col_x10 = static_cast<int>((10.0 - scene_min_x) / result.gsd);

    // Image 0 center at world (0, 0)
    EXPECT_EQ((int)pixels.layers[0].pixels(row_y0, col_x0), 255); // R
    EXPECT_EQ((int)pixels.layers[1].pixels(row_y0, col_x0), 0);   // G
    EXPECT_EQ((int)pixels.layers[2].pixels(row_y0, col_x0), 0);   // B
    EXPECT_EQ(result.cameraUUID.pixels(row_y0, col_x0), static_cast<uint32_t>(id0 & 0xFFFFFFFF));

    // Image 1 center at world (10, 0)
    EXPECT_EQ((int)pixels.layers[0].pixels(row_y0, col_x10), 0);   // R
    EXPECT_EQ((int)pixels.layers[1].pixels(row_y0, col_x10), 0);   // G
    EXPECT_EQ((int)pixels.layers[2].pixels(row_y0, col_x10), 255); // B
    EXPECT_EQ(result.cameraUUID.pixels(row_y0, col_x10), static_cast<uint32_t>(id1 & 0xFFFFFFFF));
}

TEST_F(ortho, thumbnail_pixels_are_georeferenced_at_pixel_centres)
{
    // GIVEN: a nadir camera over a tilted plane, with a non-square thumbnail where every pixel has a unique colour
    MeasurementGraph thumb_graph;
    auto cam_model = std::make_shared<CameraModel>();
    cam_model->focal_length_pixels = 500;
    cam_model->principle_point << 500, 375;
    cam_model->pixels_cols = 1000;
    cam_model->pixels_rows = 750;
    cam_model->projection_type = opencalibration::ProjectionType::PLANAR;
    cam_model->id = 7;

    const int thumb_cols = 50, thumb_rows = 38;
    image img;
    img.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
    img.position = {0, 0, 40};
    img.model = cam_model;
    img.metadata.camera_info.width_px = cam_model->pixels_cols;
    img.metadata.camera_info.height_px = cam_model->pixels_rows;
    img.thumbnail = RGBRaster(thumb_rows, thumb_cols, 3);
    for (int r = 0; r < thumb_rows; r++)
        for (int c = 0; c < thumb_cols; c++)
        {
            img.thumbnail.layers[0].pixels(r, c) = static_cast<uint8_t>(5 * c);
            img.thumbnail.layers[1].pixels(r, c) = static_cast<uint8_t>(6 * r);
            img.thumbnail.layers[2].pixels(r, c) = 100;
        }
    const image &camera = thumb_graph.getNode(thumb_graph.addNode(std::move(img)))->payload;

    surface_model points_surface;
    point_cloud cloud;
    auto plane_z = [](double x, double y) { return 0.3 * x + 0.2 * y; };
    for (int x = -10; x <= 10; x += 2)
        for (int y = -10; y <= 10; y += 2)
            cloud.emplace_back(x, y, plane_z(x, y));
    points_surface.cloud.push_back(cloud);
    surface_model surface;
    surface.mesh = rebuildMesh({camera.position, camera.position + Eigen::Vector3d(1, 0, 0)}, {points_surface});

    // WHEN: we generate the thumbnail orthomosaic
    const OrthoMosaic result = generateOrthomosaic({surface}, thumb_graph);

    // THEN: the raster covers the bounds and matches the context's georeference
    const auto expected_bounds = calculateBoundsAndMeanZ({surface});
    EXPECT_DOUBLE_EQ(result.bounds.min_x, expected_bounds.min_x);
    EXPECT_DOUBLE_EQ(result.bounds.max_y, expected_bounds.max_y);
    const auto &pixels = std::get<MultiLayerRaster<uint8_t>>(result.pixelValues);
    const int width = static_cast<int>(pixels.layers[0].pixels.cols());
    const int height = static_cast<int>(pixels.layers[0].pixels.rows());
    EXPECT_GE(width * result.gsd, result.bounds.max_x - result.bounds.min_x);
    EXPECT_GE(height * result.gsd, result.bounds.max_y - result.bounds.min_y);
    EXPECT_LT((width - 1) * result.gsd, result.bounds.max_x - result.bounds.min_x);
    EXPECT_LT((height - 1) * result.gsd, result.bounds.max_y - result.bounds.min_y);
    ASSERT_EQ(result.dsm.pixels.rows(), height);
    ASSERT_EQ(result.dsm.pixels.cols(), width);

    // AND: every pixel's DSM height and colour come from the surface/camera at that pixel's centre
    const Eigen::Matrix3d inv_rotation = camera.orientation.inverse().toRotationMatrix();
    const Eigen::Vector2d thumb_scale(static_cast<double>(thumb_cols) / cam_model->pixels_cols,
                                      static_cast<double>(thumb_rows) / cam_model->pixels_rows);
    const std::vector<surface_model> surfaces{surface};
    RayTraceContext ray_trace(surfaces);
    auto onPixelBoundary = [](const Eigen::Vector2d &pixel) {
        const Eigen::Array2d frac = pixel.array() - pixel.array().floor();
        return (frac < 1e-6).any() || (frac > 1 - 1e-6).any();
    };
    constexpr int kLabRoundTripTolerance = 2;
    int dsm_checked = 0, colour_checked = 0, dsm_mismatches = 0, colour_mismatches = 0;
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            const double x = result.bounds.min_x + (col + 0.5) * result.gsd;
            const double y = result.bounds.max_y - (row + 0.5) * result.gsd;
            const double z = ray_trace.traceHeight(x, y, camera.position.z());
            const float dsm = result.dsm.pixels(row, col);
            if (std::isnan(z))
            {
                dsm_mismatches += !std::isnan(dsm);
                continue;
            }
            dsm_checked++;
            dsm_mismatches += !(std::abs(dsm - z) < 1e-3);

            const Eigen::Vector2d thumb_pixel =
                image_from_3d(Eigen::Vector3d(x, y, z), *cam_model, camera.position, inv_rotation)
                    .cwiseProduct(thumb_scale);
            if (onPixelBoundary(thumb_pixel))
                continue;
            const int tc = static_cast<int>(std::floor(thumb_pixel.x()));
            const int tr = static_cast<int>(std::floor(thumb_pixel.y()));
            const uint8_t alpha = pixels.layers[3].pixels(row, col);
            if (tc < 0 || tc >= thumb_cols || tr < 0 || tr >= thumb_rows)
            {
                colour_mismatches += alpha != 0;
                continue;
            }
            colour_checked++;
            colour_mismatches += alpha != 255 ||
                                 std::abs(pixels.layers[0].pixels(row, col) - 5 * tc) > kLabRoundTripTolerance ||
                                 std::abs(pixels.layers[1].pixels(row, col) - 6 * tr) > kLabRoundTripTolerance ||
                                 std::abs(pixels.layers[2].pixels(row, col) - 100) > kLabRoundTripTolerance;
        }
    }
    EXPECT_GT(dsm_checked, width * height / 4);
    EXPECT_GT(colour_checked, 500);
    EXPECT_EQ(dsm_mismatches, 0);
    EXPECT_EQ(colour_mismatches, 0);
}

TEST_F(ortho, thumbnail_colour_balance_preserves_channel_order)
{
    // GIVEN: two overlapping nadir cameras that both see the same colour with distinct R, G and B
    const uint8_t rgb[3] = {200, 100, 30};
    MeasurementGraph balance_graph;
    auto cam_model = std::make_shared<CameraModel>();
    cam_model->focal_length_pixels = 500;
    cam_model->principle_point << 500, 375;
    cam_model->pixels_cols = 1000;
    cam_model->pixels_rows = 750;
    cam_model->projection_type = opencalibration::ProjectionType::PLANAR;
    cam_model->id = 7;
    point_cloud camera_locations = {{0, 0, 40}, {10, 0, 40}};
    for (const auto &position : camera_locations)
    {
        image img;
        img.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
        img.position = position;
        img.model = cam_model;
        img.metadata.camera_info.width_px = cam_model->pixels_cols;
        img.metadata.camera_info.height_px = cam_model->pixels_rows;
        img.thumbnail = RGBRaster(38, 50, 3);
        for (int c = 0; c < 3; c++)
            img.thumbnail.layers[c].pixels.fill(rgb[c]);
        balance_graph.addNode(std::move(img));
    }

    surface_model points_surface;
    point_cloud cloud;
    for (int x = -10; x <= 20; x += 2)
        for (int y = -10; y <= 10; y += 2)
            cloud.emplace_back(x, y, 0);
    points_surface.cloud.push_back(cloud);
    surface_model surface;
    surface.mesh = rebuildMesh(camera_locations, {points_surface});

    // WHEN: we generate the thumbnail, which colour balances the cameras against each other
    const OrthoMosaic result = generateOrthomosaic({surface}, balance_graph);

    // THEN: the balance ran, and every covered pixel still has the input colour in the right channels
    EXPECT_EQ(result.color_balance.per_image_params.size(), 2u);
    const auto &pixels = std::get<MultiLayerRaster<uint8_t>>(result.pixelValues);
    int covered = 0, mismatches = 0;
    for (Eigen::Index row = 0; row < pixels.layers[0].pixels.rows(); row++)
        for (Eigen::Index col = 0; col < pixels.layers[0].pixels.cols(); col++)
        {
            if (pixels.layers[3].pixels(row, col) == 0)
                continue;
            covered++;
            for (int c = 0; c < 3; c++)
                mismatches += std::abs(pixels.layers[c].pixels(row, col) - rgb[c]) > 3;
        }
    EXPECT_GT(covered, 500);
    EXPECT_EQ(mismatches, 0);
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
    ankerl::unordered_dense::set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
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

TEST(ortho_sample_geometry, view_angle_is_zero_along_tilted_optical_axis)
{
    // GIVEN: a camera tilted off nadir and yawed, and a world point straight along its optical axis
    image payload;
    payload.model = std::make_shared<CameraModel>();
    payload.model->pixels_cols = 800;
    payload.model->pixels_rows = 600;
    payload.position = Eigen::Vector3d(10, 20, 100);
    payload.orientation = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ()) *
                          Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()) *
                          Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX());
    const Eigen::Vector3d world_point = payload.position + payload.orientation * Eigen::Vector3d(0, 0, 50);

    // WHEN: we compute the sample geometry for that point
    const SampleGeometry g = sampleGeometry(payload, world_point, Eigen::Vector2d(400, 300));

    // THEN: the view angle relative to the optical axis is zero
    EXPECT_NEAR(g.view_angle_rad, 0.0, 1e-3);
}

TEST(ortho_coarsen_gsd, fits_regular_bounds)
{
    // GIVEN: 100m x 50m bounds at 0.1m gsd
    const OrthoMosaicBounds bounds{0, 100, 0, 50, 0};
    double gsd = 0.1;
    int width = 1000, height = 500;

    // WHEN: we coarsen to fit 10000 pixels
    coarsenGsdToFit(gsd, width, height, bounds, 10000);

    // THEN: the raster fits, without being coarsened far beyond what was needed
    EXPECT_LE(static_cast<uint64_t>(width) * height, 10000u);
    EXPECT_GT(static_cast<uint64_t>(width) * height, 8000u);
}

TEST(ortho_coarsen_gsd, terminates_on_degenerate_bounds)
{
    // GIVEN: zero-extent and non-finite bounds, with a pixel budget smaller than the fallback raster size
    const OrthoMosaicBounds zero_extent{5, 5, 7, 7, 0};
    const OrthoMosaicBounds non_finite{0, NAN, 0, 10, 0};
    double gsd_zero = 0.1, gsd_non_finite = 0.1;
    int w_zero = 100, h_zero = 100, w_non_finite = 100, h_non_finite = 100;

    // WHEN: we coarsen them
    coarsenGsdToFit(gsd_zero, w_zero, h_zero, zero_extent, 5000);
    coarsenGsdToFit(gsd_non_finite, w_non_finite, h_non_finite, non_finite, 5000);

    // THEN: it returns, shrinking the zero-extent raster to a single pixel and leaving the non-finite one alone
    EXPECT_EQ(w_zero, 1);
    EXPECT_EQ(h_zero, 1);
    EXPECT_EQ(w_non_finite, 100);
    EXPECT_EQ(h_non_finite, 100);
}

TEST(ortho_patch, arc_per_pixel_is_inverse_focal_length)
{
    // GIVEN: a planar camera model with a focal length of 600 pixels
    CameraModel model;
    model.focal_length_pixels = 600;
    model.principle_point << 400, 300;
    model.pixels_cols = 800;
    model.pixels_rows = 600;
    model.projection_type = ProjectionType::PLANAR;

    // WHEN: we calculate the arc per pixel
    const double arc_pixel = arcPerPixel(model);

    // THEN: it is the inverse focal length
    EXPECT_NEAR(arc_pixel, 1.0 / 600.0, 1e-9);
}

namespace
{
struct NadirCamera
{
    CameraModel model;
    Eigen::Vector3d position{0, 0, 10};
    Eigen::Matrix3d inverse_rotation =
        Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX())).inverse().toRotationMatrix();

    NadirCamera()
    {
        model.focal_length_pixels = 500;
        model.principle_point << 50.5, 50.5;
        model.pixels_cols = 100;
        model.pixels_rows = 100;
        model.projection_type = ProjectionType::PLANAR;
    }

    cv::Vec3b sample(const cv::Mat &image, double gsd) const
    {
        const Eigen::Vector3d world_point(0, 0, 0);
        const Eigen::Vector2d pixel = image_from_3d(world_point, model, position, inverse_rotation);
        cv::Vec3b result;
        PatchSampler sampler;
        sampler.sampleBlock(image, world_point, model, position, inverse_rotation, gsd, {{pixel, &result}});
        return result;
    }
};
} // namespace

TEST(ortho_patch, patch_sampler_jacobian)
{
    // GIVEN: a camera 10m above the origin looking straight down with a focal length of 500
    const NadirCamera camera;

    // WHEN: we compute the jacobian of the pixel with respect to the ground position at the origin
    const Eigen::Matrix2d J =
        PatchSampler::computeJacobian(Eigen::Vector3d::Zero(), camera.model, camera.position, camera.inverse_rotation);

    // THEN: each metre on the ground moves the pixel 500/10 = 50 pixels along the matching axis
    EXPECT_NEAR(std::abs(J(0, 0)), 50.0, 1e-6);
    EXPECT_NEAR(std::abs(J(1, 1)), 50.0, 1e-6);
    EXPECT_NEAR(J(0, 1), 0.0, 1e-6);
    EXPECT_NEAR(J(1, 0), 0.0, 1e-6);
}

TEST(ortho_patch, patch_sampler_single_pixel)
{
    // GIVEN: a black image with one coloured pixel at the principal point
    cv::Mat image(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    image.at<cv::Vec3b>(50, 50) = cv::Vec3b(100, 150, 200);
    const NadirCamera camera;

    // WHEN: we sample with an output GSD smaller than a source pixel
    const cv::Vec3b result = camera.sample(image, 0.01);

    // THEN: the nearest pixel is returned unchanged
    EXPECT_EQ(result, cv::Vec3b(100, 150, 200));
}

TEST(ortho_patch, patch_sampler_averaging)
{
    // GIVEN: a black image with a white disk around the principal point
    cv::Mat image(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::circle(image, cv::Point(50, 50), 10, cv::Scalar(255, 255, 255), -1);
    const NadirCamera camera;

    // WHEN: we sample with an output GSD covering many source pixels
    const cv::Vec3b result = camera.sample(image, 0.5);

    // THEN: the result is a grey mix of the disk and the background
    EXPECT_GT(result[0], 50);
    EXPECT_LT(result[0], 255);
    EXPECT_EQ(result[0], result[1]);
    EXPECT_EQ(result[1], result[2]);
}

TEST(ortho_tiles, tile_cameras_match_nearest_camera_of_every_pixel_centre)
{
    // GIVEN: a 10x10 pixel raster and cameras scattered around it
    const OrthoMosaicBounds bounds{0, 10, 0, 10, 0};
    const double gsd = 1;
    const int tile_size = 10;
    jk::tree::KDTree<size_t, 2> cameras;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> position(-1, 11);
    for (size_t i = 0; i < 60; i++)
        cameras.addPoint({position(rng), position(rng)}, i);

    // WHEN: we find the cameras for the tile
    const auto found = findTileCameras(0, 0, tile_size, bounds, gsd, tile_size, tile_size, cameras, 1);

    // THEN: they are exactly the nearest camera of each pixel centre
    ankerl::unordered_dense::set<size_t> expected;
    for (int row = 0; row < tile_size; row++)
        for (int col = 0; col < tile_size; col++)
            expected.insert(cameras.search({col + 0.5, 10 - (row + 0.5)}).payload);
    EXPECT_EQ(std::set<size_t>(found.begin(), found.end()), std::set<size_t>(expected.begin(), expected.end()));
}
