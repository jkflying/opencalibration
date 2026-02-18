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

#include <ceres/jet.h>
#include <eigen3/Eigen/Eigenvalues>
#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

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

// Unit tests for LAB patch sampling helper functions
namespace
{
// Test helper: these functions are in anonymous namespace in ortho.cpp,
// so we reimplement minimal versions here for testing the algorithm

double testCalculateArcPixel(const CameraModel &model)
{
    const double h = 0.001;
    Eigen::Vector2d pixel = image_from_3d(Eigen::Vector3d{0, 0, 1}, model);
    Eigen::Vector2d pixelShift = image_from_3d(Eigen::Vector3d{h, 0, 1}, model);
    return h / (pixel - pixelShift).norm();
}

int testCalculatePatchRadius(double output_gsd, double arc_pixel, double camera_elevation, int max_radius = 16)
{
    double source_gsd = std::abs(camera_elevation * arc_pixel);
    if (source_gsd <= 0 || !std::isfinite(source_gsd))
        return 0;
    double ratio = output_gsd / source_gsd;
    if (ratio <= 1.0)
        return 0;
    return std::min(static_cast<int>(std::ceil(ratio / 2.0)), max_radius);
}
} // namespace

TEST(ortho_patch, calculate_patch_radius_no_averaging_needed)
{
    // GIVEN: source GSD is larger than output GSD (lower resolution source)
    double arc_pixel = 0.001;     // 1 mrad per pixel
    double camera_elevation = 10; // 10m above ground
    // source_gsd = 10 * 0.001 = 0.01m = 1cm

    double output_gsd = 0.005; // 5mm output (higher res than source)

    // WHEN: we calculate patch radius
    int radius = testCalculatePatchRadius(output_gsd, arc_pixel, camera_elevation);

    // THEN: radius should be 0 (no averaging needed, source is lower res)
    EXPECT_EQ(radius, 0);
}

TEST(ortho_patch, calculate_patch_radius_averaging_needed)
{
    // GIVEN: source GSD is smaller than output GSD (higher resolution source)
    double arc_pixel = 0.001;     // 1 mrad per pixel
    double camera_elevation = 10; // 10m above ground
    // source_gsd = 10 * 0.001 = 0.01m = 1cm

    double output_gsd = 0.04; // 4cm output (lower res than source)
    // ratio = 0.04 / 0.01 = 4
    // patch_radius = ceil(4 / 2) = 2

    // WHEN: we calculate patch radius
    int radius = testCalculatePatchRadius(output_gsd, arc_pixel, camera_elevation);

    // THEN: radius should be 2
    EXPECT_EQ(radius, 2);
}

TEST(ortho_patch, calculate_patch_radius_max_clamp)
{
    // GIVEN: very large GSD ratio
    double arc_pixel = 0.0001; // Very fine source
    double camera_elevation = 10;
    // source_gsd = 10 * 0.0001 = 0.001m = 1mm

    double output_gsd = 1.0; // 1m output
    // ratio = 1.0 / 0.001 = 1000
    // patch_radius = ceil(1000 / 2) = 500, but clamped to 16

    // WHEN: we calculate patch radius
    int radius = testCalculatePatchRadius(output_gsd, arc_pixel, camera_elevation);

    // THEN: radius should be clamped to 16
    EXPECT_EQ(radius, 16);
}

TEST(ortho_patch, calculate_patch_radius_zero_arc_pixel)
{
    // GIVEN: zero arc_pixel (edge case)
    double arc_pixel = 0;
    double camera_elevation = 10;
    double output_gsd = 0.04;

    // WHEN: we calculate patch radius
    int radius = testCalculatePatchRadius(output_gsd, arc_pixel, camera_elevation);

    // THEN: radius should be 0 (fallback to single pixel)
    EXPECT_EQ(radius, 0);
}

TEST(ortho_patch, calculate_arc_pixel)
{
    // GIVEN: a camera model with known focal length
    CameraModel model;
    model.focal_length_pixels = 600;
    model.principle_point << 400, 300;
    model.pixels_cols = 800;
    model.pixels_rows = 600;
    model.projection_type = ProjectionType::PLANAR;

    // WHEN: we calculate arc_pixel
    double arc_pixel = testCalculateArcPixel(model);

    // THEN: arc_pixel should be approximately 1/focal_length
    // For planar projection: pixel_shift = h * focal_length
    // So arc_pixel = h / (h * focal_length) = 1 / focal_length
    EXPECT_NEAR(arc_pixel, 1.0 / 600.0, 1e-9);
}

namespace
{
class TestPatchSampler
{
  public:
    static constexpr int MAX_PATCH_RADIUS = 16;

    TestPatchSampler() : _lab_pixel(1, 1, CV_8UC3), _bgr_pixel(1, 1, CV_8UC3)
    {
    }

    Eigen::Matrix2d computeJacobian(const Eigen::Vector3d &world_point, const CameraModel &model,
                                    const Eigen::Vector3d &camera_position,
                                    const Eigen::Matrix3d &camera_orientation_inverse) const
    {
        using JetT = ceres::Jet<double, 2>;

        Eigen::Matrix<JetT, 3, 1> world_point_jet;
        world_point_jet[0] = JetT(world_point.x(), 0);
        world_point_jet[1] = JetT(world_point.y(), 1);
        world_point_jet[2] = JetT(world_point.z());

        DifferentiableCameraModel<JetT> model_jet;
        model_jet.focal_length_pixels = JetT(model.focal_length_pixels);
        model_jet.principle_point = model.principle_point.cast<JetT>();
        model_jet.radial_distortion = model.radial_distortion.cast<JetT>();
        model_jet.tangential_distortion = model.tangential_distortion.cast<JetT>();
        model_jet.pixels_cols = model.pixels_cols;
        model_jet.pixels_rows = model.pixels_rows;
        model_jet.projection_type = model.projection_type;

        Eigen::Matrix<JetT, 3, 1> camera_position_jet = camera_position.cast<JetT>();
        Eigen::Matrix<JetT, 3, 3> camera_orientation_inverse_jet = camera_orientation_inverse.cast<JetT>();

        Eigen::Matrix<JetT, 2, 1> pixel_jet =
            image_from_3d(world_point_jet, model_jet, camera_position_jet, camera_orientation_inverse_jet);

        Eigen::Matrix2d J;
        J(0, 0) = pixel_jet[0].v[0];
        J(0, 1) = pixel_jet[0].v[1];
        J(1, 0) = pixel_jet[1].v[0];
        J(1, 1) = pixel_jet[1].v[1];

        return J;
    }

    bool sampleWithJacobian(const cv::Mat &bgr_image, const Eigen::Vector3d &world_point, const CameraModel &model,
                            const Eigen::Vector3d &camera_position, const Eigen::Matrix3d &camera_orientation_inverse,
                            double output_gsd, cv::Vec3b &result)
    {
        Eigen::Vector2d pixel = image_from_3d(world_point, model, camera_position, camera_orientation_inverse);

        if (pixel.x() < 0 || pixel.x() >= bgr_image.cols || pixel.y() < 0 || pixel.y() >= bgr_image.rows)
        {
            return false;
        }

        Eigen::Matrix2d J = computeJacobian(world_point, model, camera_position, camera_orientation_inverse);
        Eigen::Matrix2d M = output_gsd * output_gsd * J * J.transpose();

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(M);
        Eigen::Vector2d eigenvalues = solver.eigenvalues();

        double a = std::sqrt(std::max(eigenvalues(1), 1e-6));
        double b = std::sqrt(std::max(eigenvalues(0), 1e-6));

        if (a < 1.0 && b < 1.0)
        {
            int px = static_cast<int>(pixel.x());
            int py = static_cast<int>(pixel.y());
            if (px >= 0 && px < bgr_image.cols && py >= 0 && py < bgr_image.rows)
            {
                result = bgr_image.at<cv::Vec3b>(py, px);
                return true;
            }
            return false;
        }

        int radius = std::min(static_cast<int>(std::ceil(a)), MAX_PATCH_RADIUS);

        int x_min = std::max(0, static_cast<int>(pixel.x()) - radius);
        int y_min = std::max(0, static_cast<int>(pixel.y()) - radius);
        int x_max = std::min(bgr_image.cols - 1, static_cast<int>(pixel.x()) + radius);
        int y_max = std::min(bgr_image.rows - 1, static_cast<int>(pixel.y()) + radius);

        if (x_min > x_max || y_min > y_max)
        {
            return false;
        }

        double det = M.determinant();
        if (det < 1e-12)
        {
            int px = static_cast<int>(pixel.x());
            int py = static_cast<int>(pixel.y());
            result = bgr_image.at<cv::Vec3b>(py, px);
            return true;
        }
        Eigen::Matrix2d M_inv = M.inverse();

        double sum_L = 0, sum_a = 0, sum_b = 0;
        int count = 0;

        for (int py = y_min; py <= y_max; py++)
        {
            for (int px = x_min; px <= x_max; px++)
            {
                Eigen::Vector2d diff(px - pixel.x(), py - pixel.y());
                double ellipse_dist = diff.transpose() * M_inv * diff;

                if (ellipse_dist <= 1.0)
                {
                    const cv::Vec3b &bgr = bgr_image.at<cv::Vec3b>(py, px);

                    _lab_pixel.at<cv::Vec3b>(0, 0) = bgr;
                    cv::cvtColor(_lab_pixel, _bgr_pixel, cv::COLOR_BGR2Lab);
                    cv::Vec3b lab = _bgr_pixel.at<cv::Vec3b>(0, 0);

                    sum_L += lab[0];
                    sum_a += lab[1];
                    sum_b += lab[2];
                    count++;
                }
            }
        }

        if (count == 0)
        {
            return false;
        }

        _lab_pixel.at<cv::Vec3b>(0, 0) =
            cv::Vec3b(static_cast<uint8_t>(sum_L / count), static_cast<uint8_t>(sum_a / count),
                      static_cast<uint8_t>(sum_b / count));
        cv::cvtColor(_lab_pixel, _bgr_pixel, cv::COLOR_Lab2BGR);
        result = _bgr_pixel.at<cv::Vec3b>(0, 0);

        return true;
    }

  private:
    mutable cv::Mat _lab_pixel;
    mutable cv::Mat _bgr_pixel;
};
} // namespace

TEST(ortho_patch, patch_sampler_jacobian)
{
    CameraModel model;
    model.focal_length_pixels = 500;
    model.principle_point << 50, 50;
    model.pixels_cols = 100;
    model.pixels_rows = 100;
    model.projection_type = ProjectionType::PLANAR;

    Eigen::Vector3d camera_position(0, 0, 10);
    auto camera_orientation = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX());
    Eigen::Matrix3d inv_rotation = Eigen::Quaterniond(camera_orientation).inverse().toRotationMatrix();

    Eigen::Vector3d world_point(0, 0, 0);

    TestPatchSampler sampler;
    Eigen::Matrix2d J = sampler.computeJacobian(world_point, model, camera_position, inv_rotation);

    // For a camera at height 10 looking straight down with focal length 500:
    // pixel = focal * (world - cam) / z_cam = 500 * world / 10 = 50 * world
    // So J should be diag(50, 50) (with sign depending on orientation)
    EXPECT_NEAR(std::abs(J(0, 0)), 50.0, 1e-6);
    EXPECT_NEAR(std::abs(J(1, 1)), 50.0, 1e-6);
    EXPECT_NEAR(J(0, 1), 0.0, 1e-6);
    EXPECT_NEAR(J(1, 0), 0.0, 1e-6);
}

TEST(ortho_patch, patch_sampler_single_pixel)
{
    cv::Mat image(100, 100, CV_8UC3, cv::Scalar(100, 150, 200));

    CameraModel model;
    model.focal_length_pixels = 500;
    model.principle_point << 50, 50;
    model.pixels_cols = 100;
    model.pixels_rows = 100;
    model.projection_type = ProjectionType::PLANAR;

    Eigen::Vector3d camera_position(0, 0, 10);
    auto camera_orientation = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX());
    Eigen::Matrix3d inv_rotation = Eigen::Quaterniond(camera_orientation).inverse().toRotationMatrix();

    Eigen::Vector3d world_point(0, 0, 0);
    double gsd = 0.01; // Small GSD means ellipse < 1 pixel, so single pixel sample

    TestPatchSampler sampler;
    cv::Vec3b result;
    bool success = sampler.sampleWithJacobian(image, world_point, model, camera_position, inv_rotation, gsd, result);

    EXPECT_TRUE(success);
    EXPECT_EQ(result[0], 100);
    EXPECT_EQ(result[1], 150);
    EXPECT_EQ(result[2], 200);
}

TEST(ortho_patch, patch_sampler_averaging)
{
    cv::Mat image(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::circle(image, cv::Point(50, 50), 10, cv::Scalar(255, 255, 255), -1);

    CameraModel model;
    model.focal_length_pixels = 500;
    model.principle_point << 50, 50;
    model.pixels_cols = 100;
    model.pixels_rows = 100;
    model.projection_type = ProjectionType::PLANAR;

    Eigen::Vector3d camera_position(0, 0, 10);
    auto camera_orientation = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX());
    Eigen::Matrix3d inv_rotation = Eigen::Quaterniond(camera_orientation).inverse().toRotationMatrix();

    Eigen::Vector3d world_point(0, 0, 0);
    double gsd = 0.5; // Large GSD means ellipse covers multiple pixels

    TestPatchSampler sampler;
    cv::Vec3b result;
    bool success = sampler.sampleWithJacobian(image, world_point, model, camera_position, inv_rotation, gsd, result);

    EXPECT_TRUE(success);
    // Result should be a gray value (mix of black background and white circle)
    EXPECT_GT(result[0], 50);
    EXPECT_LT(result[0], 255);
}
