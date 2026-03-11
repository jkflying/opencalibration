#include <opencalibration/dense/dense_stereo.hpp>
#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/surface/intersect.hpp>
#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/surface_model.hpp>

#include <gtest/gtest.h>

#include <random>

using namespace opencalibration;

namespace
{

// Build a flat 2-triangle mesh on the z=0 plane covering [-50, 50] x [-50, 50]
MeshGraph buildFlatMesh()
{
    MeshGraph mesh;

    //   2 -------- 3
    //   | \        |
    //   |   \      |
    //   |     \    |
    //   |       \  |
    //   0 -------- 1

    size_t v0 = mesh.addNode(MeshNode{Eigen::Vector3d(-50, -50, 0)});
    size_t v1 = mesh.addNode(MeshNode{Eigen::Vector3d(50, -50, 0)});
    size_t v2 = mesh.addNode(MeshNode{Eigen::Vector3d(-50, 50, 0)});
    size_t v3 = mesh.addNode(MeshNode{Eigen::Vector3d(50, 50, 0)});

    MeshEdge bottomEdge;
    bottomEdge.border = true;
    bottomEdge.triangleOppositeNodes[0] = v3;
    mesh.addEdge(bottomEdge, v0, v1);

    MeshEdge rightEdge;
    rightEdge.border = true;
    rightEdge.triangleOppositeNodes[0] = v0;
    mesh.addEdge(rightEdge, v1, v3);

    MeshEdge topEdge;
    topEdge.border = true;
    topEdge.triangleOppositeNodes[0] = v0;
    mesh.addEdge(topEdge, v2, v3);

    MeshEdge leftEdge;
    leftEdge.border = true;
    leftEdge.triangleOppositeNodes[0] = v3;
    mesh.addEdge(leftEdge, v0, v2);

    MeshEdge diagEdge;
    diagEdge.border = false;
    diagEdge.triangleOppositeNodes[0] = v1;
    diagEdge.triangleOppositeNodes[1] = v2;
    mesh.addEdge(diagEdge, v0, v3);

    return mesh;
}

CameraModel makeDownwardCamera()
{
    CameraModel model;
    model.focal_length_pixels = 800;
    model.pixels_cols = 1600;
    model.pixels_rows = 1200;
    model.principle_point = Eigen::Vector2d(800, 600);
    model.projection_type = ProjectionType::PLANAR;
    model.radial_distortion = Eigen::Vector3d::Zero();
    model.tangential_distortion = Eigen::Vector2d::Zero();
    return model;
}

// Project a 3D world point into an image, return pixel coordinates
Eigen::Vector2d projectPoint(const Eigen::Vector3d &world_pt, const CameraModel &model,
                             const Eigen::Vector3d &cam_pos, const Eigen::Quaterniond &cam_ori)
{
    return image_from_3d(world_pt, model, cam_pos, cam_ori);
}

feature_2d makeFeature(const Eigen::Vector2d &location, uint64_t descriptor_seed, float strength = 1.0f)
{
    feature_2d f;
    f.location = location;
    f.strength = strength;

    std::mt19937_64 rng(descriptor_seed);
    for (int i = 0; i < feature_2d::DESCRIPTOR_BITS; i++)
    {
        f.descriptor[i] = rng() & 1;
    }
    return f;
}

// Create a noisy copy of a descriptor: flip `num_flips` random bits
feature_2d noisyFeature(const feature_2d &f, std::mt19937 &rng, int num_flips)
{
    feature_2d out = f;
    std::uniform_int_distribution<int> bit_dist(0, feature_2d::DESCRIPTOR_BITS - 1);
    for (int i = 0; i < num_flips; i++)
    {
        out.descriptor.flip(bit_dist(rng));
    }
    return out;
}

struct SceneResult
{
    size_t total_points = 0;
    double max_xy_error = 0;
    double mean_xy_error = 0;
};

// Build a synthetic scene with two cameras and ground truth points, optionally with noise.
// Returns the output points and error statistics relative to ground truth.
SceneResult runNoisyScene(const CameraModel &cam_model, const Eigen::Quaterniond &base_ori,
                          double pixel_noise_stddev, double orientation_noise_deg, int descriptor_bit_flips)
{
    std::mt19937 rng(12345);
    std::normal_distribution<double> pixel_dist(0, pixel_noise_stddev);

    Eigen::Vector3d cam1_pos(0, 0, 100);
    Eigen::Vector3d cam2_pos(10, 0, 100);

    // Apply orientation noise as small random rotations
    auto perturbOrientation = [&](const Eigen::Quaterniond &ori) -> Eigen::Quaterniond {
        if (orientation_noise_deg == 0)
            return ori;
        double rad = orientation_noise_deg * M_PI / 180.0;
        std::normal_distribution<double> angle_dist(0, rad);
        Eigen::Vector3d axis(angle_dist(rng), angle_dist(rng), angle_dist(rng));
        double angle = axis.norm();
        if (angle < 1e-12)
            return ori;
        return ori * Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis.normalized()));
    };

    Eigen::Quaterniond cam1_ori = perturbOrientation(base_ori);
    Eigen::Quaterniond cam2_ori = perturbOrientation(base_ori);

    auto model_ptr = std::make_shared<CameraModel>(cam_model);

    // Ground truth points on z=0 in the overlap region
    std::vector<Eigen::Vector3d> gt_points;
    for (double x = 0; x <= 10; x += 2)
    {
        for (double y = -4; y <= 4; y += 2)
        {
            gt_points.emplace_back(x, y, 0);
        }
    }

    image img1;
    img1.path = "cam1";
    img1.model = model_ptr;
    img1.position = cam1_pos;
    img1.orientation = cam1_ori;

    image img2;
    img2.path = "cam2";
    img2.model = model_ptr;
    img2.position = cam2_pos;
    img2.orientation = cam2_ori;

    uint64_t seed = 500;
    for (const auto &pt : gt_points)
    {
        // Project using the *perturbed* camera orientations (as the system would see them)
        Eigen::Vector2d px1 = image_from_3d(pt, cam_model, cam1_pos, cam1_ori);
        Eigen::Vector2d px2 = image_from_3d(pt, cam_model, cam2_pos, cam2_ori);

        // Add pixel noise
        px1 += Eigen::Vector2d(pixel_dist(rng), pixel_dist(rng));
        px2 += Eigen::Vector2d(pixel_dist(rng), pixel_dist(rng));

        bool in1 =
            px1.x() >= 0 && px1.x() < cam_model.pixels_cols && px1.y() >= 0 && px1.y() < cam_model.pixels_rows;
        bool in2 =
            px2.x() >= 0 && px2.x() < cam_model.pixels_cols && px2.y() >= 0 && px2.y() < cam_model.pixels_rows;

        feature_2d base_feat = makeFeature(px1, seed);

        if (in1)
        {
            feature_2d f1 = base_feat;
            f1.location = px1;
            if (descriptor_bit_flips > 0)
                f1 = noisyFeature(f1, rng, descriptor_bit_flips);
            img1.dense_features.push_back(f1);
        }
        if (in2)
        {
            feature_2d f2 = base_feat;
            f2.location = px2;
            if (descriptor_bit_flips > 0)
                f2 = noisyFeature(f2, rng, descriptor_bit_flips);
            img2.dense_features.push_back(f2);
        }
        seed++;
    }

    MeasurementGraph graph;
    graph.addNode(std::move(img1));
    graph.addNode(std::move(img2));

    std::vector<surface_model> surfaces(1);
    surfaces[0].mesh = buildFlatMesh();

    densifyMesh(graph, surfaces);

    SceneResult result;
    double error_sum = 0;
    for (const auto &cloud : surfaces[0].cloud)
    {
        for (const auto &pt : cloud)
        {
            result.total_points++;
            // Find closest ground truth point in XY
            double min_err = std::numeric_limits<double>::max();
            for (const auto &gt : gt_points)
            {
                double err = (pt.head<2>() - gt.head<2>()).norm();
                min_err = std::min(min_err, err);
            }
            result.max_xy_error = std::max(result.max_xy_error, min_err);
            error_sum += min_err;
        }
    }
    if (result.total_points > 0)
        result.mean_xy_error = error_sum / result.total_points;

    return result;
}

} // namespace

class DenseStereoTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        // Camera looking straight down: orientation that rotates camera z-axis to world -z
        // Camera convention: z forward, x right, y down
        // We want camera z to point toward world -z (down)
        // This is a 180 degree rotation around x-axis
        cam_ori = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));

        cam_model = makeDownwardCamera();
    }

    Eigen::Quaterniond cam_ori;
    CameraModel cam_model;
};

TEST_F(DenseStereoTest, synthetic_overlapping_cameras)
{
    MeasurementGraph graph;

    // Two cameras at z=100, separated by 10 units in x, looking straight down
    Eigen::Vector3d cam1_pos(0, 0, 100);
    Eigen::Vector3d cam2_pos(10, 0, 100);

    auto model_ptr = std::make_shared<CameraModel>(cam_model);

    image img1;
    img1.path = "cam1";
    img1.model = model_ptr;
    img1.position = cam1_pos;
    img1.orientation = cam_ori;

    image img2;
    img2.path = "cam2";
    img2.model = model_ptr;
    img2.position = cam2_pos;
    img2.orientation = cam_ori;

    // Create ground truth 3D points on the z=0 plane in the overlap region
    std::vector<Eigen::Vector3d> ground_truth_points;
    for (double x = -5; x <= 15; x += 2)
    {
        for (double y = -5; y <= 5; y += 2)
        {
            ground_truth_points.emplace_back(x, y, 0);
        }
    }

    // Project each ground truth point into both cameras → dense features with matching descriptors
    uint64_t seed = 42;
    for (const auto &pt : ground_truth_points)
    {
        Eigen::Vector2d px1 = projectPoint(pt, cam_model, cam1_pos, cam_ori);
        Eigen::Vector2d px2 = projectPoint(pt, cam_model, cam2_pos, cam_ori);

        bool in1 = px1.x() >= 0 && px1.x() < cam_model.pixels_cols && px1.y() >= 0 && px1.y() < cam_model.pixels_rows;
        bool in2 = px2.x() >= 0 && px2.x() < cam_model.pixels_cols && px2.y() >= 0 && px2.y() < cam_model.pixels_rows;

        if (in1)
            img1.dense_features.push_back(makeFeature(px1, seed));
        if (in2)
            img2.dense_features.push_back(makeFeature(px2, seed));

        seed++;
    }

    ASSERT_GT(img1.dense_features.size(), 0);
    ASSERT_GT(img2.dense_features.size(), 0);

    size_t nid1 = graph.addNode(std::move(img1));
    size_t nid2 = graph.addNode(std::move(img2));
    (void)nid1;
    (void)nid2;

    // Build surface with flat mesh
    std::vector<surface_model> surfaces(1);
    surfaces[0].mesh = buildFlatMesh();

    // Verify mesh intersection works before running dense
    {
        MeshIntersectionSearcher test_searcher;
        ASSERT_TRUE(test_searcher.init(surfaces[0].mesh));
        ray_d test_ray;
        test_ray.offset = cam1_pos;
        test_ray.dir = Eigen::Vector3d(0, 0, -1);
        const auto &result = test_searcher.triangleIntersect(test_ray);
        ASSERT_EQ(result.type, MeshIntersectionSearcher::IntersectionInfo::INTERSECTION);
        EXPECT_NEAR(result.intersectionLocation.z(), 0, 1e-6);
    }

    densifyMesh(graph, surfaces);

    // Check that we got 3D points
    size_t total_points = 0;
    for (const auto &cloud : surfaces[0].cloud)
    {
        total_points += cloud.size();
    }

    EXPECT_GT(total_points, 0) << "Expected dense matching to produce 3D points";

    // All produced points should lie on the z=0 plane (the mesh surface)
    for (const auto &cloud : surfaces[0].cloud)
    {
        for (const auto &pt : cloud)
        {
            EXPECT_NEAR(pt.z(), 0.0, 0.1) << "Point should lie on mesh surface";
            EXPECT_GT(pt.x(), -50);
            EXPECT_LT(pt.x(), 50);
            EXPECT_GT(pt.y(), -50);
            EXPECT_LT(pt.y(), 50);
        }
    }
}

TEST_F(DenseStereoTest, no_match_with_different_descriptors)
{
    MeasurementGraph graph;

    Eigen::Vector3d cam1_pos(0, 0, 100);
    Eigen::Vector3d cam2_pos(10, 0, 100);

    auto model_ptr = std::make_shared<CameraModel>(cam_model);

    image img1;
    img1.path = "cam1";
    img1.model = model_ptr;
    img1.position = cam1_pos;
    img1.orientation = cam_ori;

    image img2;
    img2.path = "cam2";
    img2.model = model_ptr;
    img2.position = cam2_pos;
    img2.orientation = cam_ori;

    // Same ground point but different descriptors → should not match
    Eigen::Vector3d pt(5, 0, 0);
    Eigen::Vector2d px1 = projectPoint(pt, cam_model, cam1_pos, cam_ori);
    Eigen::Vector2d px2 = projectPoint(pt, cam_model, cam2_pos, cam_ori);

    img1.dense_features.push_back(makeFeature(px1, 100));
    img2.dense_features.push_back(makeFeature(px2, 999));

    graph.addNode(std::move(img1));
    graph.addNode(std::move(img2));

    std::vector<surface_model> surfaces(1);
    surfaces[0].mesh = buildFlatMesh();

    densifyMesh(graph, surfaces);

    size_t total_points = 0;
    for (const auto &cloud : surfaces[0].cloud)
    {
        total_points += cloud.size();
    }

    EXPECT_EQ(total_points, 0) << "Different descriptors should not produce matches";
}

TEST_F(DenseStereoTest, no_crash_empty_features)
{
    MeasurementGraph graph;

    auto model_ptr = std::make_shared<CameraModel>(cam_model);

    image img1;
    img1.path = "cam1";
    img1.model = model_ptr;
    img1.position = Eigen::Vector3d(0, 0, 100);
    img1.orientation = cam_ori;

    graph.addNode(std::move(img1));

    std::vector<surface_model> surfaces(1);
    surfaces[0].mesh = buildFlatMesh();

    densifyMesh(graph, surfaces);

    EXPECT_TRUE(surfaces[0].cloud.empty());
}

TEST_F(DenseStereoTest, no_crash_empty_surfaces)
{
    MeasurementGraph graph;
    std::vector<surface_model> surfaces;
    densifyMesh(graph, surfaces);
}

TEST_F(DenseStereoTest, points_from_multiple_cameras)
{
    MeasurementGraph graph;

    auto model_ptr = std::make_shared<CameraModel>(cam_model);

    // 4 cameras in a grid pattern at z=100
    std::vector<Eigen::Vector3d> cam_positions = {
        {0, 0, 100}, {10, 0, 100}, {0, 10, 100}, {10, 10, 100}};

    // Ground truth points in the center overlap region
    std::vector<Eigen::Vector3d> ground_points;
    for (double x = 2; x <= 8; x += 2)
    {
        for (double y = 2; y <= 8; y += 2)
        {
            ground_points.emplace_back(x, y, 0);
        }
    }

    uint64_t seed = 1000;
    for (const auto &cam_pos : cam_positions)
    {
        image img;
        img.path = "cam";
        img.model = model_ptr;
        img.position = cam_pos;
        img.orientation = cam_ori;

        uint64_t pt_seed = seed;
        for (const auto &pt : ground_points)
        {
            Eigen::Vector2d px = projectPoint(pt, cam_model, cam_pos, cam_ori);
            if (px.x() >= 0 && px.x() < cam_model.pixels_cols && px.y() >= 0 && px.y() < cam_model.pixels_rows)
            {
                img.dense_features.push_back(makeFeature(px, pt_seed));
            }
            pt_seed++;
        }

        graph.addNode(std::move(img));
    }

    std::vector<surface_model> surfaces(1);
    surfaces[0].mesh = buildFlatMesh();

    densifyMesh(graph, surfaces);

    size_t total_points = 0;
    for (const auto &cloud : surfaces[0].cloud)
    {
        total_points += cloud.size();
    }

    EXPECT_GT(total_points, 0) << "Multiple cameras should produce matches";

    // Track merging should produce at most one point per ground truth point,
    // not one per (source image, ground truth point) pair
    EXPECT_LE(total_points, ground_points.size())
        << "Track merging should deduplicate points seen from multiple cameras";

    for (const auto &cloud : surfaces[0].cloud)
    {
        for (const auto &pt : cloud)
        {
            EXPECT_NEAR(pt.z(), 0.0, 0.1);
        }
    }
}

TEST_F(DenseStereoTest, accuracy_with_pixel_noise)
{
    // 1 pixel stddev noise → at altitude 100, focal 800: ~0.125 world units per pixel
    auto result = runNoisyScene(cam_model, cam_ori, 1.0, 0, 0);

    EXPECT_GT(result.total_points, 0) << "Should still find matches with 1px noise";
    EXPECT_LT(result.max_xy_error, 0.5) << "Max XY error should be bounded with 1px noise";
    EXPECT_LT(result.mean_xy_error, 0.25) << "Mean XY error should be small with 1px noise";
}

TEST_F(DenseStereoTest, accuracy_with_orientation_noise)
{
    // 0.1 degree orientation noise → at altitude 100: ~0.17 world units
    auto result = runNoisyScene(cam_model, cam_ori, 0, 0.1, 0);

    EXPECT_GT(result.total_points, 0) << "Should still find matches with 0.1deg orientation noise";
    EXPECT_LT(result.max_xy_error, 0.5) << "Max XY error should be bounded with orientation noise";
    EXPECT_LT(result.mean_xy_error, 0.3) << "Mean XY error should be small with orientation noise";
}

TEST_F(DenseStereoTest, accuracy_with_descriptor_noise)
{
    // Flip 20 out of 256 bits (~8% noise) — should still match
    auto result = runNoisyScene(cam_model, cam_ori, 0, 0, 20);

    EXPECT_GT(result.total_points, 0) << "Should still find matches with 20-bit descriptor noise";
    // No pixel/orientation noise so points should land exactly on ground truth
    EXPECT_LT(result.max_xy_error, 0.01);
}

TEST_F(DenseStereoTest, accuracy_with_combined_noise)
{
    // Realistic scenario: small pixel noise + small orientation noise + descriptor noise
    auto result = runNoisyScene(cam_model, cam_ori, 0.5, 0.05, 10);

    EXPECT_GT(result.total_points, 0) << "Should find matches with combined noise";
    EXPECT_LT(result.max_xy_error, 0.5) << "Max error should be bounded with combined noise";
}

TEST_F(DenseStereoTest, heavy_descriptor_noise_rejects)
{
    // Flip 200 out of 256 bits per image — after independent noise on both images,
    // the expected inter-image hamming distance is ~0.5, well above the 0.3 threshold
    auto result = runNoisyScene(cam_model, cam_ori, 0, 0, 200);

    EXPECT_EQ(result.total_points, 0) << "Heavy descriptor noise should prevent matches";
}
