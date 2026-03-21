#include <opencalibration/model_inliers/ransac.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

using namespace opencalibration;

struct SyntheticScene
{
    Eigen::Matrix3d ground_truth; // H or F
    std::vector<correspondence> correspondences;
    std::vector<bool> ground_truth_inliers;

    static SyntheticScene homography(size_t n_inliers, size_t n_outliers, unsigned seed)
    {
        SyntheticScene scene;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> point_dist(-1.0, 1.0);
        std::uniform_real_distribution<double> outlier_dist(-2.0, 2.0);

        Eigen::Matrix3d R = Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        Eigen::Vector3d t(0.05, -0.03, 0.0);
        Eigen::Vector3d n(0, 0, 1);

        scene.ground_truth = R + t * n.transpose() / 10.0;
        scene.ground_truth /= scene.ground_truth(2, 2);

        scene.correspondences.reserve(n_inliers + n_outliers);
        scene.ground_truth_inliers.reserve(n_inliers + n_outliers);

        for (size_t i = 0; i < n_inliers; i++)
        {
            Eigen::Vector3d p1(point_dist(rng), point_dist(rng), 1.0);
            Eigen::Vector3d p2 = scene.ground_truth * p1;
            p2 /= p2.z();

            correspondence cor;
            cor.measurement1 = p1;
            cor.measurement2 = p2;
            scene.correspondences.push_back(cor);
            scene.ground_truth_inliers.push_back(true);
        }

        for (size_t i = 0; i < n_outliers; i++)
        {
            correspondence cor;
            cor.measurement1 = Eigen::Vector3d(outlier_dist(rng), outlier_dist(rng), 1.0);
            cor.measurement2 = Eigen::Vector3d(outlier_dist(rng), outlier_dist(rng), 1.0);
            scene.correspondences.push_back(cor);
            scene.ground_truth_inliers.push_back(false);
        }

        return scene;
    }

    static SyntheticScene fundamental(size_t n_inliers, size_t n_outliers, double planar_fraction, unsigned seed)
    {
        SyntheticScene scene;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> xy_dist(-1.0, 1.0);
        std::uniform_real_distribution<double> z_dist(5.0, 15.0);
        std::uniform_real_distribution<double> outlier_dist(-1.0, 1.0);

        Eigen::Matrix3d R1 = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t1(0, 0, 0);

        Eigen::Matrix3d R2 = Eigen::AngleAxisd(0.15, Eigen::Vector3d::UnitY()).toRotationMatrix();
        Eigen::Vector3d t2(0.5, 0.0, 0.0);

        Eigen::Vector3d e2 = R2 * (t1 - t2);
        Eigen::Matrix3d e2_cross;
        e2_cross << 0, -e2.z(), e2.y(), e2.z(), 0, -e2.x(), -e2.y(), e2.x(), 0;
        scene.ground_truth = e2_cross * R2 * R1.transpose();
        scene.ground_truth /= scene.ground_truth.norm();

        scene.correspondences.reserve(n_inliers + n_outliers);
        scene.ground_truth_inliers.reserve(n_inliers + n_outliers);

        size_t n_planar = static_cast<size_t>(n_inliers * planar_fraction);

        for (size_t i = 0; i < n_inliers; i++)
        {
            Eigen::Vector3d X;
            if (i < n_planar)
                X = Eigen::Vector3d(xy_dist(rng) * 3, xy_dist(rng) * 3, 10.0);
            else
                X = Eigen::Vector3d(xy_dist(rng) * 3, xy_dist(rng) * 3, z_dist(rng));

            Eigen::Vector3d x1 = R1 * (X - t1);
            Eigen::Vector3d x2 = R2 * (X - t2);
            x1 /= x1.z();
            x2 /= x2.z();

            correspondence cor;
            cor.measurement1 = x1;
            cor.measurement2 = x2;
            scene.correspondences.push_back(cor);
            scene.ground_truth_inliers.push_back(true);
        }

        for (size_t i = 0; i < n_outliers; i++)
        {
            correspondence cor;
            cor.measurement1 = Eigen::Vector3d(outlier_dist(rng), outlier_dist(rng), 1.0);
            cor.measurement2 = Eigen::Vector3d(outlier_dist(rng), outlier_dist(rng), 1.0);
            scene.correspondences.push_back(cor);
            scene.ground_truth_inliers.push_back(false);
        }

        return scene;
    }
};

struct BenchmarkResult
{
    double precision;
    double recall;
    double model_error;
    double time_ms;
};

template <typename Model> BenchmarkResult runBenchmark(const SyntheticScene &scene, int runs = 5)
{
    BenchmarkResult result{};
    double total_time = 0;

    for (int r = 0; r < runs; r++)
    {
        Model model;
        std::vector<bool> inliers;

        auto start = std::chrono::high_resolution_clock::now();
        ransac(scene.correspondences, model, inliers);
        auto end = std::chrono::high_resolution_clock::now();

        total_time += std::chrono::duration<double, std::milli>(end - start).count();

        if (r == 0)
        {
            size_t tp = 0, fp = 0, fn = 0;
            for (size_t i = 0; i < inliers.size(); i++)
            {
                if (inliers[i] && scene.ground_truth_inliers[i])
                    tp++;
                if (inliers[i] && !scene.ground_truth_inliers[i])
                    fp++;
                if (!inliers[i] && scene.ground_truth_inliers[i])
                    fn++;
            }
            result.precision = tp > 0 ? static_cast<double>(tp) / (tp + fp) : 0;
            result.recall = tp > 0 ? static_cast<double>(tp) / (tp + fn) : 0;

            Eigen::Matrix3d gt_norm = scene.ground_truth / scene.ground_truth.norm();
            Eigen::Matrix3d m_norm;
            if constexpr (std::is_same_v<Model, homography_model>)
                m_norm = model.homography / model.homography.norm();
            else if constexpr (std::is_same_v<Model, fundamental_matrix_model>)
                m_norm = model.fundamental_matrix / model.fundamental_matrix.norm();

            double err1 = (m_norm - gt_norm).norm();
            double err2 = (m_norm + gt_norm).norm();
            result.model_error = std::min(err1, err2);
        }
    }

    result.time_ms = total_time / runs;
    return result;
}

void printRow(const std::string &name, const BenchmarkResult &r)
{
    std::cout << std::left << std::setw(35) << name << "| " << std::fixed << std::setprecision(3) << std::setw(10)
              << r.precision << "| " << std::setw(10) << r.recall << "| " << std::scientific << std::setprecision(1)
              << std::setw(10) << r.model_error << "| " << std::fixed << std::setprecision(1) << std::setw(8)
              << r.time_ms << std::endl;
}

TEST(ransac_benchmark, homography_clean)
{
    auto scene = SyntheticScene::homography(200, 0, 42);
    auto result = runBenchmark<homography_model>(scene);
    printRow("H clean 200pt", result);

    EXPECT_GE(result.precision, 0.99);
    EXPECT_GE(result.recall, 0.99);
    EXPECT_LT(result.model_error, 1e-6);
}

TEST(ransac_benchmark, homography_30pct_outliers)
{
    auto scene = SyntheticScene::homography(140, 60, 42);
    auto result = runBenchmark<homography_model>(scene);
    printRow("H 30% outliers 200pt", result);

    EXPECT_GE(result.precision, 0.90);
    EXPECT_GE(result.recall, 0.85);
}

TEST(ransac_benchmark, homography_60pct_outliers)
{
    auto scene = SyntheticScene::homography(80, 120, 42);
    auto result = runBenchmark<homography_model>(scene);
    printRow("H 60% outliers 200pt", result);

    EXPECT_GE(result.precision, 0.80);
    EXPECT_GE(result.recall, 0.70);
}

TEST(ransac_benchmark, homography_80pct_outliers)
{
    auto scene = SyntheticScene::homography(40, 160, 42);
    auto result = runBenchmark<homography_model>(scene);
    printRow("H 80% outliers 200pt", result);

    EXPECT_GE(result.precision, 0.70);
    EXPECT_GE(result.recall, 0.60);
}

TEST(ransac_benchmark, homography_near_degenerate)
{
    SyntheticScene scene;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> noise(-0.001, 0.001);

    Eigen::Matrix3d R = Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    Eigen::Vector3d t(0.05, -0.03, 0.0);
    Eigen::Vector3d n(0, 0, 1);
    scene.ground_truth = R + t * n.transpose() / 10.0;
    scene.ground_truth /= scene.ground_truth(2, 2);

    for (int i = 0; i < 20; i++)
    {
        double t_param = -1.0 + 2.0 * i / 19.0;
        Eigen::Vector3d p1(t_param, 0.5 + noise(rng), 1.0);
        Eigen::Vector3d p2 = scene.ground_truth * p1;
        p2 /= p2.z();
        scene.correspondences.push_back(correspondence{p1, p2});
        scene.ground_truth_inliers.push_back(true);
    }

    std::uniform_real_distribution<double> point_dist(-1.0, 1.0);
    for (int i = 0; i < 80; i++)
    {
        Eigen::Vector3d p1(point_dist(rng), point_dist(rng), 1.0);
        Eigen::Vector3d p2 = scene.ground_truth * p1;
        p2 /= p2.z();
        scene.correspondences.push_back(correspondence{p1, p2});
        scene.ground_truth_inliers.push_back(true);
    }

    auto result = runBenchmark<homography_model>(scene);
    printRow("H near-degenerate 100pt", result);

    EXPECT_GE(result.precision, 0.95);
    EXPECT_GE(result.recall, 0.95);
    EXPECT_LT(result.model_error, 1e-6);
}

TEST(ransac_benchmark, fundamental_clean)
{
    auto scene = SyntheticScene::fundamental(200, 0, 0.0, 42);
    auto result = runBenchmark<fundamental_matrix_model>(scene);
    printRow("F clean 200pt", result);

    EXPECT_GE(result.precision, 0.95);
    EXPECT_GE(result.recall, 0.80);
}

TEST(ransac_benchmark, fundamental_30pct_outliers)
{
    auto scene = SyntheticScene::fundamental(140, 60, 0.0, 42);
    auto result = runBenchmark<fundamental_matrix_model>(scene);
    printRow("F 30% outliers 200pt", result);

    EXPECT_GE(result.precision, 0.85);
    EXPECT_GE(result.recall, 0.70);
}

TEST(ransac_benchmark, DISABLED_fundamental_80pct_outliers)
{
    auto scene = SyntheticScene::fundamental(40, 160, 0.0, 42);
    auto result = runBenchmark<fundamental_matrix_model>(scene);
    printRow("F 80% outliers 200pt", result);

    EXPECT_GE(result.precision, 0.50);
    EXPECT_GE(result.recall, 0.30);
}

TEST(ransac_benchmark, fundamental_dominant_plane)
{
    auto scene = SyntheticScene::fundamental(200, 0, 0.8, 42);
    auto result = runBenchmark<fundamental_matrix_model>(scene);
    printRow("F dominant plane 200pt", result);

    EXPECT_GE(result.precision, 0.95);
    EXPECT_GE(result.recall, 0.95);
}
