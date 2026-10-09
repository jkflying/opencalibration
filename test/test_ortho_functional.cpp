#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/geo_coord/geo_coord.hpp>
#include <opencalibration/io/serialize.hpp>
#include <opencalibration/ortho/blending.hpp>
#include <opencalibration/ortho/color_balance.hpp>
#include <opencalibration/ortho/gdal_dataset.hpp>
#include <opencalibration/ortho/ortho.hpp>
#include <opencalibration/surface/expand_mesh.hpp>
#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_pose.hpp>
#include <opencalibration/types/point_cloud.hpp>

#include <cpl_string.h>
#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

using namespace opencalibration;
using namespace opencalibration::orthomosaic;

// ==================== GeoTIFF Generation Tests ====================

struct ortho : public ::testing::Test
{
    size_t id[3];
    MeasurementGraph graph;
    std::vector<NodePose> nodePoses;
    ankerl::unordered_dense::map<size_t, CameraModel> cam_models;
    std::shared_ptr<CameraModel> model;
    Eigen::Quaterniond ground_ori[3];
    Eigen::Vector3d ground_pos[3];

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
        model->focal_length_pixels = 100;
        model->principle_point << 50, 50;
        model->pixels_cols = 100;
        model->pixels_rows = 100;
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

    void generateOrthomosaicGeoTIFF(const std::vector<surface_model> &surfaces, const MeasurementGraph &graph_ref,
                                    const GeoCoord &coord_system, const std::string &output_path, int tile_size = 1024,
                                    double max_output_megapixels = 0.0)
    {
        OrthoMosaicConfig config;
        config.tile_size = tile_size;
        config.max_output_megapixels = max_output_megapixels;
        generateGeoTIFF(surfaces, graph_ref, coord_system, ColorBalanceResult{}, output_path, "", config);
    }

    struct RenderedGeoTIFF
    {
        int width = 0;
        int height = 0;
        double geotransform[6];
        std::vector<uint8_t> rgba;
        std::vector<float> dsm;
    };

    surface_model nearestCameraScene(const std::string &prefix)
    {
        surface_model points_surface;
        points_surface.cloud.push_back(generate_planar_points());
        point_cloud camera_locations;
        for (const auto &nodePose : nodePoses)
            camera_locations.push_back(nodePose.position);
        surface_model mesh_surface;
        mesh_surface.mesh = rebuildMesh(camera_locations, {points_surface});

        for (int i = 0; i < 3; i++)
        {
            std::string path = prefix + "_" + std::to_string(i) + ".png";
            cv::imwrite(path, cv::Mat(100, 100, CV_8UC3, cameraColorBGR(i)));
            graph.getNode(id[i])->payload.path = path;
        }
        return mesh_surface;
    }

    RenderedGeoTIFF renderNearestCameraScene(const std::string &prefix, int blend_transition_radius)
    {
        const surface_model mesh_surface = nearestCameraScene(prefix);
        GeoCoord coord_system;
        coord_system.setOrigin(0, 0);
        OrthoMosaicConfig config;
        config.tile_size = 64;
        config.blend_transition_radius = blend_transition_radius;
        generateGeoTIFF({mesh_surface}, graph, coord_system, ColorBalanceResult{}, prefix + ".tif", prefix + ".dsm.tif",
                        config);

        GDALDatasetPtr ds(GDALOpen((prefix + ".tif").c_str(), GA_ReadOnly));
        GDALDatasetPtr dsm_ds(GDALOpen((prefix + ".dsm.tif").c_str(), GA_ReadOnly));
        RenderedGeoTIFF out;
        if (!ds || !dsm_ds)
            return out;
        out.width = GDALGetRasterXSize(ds.get());
        out.height = GDALGetRasterYSize(ds.get());
        GDALGetGeoTransform(ds.get(), out.geotransform);
        out.rgba.resize(static_cast<size_t>(out.width) * out.height * 4);
        out.dsm.resize(static_cast<size_t>(out.width) * out.height);
        if (GDALDatasetRasterIO(ds.get(), GF_Read, 0, 0, out.width, out.height, out.rgba.data(), out.width, out.height,
                                GDT_Byte, 4, nullptr, 4, out.width * 4, 1) != CE_None ||
            GDALRasterIO(GDALGetRasterBand(dsm_ds.get(), 1), GF_Read, 0, 0, out.width, out.height, out.dsm.data(),
                         out.width, out.height, GDT_Float32, 0, 0) != CE_None)
            return {};
        return out;
    }

    static cv::Scalar cameraColorBGR(int cam)
    {
        return cv::Scalar(cam * 80, 100, 200);
    }
};

TEST_F(ortho, geotiff_creation)
{
    // GIVEN: A scene with images and surface
    init_cameras();

    surface_model points_surface;
    points_surface.cloud.push_back(generate_planar_points());

    point_cloud camera_locations;
    for (const auto &nodePose : nodePoses)
    {
        camera_locations.push_back(nodePose.position);
    }
    surface_model mesh_surface;
    mesh_surface.mesh = rebuildMesh(camera_locations, {points_surface});

    // Create temporary test images for the graph
    std::vector<std::string> temp_image_paths;
    for (int i = 0; i < 3; i++)
    {
        std::string path = TEST_DATA_OUTPUT_DIR "test_geotiff_image_" + std::to_string(i) + ".png";
        cv::Mat img(100, 100, CV_8UC3, cv::Scalar(i * 80, i * 80, i * 80));
        cv::imwrite(path, img);
        temp_image_paths.push_back(path);

        // Update the image path in the graph
        graph.getNode(id[i])->payload.path = path;
    }

    // Set up coordinate system
    GeoCoord coord_system;
    coord_system.setOrigin(0, 0);

    std::string output_path = TEST_DATA_OUTPUT_DIR "test_ortho_output.tif";

    // WHEN: we generate a GeoTIFF orthomosaic
    EXPECT_NO_THROW(generateOrthomosaicGeoTIFF({mesh_surface}, graph, coord_system, output_path, 512));

    // THEN: the file should exist
    EXPECT_TRUE(std::filesystem::exists(output_path));

    // Verify GeoTIFF properties using GDAL
    GDALDatasetPtr dataset = openGDALDataset(output_path);
    ASSERT_NE(dataset.get(), nullptr);

    GDALDatasetWrapper ds_wrapper(dataset.get());

    // Check dimensions
    EXPECT_GT(ds_wrapper.GetRasterXSize(), 0);
    EXPECT_GT(ds_wrapper.GetRasterYSize(), 0);

    // Check number of bands (should be 4: RGBA)
    EXPECT_EQ(ds_wrapper.GetRasterCount(), 4);

    // Check geotransform
    double geotransform[6];
    ds_wrapper.GetGeoTransform(geotransform);
    EXPECT_GT(geotransform[1], 0); // GSD (pixel width)
    EXPECT_LT(geotransform[5], 0); // Negative pixel height

    // Check projection (should have WKT)
    const char *projection = ds_wrapper.GetProjectionRef();
    EXPECT_NE(projection, nullptr);
    EXPECT_GT(strlen(projection), 0);

    // Check band color interpretation
    GDALRasterBandWrapper band1(ds_wrapper.GetRasterBand(1));
    GDALRasterBandWrapper band2(ds_wrapper.GetRasterBand(2));
    GDALRasterBandWrapper band3(ds_wrapper.GetRasterBand(3));
    GDALRasterBandWrapper band4(ds_wrapper.GetRasterBand(4));
    EXPECT_EQ(band1.GetColorInterpretation(), GCI_RedBand);
    EXPECT_EQ(band2.GetColorInterpretation(), GCI_GreenBand);
    EXPECT_EQ(band3.GetColorInterpretation(), GCI_BlueBand);
    EXPECT_EQ(band4.GetColorInterpretation(), GCI_AlphaBand);

    // Clean up not needed - output directory is for test artifacts
}

TEST_F(ortho, geotiff_small_tile_size)
{
    // GIVEN: A scene with images and surface
    init_cameras();

    surface_model points_surface;
    points_surface.cloud.push_back(generate_planar_points());

    point_cloud camera_locations;
    for (const auto &nodePose : nodePoses)
    {
        camera_locations.push_back(nodePose.position);
    }
    surface_model mesh_surface;
    mesh_surface.mesh = rebuildMesh(camera_locations, {points_surface});

    // Create temporary test images
    std::vector<std::string> temp_image_paths;
    for (int i = 0; i < 3; i++)
    {
        std::string path = TEST_DATA_OUTPUT_DIR "test_geotiff_small_" + std::to_string(i) + ".png";
        cv::Mat img(100, 100, CV_8UC3, cv::Scalar(255 - i * 80, 100, i * 80));
        cv::imwrite(path, img);
        temp_image_paths.push_back(path);
        graph.getNode(id[i])->payload.path = path;
    }

    GeoCoord coord_system;
    coord_system.setOrigin(0, 0);

    std::string output_path = TEST_DATA_OUTPUT_DIR "test_ortho_small_tile.tif";

    // WHEN: we generate a GeoTIFF with small tile size (should create multiple tiles)
    EXPECT_NO_THROW(generateOrthomosaicGeoTIFF({mesh_surface}, graph, coord_system, output_path, 128));

    // THEN: the file should exist and be valid
    EXPECT_TRUE(std::filesystem::exists(output_path));

    GDALDatasetPtr dataset = openGDALDataset(output_path);
    ASSERT_NE(dataset.get(), nullptr);

    GDALDatasetWrapper ds_wrapper(dataset.get());
    EXPECT_GT(ds_wrapper.GetRasterXSize(), 0);
    EXPECT_GT(ds_wrapper.GetRasterYSize(), 0);
    EXPECT_EQ(ds_wrapper.GetRasterCount(), 4);

    // Clean up not needed - output directory is for test artifacts
}

TEST_F(ortho, geotiff_without_feathering_uses_nearest_visible_camera_per_pixel)
{
    // GIVEN: a scene with differently coloured images, split into several tiles and blocks
    init_cameras();

    // WHEN: generating the GeoTIFF with no blend transition
    const auto out = renderNearestCameraScene(TEST_DATA_OUTPUT_DIR "test_nearest_camera", 0);

    // THEN: every pixel has the colour of the XY-nearest camera that sees it, with no block pattern
    ASSERT_GT(out.width, 64);
    auto sees = [&](int cam, const Eigen::Vector3d &point) {
        const Eigen::Matrix3d inv_rotation = ground_ori[cam].inverse().toRotationMatrix();
        if ((inv_rotation * (point - ground_pos[cam])).z() <= 0 ||
            (ground_pos[cam] - point).head<2>().norm() > ground_pos[cam].z() - point.z())
            return false;
        Eigen::Vector2d pixel = image_from_3d(point, *model, ground_pos[cam], inv_rotation);
        return pixel.x() >= 0 && pixel.x() < model->pixels_cols && pixel.y() >= 0 && pixel.y() < model->pixels_rows;
    };

    std::set<int> assigned_cameras;
    int mismatches = 0;
    for (int row = 0; row < out.height; row++)
    {
        for (int col = 0; col < out.width; col++)
        {
            const int i = row * out.width + col;
            if (std::isnan(out.dsm[i]))
                continue;
            const Eigen::Vector3d point(out.geotransform[0] + (col + 0.5) * out.geotransform[1],
                                        out.geotransform[3] + (row + 0.5) * out.geotransform[5], out.dsm[i]);

            std::array<int, 3> order{0, 1, 2};
            std::sort(order.begin(), order.end(), [&](int l, int r) {
                return (ground_pos[l] - point).head<2>().squaredNorm() <
                       (ground_pos[r] - point).head<2>().squaredNorm();
            });
            auto nearest = std::find_if(order.begin(), order.end(), [&](int cam) { return sees(cam, point); });

            const uint8_t *rgba = &out.rgba[static_cast<size_t>(i) * 4];
            if (nearest == order.end())
            {
                mismatches += rgba[3] != 0;
                continue;
            }
            assigned_cameras.insert(*nearest);
            const cv::Scalar bgr = cameraColorBGR(*nearest);
            mismatches += rgba[3] != 255 || std::abs(rgba[0] - bgr[2]) > 2 || std::abs(rgba[1] - bgr[1]) > 2 ||
                          std::abs(rgba[2] - bgr[0]) > 2;
        }
    }
    EXPECT_EQ(assigned_cameras, (std::set<int>{0, 1, 2}));
    EXPECT_EQ(mismatches, 0);
}

TEST_F(ortho, geotiff_does_not_see_through_the_mesh)
{
    // GIVEN: a red camera just west of a 30m ridge, and a blue camera further east
    auto cam_model = std::make_shared<CameraModel>();
    cam_model->focal_length_pixels = 200;
    cam_model->principle_point << 200, 150;
    cam_model->pixels_cols = 400;
    cam_model->pixels_rows = 300;
    cam_model->projection_type = opencalibration::ProjectionType::PLANAR;
    MeasurementGraph ridge_graph;
    const std::string prefix = TEST_DATA_OUTPUT_DIR "test_ridge_occlusion";
    const std::vector<std::pair<Eigen::Vector3d, cv::Scalar>> cameras{{{-8, 0, 40}, cv::Scalar(0, 0, 255)},
                                                                      {{25, 0, 40}, cv::Scalar(255, 0, 0)}};
    for (size_t i = 0; i < cameras.size(); i++)
    {
        image img;
        img.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
        img.position = cameras[i].first;
        img.model = cam_model;
        img.path = prefix + "_" + std::to_string(i) + ".png";
        cv::imwrite(img.path, cv::Mat(cam_model->pixels_rows, cam_model->pixels_cols, CV_8UC3, cameras[i].second));
        ridge_graph.addNode(std::move(img));
    }

    surface_model ground;
    ground.cloud.push_back({Eigen::Vector3d(0, 0, 0)});
    surface_model surface;
    surface.mesh = rebuildMesh({Eigen::Vector3d(0, 0, 40), Eigen::Vector3d(1, 0, 40)}, {ground});
    for (auto it = surface.mesh.nodebegin(); it != surface.mesh.nodeend(); ++it)
    {
        Eigen::Vector3d &location = it->second.payload.location;
        location.z() = std::max(0.0, 30 * (1 - std::abs(location.x()) / 1.5));
    }

    // WHEN: generating the GeoTIFF with no blend transition
    GeoCoord coord_system;
    coord_system.setOrigin(0, 0);
    OrthoMosaicConfig config;
    config.tile_size = 256;
    config.blend_transition_radius = 0;
    generateGeoTIFF({surface}, ridge_graph, coord_system, ColorBalanceResult{}, prefix + ".tif", "", config);

    // THEN: ground east of the ridge, hidden from the nearer red camera, is coloured by the blue camera
    GDALDatasetPtr ds(GDALOpen((prefix + ".tif").c_str(), GA_ReadOnly));
    ASSERT_TRUE(ds);
    double geotransform[6];
    GDALGetGeoTransform(ds.get(), geotransform);
    auto rgbaAt = [&](double x, double y) {
        const int col = static_cast<int>(std::floor((x - geotransform[0]) / geotransform[1]));
        const int row = static_cast<int>(std::floor((y - geotransform[3]) / geotransform[5]));
        std::array<uint8_t, 4> rgba{};
        EXPECT_EQ(
            GDALDatasetRasterIO(ds.get(), GF_Read, col, row, 1, 1, rgba.data(), 1, 1, GDT_Byte, 4, nullptr, 4, 4, 1),
            CE_None);
        return rgba;
    };
    for (double y = -4; y <= 4; y += 1)
    {
        for (double x = 2; x <= 6; x += 0.5)
        {
            const auto rgba = rgbaAt(x, y);
            EXPECT_EQ(rgba[3], 255) << x << " " << y;
            EXPECT_LT(rgba[0], 30) << x << " " << y;
            EXPECT_GT(rgba[2], 225) << x << " " << y;
        }

        // AND: ground west of the ridge is still coloured by the nearer red camera
        for (double x = -6; x <= -2; x += 0.5)
        {
            const auto rgba = rgbaAt(x, y);
            EXPECT_GT(rgba[0], 225) << x << " " << y;
            EXPECT_LT(rgba[2], 30) << x << " " << y;
        }
    }
}

TEST_F(ortho, geotiff_feathers_colours_across_camera_seams)
{
    // GIVEN: a scene with differently coloured images
    init_cameras();

    // WHEN: generating the GeoTIFF with a blend transition
    const auto out = renderNearestCameraScene(TEST_DATA_OUTPUT_DIR "test_feathered_camera", 64);

    // THEN: some pixels are a mix of cameras, and every blue value stays within the camera colour range
    ASSERT_GT(out.width, 0);
    int mixed_pixels = 0, out_of_range = 0;
    for (size_t i = 0; i < out.dsm.size(); i++)
    {
        const uint8_t *rgba = &out.rgba[i * 4];
        if (rgba[3] != 255)
            continue;
        const int blue = rgba[2];
        mixed_pixels += std::abs(blue - 0) > 2 && std::abs(blue - 80) > 2 && std::abs(blue - 160) > 2;
        out_of_range += blue > 162;
    }
    EXPECT_GT(mixed_pixels, 0);
    EXPECT_EQ(out_of_range, 0);
}

TEST_F(ortho, geotiff_writes_dsm_without_orthomosaic)
{
    // GIVEN: a scene and only a DSM output path
    init_cameras();
    const std::string dsm_path = TEST_DATA_OUTPUT_DIR "test_dsm_only.dsm.tif";
    std::remove(dsm_path.c_str());
    const surface_model mesh_surface = nearestCameraScene(TEST_DATA_OUTPUT_DIR "test_dsm_only");
    GeoCoord coord_system;
    coord_system.setOrigin(0, 0);

    // WHEN: we generate with an empty orthomosaic path
    generateGeoTIFF({mesh_surface}, graph, coord_system, ColorBalanceResult{}, "", dsm_path);

    // THEN: the DSM is written
    GDALDatasetPtr dsm_ds(GDALOpen(dsm_path.c_str(), GA_ReadOnly));
    ASSERT_TRUE(dsm_ds);
    EXPECT_GT(GDALGetRasterXSize(dsm_ds.get()), 0);
}

TEST_F(ortho, geotiff_respects_max_megapixel_limit)
{
    // GIVEN: A scene with images and surface
    init_cameras();

    surface_model points_surface;
    points_surface.cloud.push_back(generate_planar_points());

    point_cloud camera_locations;
    for (const auto &nodePose : nodePoses)
    {
        camera_locations.push_back(nodePose.position);
    }
    surface_model mesh_surface;
    mesh_surface.mesh = rebuildMesh(camera_locations, {points_surface});

    // Create temporary test images
    for (int i = 0; i < 3; i++)
    {
        std::string path = TEST_DATA_OUTPUT_DIR "test_geotiff_capped_" + std::to_string(i) + ".png";
        cv::Mat img(100, 100, CV_8UC3, cv::Scalar(60 + i * 60, 120, 180));
        cv::imwrite(path, img);
        graph.getNode(id[i])->payload.path = path;
    }

    GeoCoord coord_system;
    coord_system.setOrigin(0, 0);

    std::string output_path = TEST_DATA_OUTPUT_DIR "test_ortho_capped.tif";
    constexpr double max_megapixels = 0.01; // 10k pixels

    // WHEN: we generate a GeoTIFF with a strict output megapixel cap
    EXPECT_NO_THROW(generateOrthomosaicGeoTIFF({mesh_surface}, graph, coord_system, output_path, 256, max_megapixels));

    // THEN: output dimensions should respect the requested cap
    GDALDatasetPtr dataset = openGDALDataset(output_path);
    ASSERT_NE(dataset.get(), nullptr);

    GDALDatasetWrapper ds_wrapper(dataset.get());
    uint64_t output_pixels =
        static_cast<uint64_t>(ds_wrapper.GetRasterXSize()) * static_cast<uint64_t>(ds_wrapper.GetRasterYSize());
    uint64_t max_pixels = static_cast<uint64_t>(max_megapixels * 1000000.0);

    EXPECT_LE(output_pixels, max_pixels);
}

TEST_F(ortho, pixel_values_with_known_colors)
{
    // GIVEN: A scene with distinct colored images to verify pixel lookup and blending
    init_cameras();

    surface_model points_surface;
    points_surface.cloud.push_back(generate_planar_points());

    point_cloud camera_locations;
    for (const auto &nodePose : nodePoses)
    {
        camera_locations.push_back(nodePose.position);
    }
    surface_model mesh_surface;
    mesh_surface.mesh = rebuildMesh(camera_locations, {points_surface});

    const auto groundColor = [](const Eigen::Vector3d &p) {
        return cv::Vec3b(p.x() < 10 && p.y() >= 10 ? 255 : 0, p.x() >= 10 && p.y() < 10 ? 255 : 0,
                         p.x() < 10 && p.y() < 10 ? 255 : 0);
    };
    for (int i = 0; i < 3; i++)
    {
        cv::Mat img(100, 100, CV_8UC3);
        for (int row = 0; row < img.rows; row++)
            for (int col = 0; col < img.cols; col++)
            {
                const Eigen::Vector3d ray = ground_ori[i] * image_to_3d(Eigen::Vector2d(col + 0.5, row + 0.5), *model);
                img.at<cv::Vec3b>(row, col) = groundColor(ground_pos[i] + ray * (-10 - ground_pos[i].z()) / ray.z());
            }
        std::string path = TEST_DATA_OUTPUT_DIR "test_color_image_" + std::to_string(i) + ".png";
        cv::imwrite(path, img);
        graph.getNode(id[i])->payload.path = path;
    }

    GeoCoord coord_system;
    coord_system.setOrigin(0, 0);

    std::string output_path = TEST_DATA_OUTPUT_DIR "test_ortho_pixel_values.tif";

    // WHEN: we generate a GeoTIFF orthomosaic
    EXPECT_NO_THROW(generateOrthomosaicGeoTIFF({mesh_surface}, graph, coord_system, output_path, 512));

    // THEN: the mosaic reproduces the ground pattern away from its edges
    GDALDatasetPtr dataset = openGDALDataset(output_path);
    ASSERT_NE(dataset.get(), nullptr);
    ASSERT_EQ(GDALGetRasterCount(dataset.get()), 4);

    double geotransform[6];
    ASSERT_EQ(GDALGetGeoTransform(dataset.get(), geotransform), CE_None);

    for (const Eigen::Vector2d &ground :
         {Eigen::Vector2d(7, 7), Eigen::Vector2d(13, 7), Eigen::Vector2d(7, 13), Eigen::Vector2d(13, 13)})
    {
        const int px = static_cast<int>((ground.x() - geotransform[0]) / geotransform[1]);
        const int py = static_cast<int>((ground.y() - geotransform[3]) / geotransform[5]);

        std::array<uint8_t, 4> rgba{};
        for (int band = 0; band < 4; band++)
            ASSERT_EQ(GDALRasterIO(GDALGetRasterBand(dataset.get(), band + 1), GF_Read, px, py, 1, 1, &rgba[band], 1, 1,
                                   GDT_Byte, 0, 0),
                      CE_None);

        const cv::Vec3b bgr = groundColor({ground.x(), ground.y(), -10});
        EXPECT_EQ(rgba[3], 255) << ground.transpose();
        EXPECT_NEAR(rgba[0], bgr[2], 10) << ground.transpose();
        EXPECT_NEAR(rgba[1], bgr[1], 10) << ground.transpose();
        EXPECT_NEAR(rgba[2], bgr[0], 10) << ground.transpose();
    }
}

TEST_F(ortho, single_image_coverage)
{
    // GIVEN: A scene where only one image covers the surface
    init_cameras();

    surface_model points_surface;
    points_surface.cloud.push_back(generate_planar_points());

    point_cloud camera_locations;
    for (const auto &nodePose : nodePoses)
    {
        camera_locations.push_back(nodePose.position);
    }
    surface_model mesh_surface;
    mesh_surface.mesh = rebuildMesh(camera_locations, {points_surface});

    // Create distinct colored test images
    std::vector<std::string> temp_image_paths;
    std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 0, 255), // Image 0: Pure red
        cv::Scalar(0, 255, 0), // Image 1: Pure green
        cv::Scalar(255, 0, 0)  // Image 2: Pure blue
    };

    for (int i = 0; i < 3; i++)
    {
        std::string path = TEST_DATA_OUTPUT_DIR "test_single_coverage_" + std::to_string(i) + ".png";
        cv::Mat img(100, 100, CV_8UC3, colors[i]);
        cv::imwrite(path, img);
        temp_image_paths.push_back(path);
        graph.getNode(id[i])->payload.path = path;
    }

    GeoCoord coord_system;
    coord_system.setOrigin(0, 0);

    std::string output_path = TEST_DATA_OUTPUT_DIR "test_ortho_single_coverage.tif";

    // WHEN: we generate a GeoTIFF orthomosaic
    EXPECT_NO_THROW(generateOrthomosaicGeoTIFF({mesh_surface}, graph, coord_system, output_path, 256));

    // THEN: verify that the output has valid data
    EXPECT_TRUE(std::filesystem::exists(output_path));

    GDALDatasetPtr dataset = openGDALDataset(output_path);
    ASSERT_NE(dataset.get(), nullptr);

    GDALDatasetWrapper ds_wrapper(dataset.get());
    int width = ds_wrapper.GetRasterXSize();
    int height = ds_wrapper.GetRasterYSize();
    EXPECT_GT(width, 0);
    EXPECT_GT(height, 0);

    // Count valid pixels (alpha == 255) in the center region
    GDALRasterBandH alpha_band = GDALGetRasterBand(dataset.get(), 4);
    ASSERT_NE(alpha_band, nullptr);

    int center_x = width / 4;
    int center_y = height / 4;
    int region_size = std::min(width, height) / 4;

    std::vector<uint8_t> alpha_data(region_size * region_size);
    CPLErr err = GDALRasterIO(alpha_band, GF_Read, center_x, center_y, region_size, region_size, alpha_data.data(),
                              region_size, region_size, GDT_Byte, 0, 0);

    EXPECT_EQ(err, CE_None);

    // At least some pixels in the center should be valid (covered by cameras)
    int valid_pixel_count = 0;
    for (uint8_t alpha : alpha_data)
    {
        if (alpha == 255)
            valid_pixel_count++;
    }

    EXPECT_GT(valid_pixel_count, 0) << "Expected some valid pixels in center region, got 0";
    EXPECT_LE(valid_pixel_count, region_size * region_size) << "Valid pixel count exceeds region size";

    // Clean up not needed - output directory is for test artifacts
}

TEST(ortho_texture, textured_obj_downscales_texture_beyond_jpeg_limit)
{
    // GIVEN: an RGBA GeoTIFF wider than a JPEG can store
    GDALAllRegister();
    const std::string geotiff_path = TEST_DATA_OUTPUT_DIR "test_wide_texture.tif";
    const std::string jpg_path = TEST_DATA_OUTPUT_DIR "test_wide_texture.jpg";
    std::filesystem::remove(jpg_path);
    {
        char **options = CSLSetNameValue(nullptr, "SPARSE_OK", "YES");
        GDALDatasetPtr ds(
            GDALCreate(GDALGetDriverByName("GTiff"), geotiff_path.c_str(), 70000, 2, 4, GDT_Byte, options));
        CSLDestroy(options);
        ASSERT_TRUE(ds);
        double geotransform[6] = {0, 0.1, 0, 0.2, 0, -0.1};
        GDALSetGeoTransform(ds.get(), geotransform);
    }

    // WHEN: we export a textured OBJ from it
    generateTexturedOBJ({}, geotiff_path, TEST_DATA_OUTPUT_DIR "test_wide_texture.obj");

    // THEN: a texture is written within the JPEG size limit
    const cv::Mat texture = cv::imread(jpg_path);
    ASSERT_FALSE(texture.empty());
    EXPECT_LE(texture.cols, 65500);
}

TEST_F(ortho, textured_obj_export)
{
    // GIVEN: A scene with images, surface, and a generated orthomosaic GeoTIFF
    init_cameras();

    surface_model points_surface;
    points_surface.cloud.push_back(generate_planar_points());

    point_cloud camera_locations;
    for (const auto &nodePose : nodePoses)
    {
        camera_locations.push_back(nodePose.position);
    }
    surface_model mesh_surface;
    mesh_surface.mesh = rebuildMesh(camera_locations, {points_surface});

    for (int i = 0; i < 3; i++)
    {
        std::string path = TEST_DATA_OUTPUT_DIR "test_textured_mesh_image_" + std::to_string(i) + ".png";
        const cv::Scalar red_dominant_bgr(i * 30, 100, 220 - i * 20);
        cv::Mat img(600, 800, CV_8UC3, red_dominant_bgr);
        cv::imwrite(path, img);
        graph.getNode(id[i])->payload.path = path;
    }

    GeoCoord coord_system;
    coord_system.setOrigin(0, 0);

    std::string geotiff_path = TEST_DATA_OUTPUT_DIR "test_textured_mesh_ortho.tif";
    generateOrthomosaicGeoTIFF({mesh_surface}, graph, coord_system, geotiff_path, 512);
    ASSERT_TRUE(std::filesystem::exists(geotiff_path));

    // WHEN: we generate a textured OBJ from the mesh and GeoTIFF
    std::string obj_path = TEST_DATA_OUTPUT_DIR "test_textured_mesh.obj";
    std::string mtl_path = TEST_DATA_OUTPUT_DIR "test_textured_mesh.mtl";
    std::string jpg_path = TEST_DATA_OUTPUT_DIR "test_textured_mesh.jpg";

    EXPECT_NO_THROW(generateTexturedOBJ({mesh_surface}, geotiff_path, obj_path));

    // THEN: all three output files should exist
    EXPECT_TRUE(std::filesystem::exists(obj_path));
    EXPECT_TRUE(std::filesystem::exists(mtl_path));
    EXPECT_TRUE(std::filesystem::exists(jpg_path));

    // Verify JPEG texture dimensions match GeoTIFF
    GDALDatasetPtr dataset = openGDALDataset(geotiff_path);
    ASSERT_NE(dataset.get(), nullptr);
    GDALDatasetWrapper ds(dataset.get());
    int expected_width = ds.GetRasterXSize();
    int expected_height = ds.GetRasterYSize();
    cv::Mat ortho_rgb(expected_height, expected_width, CV_8UC3);
    ASSERT_EQ(GDALDatasetRasterIO(dataset.get(), GF_Read, 0, 0, expected_width, expected_height, ortho_rgb.data,
                                  expected_width, expected_height, GDT_Byte, 3, nullptr, 3,
                                  static_cast<int>(ortho_rgb.step), 1),
              CE_None);
    dataset.reset();

    cv::Mat texture = cv::imread(jpg_path);
    ASSERT_FALSE(texture.empty());
    EXPECT_EQ(texture.cols, expected_width);
    EXPECT_EQ(texture.rows, expected_height);

    // AND: the texture has the GeoTIFF's colours in the same channels
    cv::Mat texture_rgb;
    cv::cvtColor(texture, texture_rgb, cv::COLOR_BGR2RGB);
    const cv::Scalar ortho_mean = cv::mean(ortho_rgb), texture_mean = cv::mean(texture_rgb);
    for (int c = 0; c < 3; c++)
        EXPECT_NEAR(texture_mean[c], ortho_mean[c], 2);
    EXPECT_GT(std::abs(ortho_mean[0] - ortho_mean[2]), 10) << "scene must distinguish R from B";

    // Verify OBJ file contents
    std::ifstream obj_file(obj_path);
    ASSERT_TRUE(obj_file.is_open());
    std::string obj_contents((std::istreambuf_iterator<char>(obj_file)), std::istreambuf_iterator<char>());
    obj_file.close();

    // Count vertices, texture coordinates, and faces
    int vertex_count = 0, vt_count = 0, face_count = 0;
    bool has_mtllib = false, has_usemtl = false;
    std::istringstream obj_stream(obj_contents);
    std::string line;
    while (std::getline(obj_stream, line))
    {
        if (line.substr(0, 2) == "v ")
            vertex_count++;
        else if (line.substr(0, 3) == "vt ")
            vt_count++;
        else if (line.substr(0, 2) == "f ")
            face_count++;
        else if (line.find("mtllib") != std::string::npos)
            has_mtllib = true;
        else if (line.find("usemtl") != std::string::npos)
            has_usemtl = true;
    }

    EXPECT_GT(vertex_count, 0) << "OBJ should contain vertices";
    EXPECT_EQ(vt_count, vertex_count) << "Each vertex should have a texture coordinate";
    EXPECT_GT(face_count, 0) << "OBJ should contain faces";
    EXPECT_TRUE(has_mtllib) << "OBJ should reference an MTL file";
    EXPECT_TRUE(has_usemtl) << "OBJ should use a material";

    // Verify UV coordinates are within [0, 1] range
    std::istringstream uv_stream(obj_contents);
    while (std::getline(uv_stream, line))
    {
        if (line.substr(0, 3) == "vt ")
        {
            double u, v;
            ASSERT_EQ(sscanf(line.c_str(), "vt %lf %lf", &u, &v), 2);
            EXPECT_GE(u, -0.1) << "UV u coordinate should be near [0,1]: " << u;
            EXPECT_LE(u, 1.1) << "UV u coordinate should be near [0,1]: " << u;
            EXPECT_GE(v, -0.1) << "UV v coordinate should be near [0,1]: " << v;
            EXPECT_LE(v, 1.1) << "UV v coordinate should be near [0,1]: " << v;
        }
    }

    // Verify face references are valid (v/vt format with indices in range)
    std::istringstream face_stream(obj_contents);
    while (std::getline(face_stream, line))
    {
        if (line.substr(0, 2) == "f ")
        {
            int v0, vt0, v1, vt1, v2, vt2;
            ASSERT_EQ(sscanf(line.c_str(), "f %d/%d %d/%d %d/%d", &v0, &vt0, &v1, &vt1, &v2, &vt2), 6)
                << "Face line should have v/vt format: " << line;
            EXPECT_GE(v0, 1);
            EXPECT_LE(v0, vertex_count);
            EXPECT_GE(v1, 1);
            EXPECT_LE(v1, vertex_count);
            EXPECT_GE(v2, 1);
            EXPECT_LE(v2, vertex_count);
        }
    }

    // Verify MTL file references the JPEG texture
    std::ifstream mtl_file(mtl_path);
    ASSERT_TRUE(mtl_file.is_open());
    std::string mtl_contents((std::istreambuf_iterator<char>(mtl_file)), std::istreambuf_iterator<char>());
    mtl_file.close();

    EXPECT_NE(mtl_contents.find("test_textured_mesh.jpg"), std::string::npos)
        << "MTL should reference the JPEG texture file";
    EXPECT_NE(mtl_contents.find("newmtl"), std::string::npos) << "MTL should define a material";
}

TEST_F(ortho, obj_and_ply_mesh_geometry_match)
{
    // GIVEN: A mesh, and both PLY and OBJ exports of it
    init_cameras();

    surface_model points_surface;
    points_surface.cloud.push_back(generate_planar_points());

    point_cloud camera_locations;
    for (const auto &nodePose : nodePoses)
    {
        camera_locations.push_back(nodePose.position);
    }
    surface_model mesh_surface;
    mesh_surface.mesh = rebuildMesh(camera_locations, {points_surface});

    for (int i = 0; i < 3; i++)
    {
        std::string path = TEST_DATA_OUTPUT_DIR "test_mesh_compare_image_" + std::to_string(i) + ".png";
        cv::Mat img(600, 800, CV_8UC3, cv::Scalar(i * 80, 100, 200 - i * 60));
        cv::imwrite(path, img);
        graph.getNode(id[i])->payload.path = path;
    }

    GeoCoord coord_system;
    coord_system.setOrigin(0, 0);

    std::string geotiff_path = TEST_DATA_OUTPUT_DIR "test_mesh_compare_ortho.tif";
    generateOrthomosaicGeoTIFF({mesh_surface}, graph, coord_system, geotiff_path, 512);

    // Write PLY
    std::string ply_path = TEST_DATA_OUTPUT_DIR "test_mesh_compare.ply";
    {
        std::ofstream ply_out(ply_path, std::ios::binary);
        ASSERT_TRUE(ply_out.is_open());
        serialize(mesh_surface.mesh, ply_out);
    }

    // Write OBJ
    std::string obj_path = TEST_DATA_OUTPUT_DIR "test_mesh_compare.obj";
    generateTexturedOBJ({mesh_surface}, geotiff_path, obj_path);

    ASSERT_TRUE(std::filesystem::exists(ply_path));
    ASSERT_TRUE(std::filesystem::exists(obj_path));

    // Parse PLY vertices and faces
    struct Vec3
    {
        double x, y, z;
        bool operator==(const Vec3 &o) const
        {
            return std::abs(x - o.x) < 1e-6 && std::abs(y - o.y) < 1e-6 && std::abs(z - o.z) < 1e-6;
        }
    };

    std::vector<Vec3> ply_vertices;
    std::vector<std::array<int, 3>> ply_faces;
    {
        std::ifstream ply_file(ply_path);
        ASSERT_TRUE(ply_file.is_open());
        std::string line;

        // Parse header for counts
        size_t num_vertices = 0, num_faces = 0;
        while (std::getline(ply_file, line))
        {
            if (sscanf(line.c_str(), "element vertex %zu", &num_vertices) == 1)
                continue;
            if (sscanf(line.c_str(), "element face %zu", &num_faces) == 1)
                continue;
            if (line == "end_header")
                break;
        }
        ASSERT_GT(num_vertices, 0u);
        ASSERT_GT(num_faces, 0u);

        // Read vertices
        for (size_t i = 0; i < num_vertices; i++)
        {
            ASSERT_TRUE(std::getline(ply_file, line)) << "Expected vertex line " << i;
            double x, y, z;
            int node_id;
            ASSERT_EQ(sscanf(line.c_str(), "%lf %lf %lf %d", &x, &y, &z, &node_id), 4)
                << "Failed to parse PLY vertex: " << line;
            ply_vertices.push_back({x, y, z});
        }

        // Read exactly num_faces faces (edges follow after)
        for (size_t i = 0; i < num_faces; i++)
        {
            ASSERT_TRUE(std::getline(ply_file, line)) << "Expected face line " << i;
            int count, v0, v1, v2;
            ASSERT_EQ(sscanf(line.c_str(), "%d %d %d %d", &count, &v0, &v1, &v2), 4)
                << "Failed to parse PLY face: " << line;
            ASSERT_EQ(count, 3);
            ply_faces.push_back({v0, v1, v2});
        }
    }

    // Parse OBJ vertices and faces
    std::vector<Vec3> obj_vertices;
    std::vector<std::array<int, 3>> obj_faces;
    {
        std::ifstream obj_file(obj_path);
        ASSERT_TRUE(obj_file.is_open());
        std::string line;
        while (std::getline(obj_file, line))
        {
            if (line.substr(0, 2) == "v ")
            {
                double x, y, z;
                ASSERT_EQ(sscanf(line.c_str(), "v %lf %lf %lf", &x, &y, &z), 3)
                    << "Failed to parse OBJ vertex: " << line;
                obj_vertices.push_back({x, y, z});
            }
            else if (line.substr(0, 2) == "f ")
            {
                int v0, vt0, v1, vt1, v2, vt2;
                ASSERT_EQ(sscanf(line.c_str(), "f %d/%d %d/%d %d/%d", &v0, &vt0, &v1, &vt1, &v2, &vt2), 6)
                    << "Failed to parse OBJ face: " << line;
                // Convert OBJ 1-based to 0-based for comparison
                obj_faces.push_back({v0 - 1, v1 - 1, v2 - 1});
            }
        }
    }

    // Compare vertex counts
    ASSERT_EQ(ply_vertices.size(), obj_vertices.size())
        << "PLY has " << ply_vertices.size() << " vertices, OBJ has " << obj_vertices.size();

    // Compare vertex positions (both sorted by node ID, so should be in same order)
    for (size_t i = 0; i < ply_vertices.size(); i++)
    {
        EXPECT_TRUE(ply_vertices[i] == obj_vertices[i])
            << "Vertex " << i << " differs: PLY=(" << ply_vertices[i].x << ", " << ply_vertices[i].y << ", "
            << ply_vertices[i].z << ") OBJ=(" << obj_vertices[i].x << ", " << obj_vertices[i].y << ", "
            << obj_vertices[i].z << ")";
    }

    // Compare face counts
    ASSERT_EQ(ply_faces.size(), obj_faces.size())
        << "PLY has " << ply_faces.size() << " faces, OBJ has " << obj_faces.size();

    // Compare faces - both are sorted deterministically, so should match directly
    for (size_t i = 0; i < ply_faces.size(); i++)
    {
        EXPECT_EQ(ply_faces[i], obj_faces[i])
            << "Face " << i << " differs: PLY=(" << ply_faces[i][0] << ", " << ply_faces[i][1] << ", "
            << ply_faces[i][2] << ") OBJ=(" << obj_faces[i][0] << ", " << obj_faces[i][1] << ", " << obj_faces[i][2]
            << ")";
    }
}
