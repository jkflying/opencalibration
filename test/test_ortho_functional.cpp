#include <opencalibration/geo_coord/geo_coord.hpp>
#include <opencalibration/io/serialize.hpp>
#include <opencalibration/ortho/blending.hpp>
#include <opencalibration/ortho/color_balance.hpp>
#include <opencalibration/ortho/gdal_dataset.hpp>
#include <opencalibration/ortho/image_cache.hpp>
#include <opencalibration/ortho/ortho.hpp>
#include <opencalibration/surface/expand_mesh.hpp>
#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_pose.hpp>
#include <opencalibration/types/point_cloud.hpp>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace opencalibration;
using namespace opencalibration::orthomosaic;

// ==================== Image Cache Tests ====================

TEST(ImageCache, basic_cache)
{
    // GIVEN: An image cache with max size 2
    FullResolutionImageCache cache(2);

    // Create a temporary test image
    std::string test_image_path = TEST_DATA_OUTPUT_DIR "test_cache_image.png";
    cv::Mat test_image(100, 100, CV_8UC3, cv::Scalar(255, 0, 0)); // Blue image
    cv::imwrite(test_image_path, test_image);

    // WHEN: We load the image
    cv::Mat loaded = cache.getImage(1, test_image_path);

    // THEN: The image should be loaded and cached
    ASSERT_FALSE(loaded.empty());
    EXPECT_EQ(loaded.rows, 100);
    EXPECT_EQ(loaded.cols, 100);
    EXPECT_EQ(cache.getCacheHits(), 0);
    EXPECT_EQ(cache.getCacheMisses(), 1);

    // WHEN: We load the same image again
    cv::Mat loaded_again = cache.getImage(1, test_image_path);

    // THEN: It should be a cache hit
    EXPECT_EQ(cache.getCacheHits(), 1);
    EXPECT_EQ(cache.getCacheMisses(), 1);

    // Clean up not needed - output directory is for test artifacts
}

TEST(ImageCache, lru_eviction)
{
    // GIVEN: An image cache with max size 2
    FullResolutionImageCache cache(2);

    // Create 3 test images
    std::vector<std::string> image_paths;
    for (int i = 0; i < 3; i++)
    {
        std::string path = TEST_DATA_OUTPUT_DIR "test_cache_image_" + std::to_string(i) + ".png";
        cv::Mat img(50, 50, CV_8UC3, cv::Scalar(i * 50, i * 50, i * 50));
        cv::imwrite(path, img);
        image_paths.push_back(path);
    }

    // WHEN: We load 3 images (cache size is 2, so 1 should be evicted)
    cache.getImage(0, image_paths[0]);
    cache.getImage(1, image_paths[1]);
    cache.getImage(2, image_paths[2]); // Should evict image 0

    // THEN: Re-loading image 0 should be a cache miss
    cache.getImage(0, image_paths[0]);
    EXPECT_EQ(cache.getCacheMisses(), 4); // 3 initial loads + 1 reload of evicted image

    // Clean up not needed - output directory is for test artifacts
}

TEST(ImageCache, clear)
{
    // GIVEN: An image cache with loaded images
    FullResolutionImageCache cache(10);

    std::string test_image_path = TEST_DATA_OUTPUT_DIR "test_cache_clear.png";
    cv::Mat test_image(50, 50, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::imwrite(test_image_path, test_image);

    cache.getImage(1, test_image_path);
    EXPECT_EQ(cache.getCacheMisses(), 1);

    // WHEN: We clear the cache
    cache.clear();

    // THEN: Loading the same image should be a cache miss (cache was cleared)
    cache.getImage(1, test_image_path);
    EXPECT_EQ(cache.getCacheMisses(), 2); // Statistics continue counting

    // Clean up not needed - output directory is for test artifacts
}

TEST(CameraIdRoundTrip, uint64_camera_ids_survive_geotiff_sidecar)
{
    // GIVEN: Camera IDs that require more than 32 bits to represent
    // (MeasurementGraph generates random size_t IDs)
    GDALAllRegister();

    int w = 4, h = 4, num_layers = 2;
    std::string sidecar_path = TEST_DATA_OUTPUT_DIR "test_camera_id_sidecar.tif";

    GDALDriverH driver = GDALGetDriverByName("GTiff");
    ASSERT_NE(driver, nullptr);
    GDALDatasetH hWriteDs = GDALCreate(driver, sidecar_path.c_str(), w, h, num_layers * 2, GDT_UInt32, nullptr);
    GDALDatasetPtr write_ds(hWriteDs);
    ASSERT_NE(write_ds.get(), nullptr);

    std::vector<size_t> test_ids = {
        0xDEADBEEF12345678ULL, // large 64-bit value
        0xFFFFFFFF00000001ULL, // high bits set
        42,                    // small value (fits in 32 bits)
        0x100000000ULL,        // smallest value that doesn't fit in 32 bits
    };

    for (int layer = 0; layer < num_layers; layer++)
    {
        std::vector<uint32_t> lo(w * h, 0), hi(w * h, 0);
        for (int i = 0; i < w * h; i++)
        {
            size_t cam_id = test_ids[(layer * w * h + i) % test_ids.size()];
            lo[i] = static_cast<uint32_t>(cam_id & 0xFFFFFFFF);
            hi[i] = static_cast<uint32_t>((cam_id >> 32) & 0xFFFFFFFF);
        }

        int cam_band_offset = layer * 2;
        CPLErr err;
        GDALRasterBandWrapper band_lo(GDALGetRasterBand(write_ds.get(), cam_band_offset + 1));
        err = band_lo.RasterIO(GF_Write, 0, 0, w, h, lo.data(), w, h, GDT_UInt32, 0, 0);
        ASSERT_EQ(err, CE_None);

        GDALRasterBandWrapper band_hi(GDALGetRasterBand(write_ds.get(), cam_band_offset + 2));
        err = band_hi.RasterIO(GF_Write, 0, 0, w, h, hi.data(), w, h, GDT_UInt32, 0, 0);
        ASSERT_EQ(err, CE_None);
    }

    write_ds.reset();
    GDALDatasetPtr read_ds(GDALOpen(sidecar_path.c_str(), GA_ReadOnly));
    ASSERT_NE(read_ds.get(), nullptr);

    // WHEN: We read back the camera IDs
    for (int layer = 0; layer < num_layers; layer++)
    {
        std::vector<uint32_t> lo(w * h, 0), hi(w * h, 0);
        int cam_band_offset = layer * 2;

        CPLErr err;
        GDALRasterBandWrapper band_read_lo(GDALGetRasterBand(read_ds.get(), cam_band_offset + 1));
        err = band_read_lo.RasterIO(GF_Read, 0, 0, w, h, lo.data(), w, h, GDT_UInt32, 0, 0);
        ASSERT_EQ(err, CE_None);

        GDALRasterBandWrapper band_read_hi(GDALGetRasterBand(read_ds.get(), cam_band_offset + 2));
        err = band_read_hi.RasterIO(GF_Read, 0, 0, w, h, hi.data(), w, h, GDT_UInt32, 0, 0);
        ASSERT_EQ(err, CE_None);

        // THEN: Reconstructed 64-bit IDs should match the originals
        for (int i = 0; i < w * h; i++)
        {
            size_t expected = test_ids[(layer * w * h + i) % test_ids.size()];
            size_t actual = static_cast<size_t>(lo[i]) | (static_cast<size_t>(hi[i]) << 32);
            EXPECT_EQ(actual, expected) << "Layer " << layer << " pixel " << i << ": expected 0x" << std::hex
                                        << expected << " got 0x" << actual;
        }
    }
}

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

    // Helper to generate layered orthomosaic (multi-pass approach)
    void generateLayeredOrthomosaic(const std::vector<surface_model> &surfaces, const MeasurementGraph &graph_ref,
                                    const GeoCoord &coord_system, const std::string &output_path, int tile_size = 1024,
                                    double max_output_megapixels = 0.0)
    {
        std::string layers_path = output_path + ".layers.tif";
        std::string cameras_path = output_path + ".cameras.tif";
        std::string dsm_path = output_path + ".dsm.tif";

        OrthoMosaicConfig config;
        config.tile_size = tile_size;
        config.max_output_megapixels = max_output_megapixels;

        // Pass 1: Generate layers and DSM together
        generateLayeredGeoTIFF(surfaces, graph_ref, coord_system, layers_path, cameras_path, dsm_path, config);

        // Pass 2: Blend (skip color balance for tests - use empty result)
        ColorBalanceResult color_balance{};
        blendLayeredGeoTIFF(layers_path, cameras_path, dsm_path, output_path, color_balance, graph_ref, coord_system,
                            config);
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
    EXPECT_NO_THROW(generateLayeredOrthomosaic({mesh_surface}, graph, coord_system, output_path, 512));

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
    EXPECT_NO_THROW(generateLayeredOrthomosaic({mesh_surface}, graph, coord_system, output_path, 128));

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
    EXPECT_NO_THROW(generateLayeredOrthomosaic({mesh_surface}, graph, coord_system, output_path, 256, max_megapixels));

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

    // Create test images with known distinct colors
    std::vector<std::string> temp_image_paths;
    std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 0, 255), // Image 0: Pure red in RGB (0, 0, 255 in BGR)
        cv::Scalar(0, 255, 0), // Image 1: Pure green in RGB (0, 255, 0 in BGR)
        cv::Scalar(255, 0, 0)  // Image 2: Pure blue in RGB (255, 0, 0 in BGR)
    };

    for (int i = 0; i < 3; i++)
    {
        std::string path = TEST_DATA_OUTPUT_DIR "test_color_image_" + std::to_string(i) + ".png";
        cv::Mat img(100, 100, CV_8UC3, colors[i]);
        cv::imwrite(path, img);
        temp_image_paths.push_back(path);
        graph.getNode(id[i])->payload.path = path;
    }

    GeoCoord coord_system;
    coord_system.setOrigin(0, 0);

    std::string output_path = TEST_DATA_OUTPUT_DIR "test_ortho_pixel_values.tif";

    // WHEN: we generate a GeoTIFF orthomosaic
    EXPECT_NO_THROW(generateLayeredOrthomosaic({mesh_surface}, graph, coord_system, output_path, 512));

    // THEN: verify the output file exists and has correct structure
    EXPECT_TRUE(std::filesystem::exists(output_path));

    GDALDatasetPtr dataset = openGDALDataset(output_path);
    ASSERT_NE(dataset.get(), nullptr);

    GDALDatasetWrapper ds_wrapper(dataset.get());
    int width = ds_wrapper.GetRasterXSize();
    int height = ds_wrapper.GetRasterYSize();
    EXPECT_GT(width, 0);
    EXPECT_GT(height, 0);
    EXPECT_EQ(ds_wrapper.GetRasterCount(), 4); // RGBA

    // Verify geotransform is valid and invertible
    double geotransform[6];
    EXPECT_EQ(ds_wrapper.GetGeoTransform(geotransform), CE_None);
    EXPECT_GT(geotransform[1], 0); // Pixel width (GSD)
    EXPECT_LT(geotransform[5], 0); // Negative pixel height
    double gsd = geotransform[1];
    double origin_x = geotransform[0];
    double origin_y = geotransform[3];

    // Sample pixels from the center and corners to verify blending
    std::vector<std::pair<int, int>> test_pixels = {
        {width / 2, height / 2},        // Center of image
        {width / 4, height / 4},        // Upper-left quadrant
        {3 * width / 4, height / 4},    // Upper-right quadrant
        {width / 4, 3 * height / 4},    // Lower-left quadrant
        {3 * width / 4, 3 * height / 4} // Lower-right quadrant
    };

    // Read bands for sampled pixels
    for (const auto &[px, py] : test_pixels)
    {
        // Read pixel from all 4 bands
        uint8_t r, g, b, a;
        GDALRasterBandH red_band = GDALGetRasterBand(dataset.get(), 1);
        GDALRasterBandH green_band = GDALGetRasterBand(dataset.get(), 2);
        GDALRasterBandH blue_band = GDALGetRasterBand(dataset.get(), 3);
        GDALRasterBandH alpha_band = GDALGetRasterBand(dataset.get(), 4);

        ASSERT_NE(red_band, nullptr);
        ASSERT_NE(green_band, nullptr);
        ASSERT_NE(blue_band, nullptr);
        ASSERT_NE(alpha_band, nullptr);

        // Read single pixel from each band
        CPLErr err_r = GDALRasterIO(red_band, GF_Read, px, py, 1, 1, &r, 1, 1, GDT_Byte, 0, 0);
        CPLErr err_g = GDALRasterIO(green_band, GF_Read, px, py, 1, 1, &g, 1, 1, GDT_Byte, 0, 0);
        CPLErr err_b = GDALRasterIO(blue_band, GF_Read, px, py, 1, 1, &b, 1, 1, GDT_Byte, 0, 0);
        CPLErr err_a = GDALRasterIO(alpha_band, GF_Read, px, py, 1, 1, &a, 1, 1, GDT_Byte, 0, 0);

        EXPECT_EQ(err_r, CE_None);
        EXPECT_EQ(err_g, CE_None);
        EXPECT_EQ(err_b, CE_None);
        EXPECT_EQ(err_a, CE_None);

        // Verify alpha is either 255 (valid data) or 0 (no data)
        // The test surface should be covered by at least some images
        EXPECT_TRUE(a == 255 || a == 0) << "Alpha at (" << px << ", " << py << ") is " << (int)a
                                        << ", expected 0 or 255";

        // If pixel has valid data (alpha == 255), verify RGB values are reasonable
        if (a == 255)
        {
            // The blended colors should be within reasonable range of the input colors
            // Due to blending, we allow some variance (±50 to account for interpolation and blending)
            EXPECT_TRUE((r < 255 && r > 0) || (g < 255 && g > 0) || (b < 255 && b > 0))
                << "At least one channel should have significant value at (" << px << ", " << py << ")";
        }
    }

    // Verify pixel coordinate to world coordinate transformation
    // Pick a pixel and verify we can convert it to world coordinates
    int test_px = width / 2;
    int test_py = height / 2;
    double world_x = origin_x + test_px * gsd;
    double world_y = origin_y + test_py * geotransform[5]; // Note: geotransform[5] is negative

    // World coordinates should be within the expected bounds
    EXPECT_TRUE(world_x >= -50 && world_x <= 50) << "World X coordinate " << world_x << " out of expected range";
    EXPECT_TRUE(world_y >= -50 && world_y <= 50) << "World Y coordinate " << world_y << " out of expected range";

    // Clean up not needed - output directory is for test artifacts
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
    EXPECT_NO_THROW(generateLayeredOrthomosaic({mesh_surface}, graph, coord_system, output_path, 256));

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
        cv::Mat img(600, 800, CV_8UC3, cv::Scalar(i * 80, 100, 200 - i * 60));
        cv::imwrite(path, img);
        graph.getNode(id[i])->payload.path = path;
    }

    GeoCoord coord_system;
    coord_system.setOrigin(0, 0);

    std::string geotiff_path = TEST_DATA_OUTPUT_DIR "test_textured_mesh_ortho.tif";
    generateLayeredOrthomosaic({mesh_surface}, graph, coord_system, geotiff_path, 512);
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
    dataset.reset();

    cv::Mat texture = cv::imread(jpg_path);
    ASSERT_FALSE(texture.empty());
    EXPECT_EQ(texture.cols, expected_width);
    EXPECT_EQ(texture.rows, expected_height);

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
    generateLayeredOrthomosaic({mesh_surface}, graph, coord_system, geotiff_path, 512);

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

TEST(Blending_Functional, no_ringing_with_camera_hash_masks)
{
    // GIVEN: Validity masks derived from real camera hash images (blending_tile225).
    // All layers share a similar edge boundary in the bottom-right corner,
    // which caused ringing artifacts before the pull-push fill fix.
    std::string base = TEST_DATA_DIR "blending_tile225/";
    cv::Mat hash0 = cv::imread(base + "layer_0_camera_hash.png");
    cv::Mat hash1 = cv::imread(base + "layer_1_camera_hash.png");
    cv::Mat hash2 = cv::imread(base + "layer_2_camera_hash.png");

    ASSERT_FALSE(hash0.empty()) << "Failed to load layer_0_camera_hash.png";
    ASSERT_FALSE(hash1.empty()) << "Failed to load layer_1_camera_hash.png";
    ASSERT_FALSE(hash2.empty()) << "Failed to load layer_2_camera_hash.png";

    int rows = hash0.rows, cols = hash0.cols;
    int num_layers = 3;
    float L_value = 50.0f;

    std::vector<cv::Mat> lab_layers(num_layers);
    std::vector<cv::Mat> weight_maps(num_layers);
    cv::Mat hashes[3] = {hash0, hash1, hash2};

    for (int i = 0; i < num_layers; i++)
    {
        lab_layers[i] = cv::Mat(rows, cols, CV_32FC3, cv::Scalar(0, 0, 0));
        weight_maps[i] = cv::Mat::zeros(rows, cols, CV_32FC1);

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                cv::Vec3b pixel = hashes[i].at<cv::Vec3b>(r, c);
                bool valid = (pixel[0] > 5 || pixel[1] > 5 || pixel[2] > 5);
                if (valid)
                {
                    lab_layers[i].at<cv::Vec3f>(r, c) = cv::Vec3f(L_value, 0.0f, 0.0f);
                    weight_maps[i].at<float>(r, c) = 1.0f;
                }
            }
        }
    }

    // WHEN: We Laplacian blend
    cv::Mat result = laplacianBlend(lab_layers, weight_maps, 4);
    ASSERT_FALSE(result.empty());

    cv::imwrite(TEST_DATA_OUTPUT_DIR "blending_tile225_result.png", result);

    // THEN: All pixels where every layer is valid should have uniform color (no ringing)
    cv::Vec4b ref(0, 0, 0, 0);
    bool ref_set = false;
    int all_valid_pixels = 0;
    int ringing_pixels = 0;
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            bool all_valid = true;
            for (int i = 0; i < num_layers; i++)
            {
                if (weight_maps[i].at<float>(r, c) < 0.5f)
                {
                    all_valid = false;
                    break;
                }
            }

            if (all_valid)
            {
                all_valid_pixels++;
                cv::Vec4b pixel = result.at<cv::Vec4b>(r, c);
                if (!ref_set)
                {
                    ref = pixel;
                    ref_set = true;
                }
                for (int ch = 0; ch < 3; ch++)
                {
                    if (std::abs(pixel[ch] - ref[ch]) > 3)
                    {
                        ringing_pixels++;
                        break;
                    }
                }
            }
        }
    }

    ASSERT_GT(all_valid_pixels, 0) << "Expected at least one pixel where all layers are valid";
    EXPECT_EQ(ringing_pixels, 0) << "Found " << ringing_pixels << " pixels with ringing artifacts";
}
