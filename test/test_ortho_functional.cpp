#include <opencalibration/geo_coord/geo_coord.hpp>
#include <opencalibration/ortho/gdal_dataset.hpp>
#include <opencalibration/ortho/image_cache.hpp>
#include <opencalibration/ortho/ortho.hpp>
#include <opencalibration/surface/expand_mesh.hpp>
#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_pose.hpp>
#include <opencalibration/types/point_cloud.hpp>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include <filesystem>

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

    GDALDriver *driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    ASSERT_NE(driver, nullptr);
    GDALDatasetPtr write_ds(driver->Create(sidecar_path.c_str(), w, h, num_layers * 2, GDT_UInt32, nullptr));
    ASSERT_NE(write_ds, nullptr);

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
        err = write_ds->GetRasterBand(cam_band_offset + 1)
                  ->RasterIO(GF_Write, 0, 0, w, h, lo.data(), w, h, GDT_UInt32, 0, 0);
        ASSERT_EQ(err, CE_None);
        err = write_ds->GetRasterBand(cam_band_offset + 2)
                  ->RasterIO(GF_Write, 0, 0, w, h, hi.data(), w, h, GDT_UInt32, 0, 0);
        ASSERT_EQ(err, CE_None);
    }

    write_ds.reset();
    GDALDatasetPtr read_ds(static_cast<GDALDataset *>(GDALOpen(sidecar_path.c_str(), GA_ReadOnly)));
    ASSERT_NE(read_ds, nullptr);

    // WHEN: We read back the camera IDs
    for (int layer = 0; layer < num_layers; layer++)
    {
        std::vector<uint32_t> lo(w * h, 0), hi(w * h, 0);
        int cam_band_offset = layer * 2;

        CPLErr err;
        err = read_ds->GetRasterBand(cam_band_offset + 1)
                  ->RasterIO(GF_Read, 0, 0, w, h, lo.data(), w, h, GDT_UInt32, 0, 0);
        ASSERT_EQ(err, CE_None);
        err = read_ds->GetRasterBand(cam_band_offset + 2)
                  ->RasterIO(GF_Read, 0, 0, w, h, hi.data(), w, h, GDT_UInt32, 0, 0);
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
    std::unordered_map<size_t, CameraModel> cam_models;
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
        cv::Mat img(600, 800, CV_8UC3, cv::Scalar(i * 80, i * 80, i * 80));
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
    EXPECT_NO_THROW(generateGeoTIFFOrthomosaic({mesh_surface}, graph, coord_system, output_path, 512));

    // THEN: the file should exist
    EXPECT_TRUE(std::filesystem::exists(output_path));

    // Verify GeoTIFF properties using GDAL
    GDALDatasetPtr dataset = openGDALDataset(output_path);
    ASSERT_NE(dataset, nullptr);

    // Check dimensions
    EXPECT_GT(dataset->GetRasterXSize(), 0);
    EXPECT_GT(dataset->GetRasterYSize(), 0);

    // Check number of bands (should be 4: RGBA)
    EXPECT_EQ(dataset->GetRasterCount(), 4);

    // Check geotransform
    double geotransform[6];
    dataset->GetGeoTransform(geotransform);
    EXPECT_GT(geotransform[1], 0); // GSD (pixel width)
    EXPECT_LT(geotransform[5], 0); // Negative pixel height

    // Check projection (should have WKT)
    const char *projection = dataset->GetProjectionRef();
    EXPECT_NE(projection, nullptr);
    EXPECT_GT(strlen(projection), 0);

    // Check band color interpretation
    EXPECT_EQ(dataset->GetRasterBand(1)->GetColorInterpretation(), GCI_RedBand);
    EXPECT_EQ(dataset->GetRasterBand(2)->GetColorInterpretation(), GCI_GreenBand);
    EXPECT_EQ(dataset->GetRasterBand(3)->GetColorInterpretation(), GCI_BlueBand);
    EXPECT_EQ(dataset->GetRasterBand(4)->GetColorInterpretation(), GCI_AlphaBand);

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
        cv::Mat img(600, 800, CV_8UC3, cv::Scalar(255 - i * 80, 100, i * 80));
        cv::imwrite(path, img);
        temp_image_paths.push_back(path);
        graph.getNode(id[i])->payload.path = path;
    }

    GeoCoord coord_system;
    coord_system.setOrigin(0, 0);

    std::string output_path = TEST_DATA_OUTPUT_DIR "test_ortho_small_tile.tif";

    // WHEN: we generate a GeoTIFF with small tile size (should create multiple tiles)
    EXPECT_NO_THROW(generateGeoTIFFOrthomosaic({mesh_surface}, graph, coord_system, output_path, 128));

    // THEN: the file should exist and be valid
    EXPECT_TRUE(std::filesystem::exists(output_path));

    GDALDatasetPtr dataset = openGDALDataset(output_path);
    ASSERT_NE(dataset, nullptr);

    EXPECT_GT(dataset->GetRasterXSize(), 0);
    EXPECT_GT(dataset->GetRasterYSize(), 0);
    EXPECT_EQ(dataset->GetRasterCount(), 4);

    // Clean up not needed - output directory is for test artifacts
}
