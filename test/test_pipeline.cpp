#include <opencalibration/pipeline/pipeline.hpp>

#include <opencalibration/ortho/gdal_dataset.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>

using namespace opencalibration;
using namespace std::chrono_literals;

TEST(pipeline, constructs_with_lots_of_threads)
{
    Pipeline p(100);
    std::this_thread::sleep_for(1ms);
}

TEST(pipeline, processes_4_images)
{

    // GIVEN: a pipeline and paths
    Pipeline p(2);
    std::string path1 = TEST_DATA_DIR "P2530253.JPG";
    std::string path2 = TEST_DATA_DIR "P2540254.JPG";
    std::string path3 = TEST_DATA_DIR "P2550255.JPG";
    std::string path4 = TEST_DATA_DIR "P2560256.JPG";

    // WHEN: we add the paths
    p.add({path1, path2, path3, path4});

    // THEN: after some time they should all be processed
    while (p.getState() != PipelineState::COMPLETE)
    {
        p.iterateOnce();
    }
}

#include <filesystem>

TEST(pipeline, generates_thumbnails_when_requested)
{
    // GIVEN: a pipeline with custom thumbnail filenames
    Pipeline p(2);
    p.set_thumbnail_filenames("test_thumb.tiff", "test_source.png", "");

    std::string path1 = TEST_DATA_DIR "P2530253.JPG";
    std::string path2 = TEST_DATA_DIR "P2540254.JPG";
    std::string path3 = TEST_DATA_DIR "P2550255.JPG";
    std::string path4 = TEST_DATA_DIR "P2560256.JPG";

    std::filesystem::remove("test_thumb.tiff");
    std::filesystem::remove("test_source.png");
    std::filesystem::remove("overlap.png");
    std::filesystem::remove("thumbnail.tiff");
    std::filesystem::remove("source.png");

    // WHEN: we process images
    p.add({path1, path2, path3, path4});
    while (p.getState() != PipelineState::COMPLETE)
    {
        p.iterateOnce();
    }

    // THEN: requested files should exist, others should not
    EXPECT_TRUE(std::filesystem::exists("test_thumb.tiff"));
    EXPECT_TRUE(std::filesystem::exists("test_source.png"));
    EXPECT_FALSE(std::filesystem::exists("overlap.png"));
    EXPECT_FALSE(std::filesystem::exists("thumbnail.tiff"));
    EXPECT_FALSE(std::filesystem::exists("source.png"));

    std::filesystem::remove("test_thumb.tiff");
    std::filesystem::remove("test_source.png");
}

TEST(pipeline, generates_geotiff_when_requested)
{
    // GIVEN: a pipeline with geotiff filename
    Pipeline p(2);
    std::string output_path = TEST_DATA_OUTPUT_DIR "test_pipeline_ortho.tif";
    constexpr double max_output_megapixels = 0.1;
    p.set_geotiff_filename(output_path);
    p.set_orthomosaic_max_megapixels(max_output_megapixels);

    std::string path1 = TEST_DATA_DIR "P2530253.JPG";
    std::string path2 = TEST_DATA_DIR "P2540254.JPG";
    std::string path3 = TEST_DATA_DIR "P2550255.JPG";
    std::string path4 = TEST_DATA_DIR "P2560256.JPG";

    // WHEN: we process images
    p.add({path1, path2, path3, path4});
    while (p.getState() != PipelineState::COMPLETE)
    {
        p.iterateOnce();
    }

    // THEN: the GeoTIFF file should exist
    EXPECT_TRUE(std::filesystem::exists(output_path));

    // AND: it should be a valid GeoTIFF with proper properties
    using namespace opencalibration::orthomosaic;
    GDALDatasetPtr dataset = openGDALDataset(output_path);
    ASSERT_NE(dataset.get(), nullptr) << "Failed to open GeoTIFF file";

    GDALDatasetWrapper ds_wrapper(dataset.get());

    // Verify dimensions
    EXPECT_GT(ds_wrapper.GetRasterXSize(), 0) << "GeoTIFF width should be > 0";
    EXPECT_GT(ds_wrapper.GetRasterYSize(), 0) << "GeoTIFF height should be > 0";
    uint64_t output_pixels =
        static_cast<uint64_t>(ds_wrapper.GetRasterXSize()) * static_cast<uint64_t>(ds_wrapper.GetRasterYSize());
    uint64_t max_output_pixels = static_cast<uint64_t>(max_output_megapixels * 1000000.0);
    EXPECT_LE(output_pixels, max_output_pixels) << "GeoTIFF should honor configured max megapixels";

    // Verify 4 bands (RGBA)
    EXPECT_EQ(ds_wrapper.GetRasterCount(), 4) << "GeoTIFF should have 4 bands (RGBA)";

    // Verify geotransform is set
    double geotransform[6];
    CPLErr err = ds_wrapper.GetGeoTransform(geotransform);
    EXPECT_EQ(err, CE_None) << "GeoTIFF should have a geotransform";
    EXPECT_GT(geotransform[1], 0) << "GSD (pixel width) should be > 0";
    EXPECT_LT(geotransform[5], 0) << "Pixel height should be negative (north-up orientation)";

    // Verify projection is set
    const char *projection = ds_wrapper.GetProjectionRef();
    EXPECT_NE(projection, nullptr) << "GeoTIFF should have a projection";
    EXPECT_GT(strlen(projection), 0) << "Projection WKT should not be empty";

    // Verify band color interpretation
    GDALRasterBandWrapper band1(ds_wrapper.GetRasterBand(1));
    GDALRasterBandWrapper band2(ds_wrapper.GetRasterBand(2));
    GDALRasterBandWrapper band3(ds_wrapper.GetRasterBand(3));
    GDALRasterBandWrapper band4(ds_wrapper.GetRasterBand(4));
    EXPECT_EQ(band1.GetColorInterpretation(), GCI_RedBand);
    EXPECT_EQ(band2.GetColorInterpretation(), GCI_GreenBand);
    EXPECT_EQ(band3.GetColorInterpretation(), GCI_BlueBand);
    EXPECT_EQ(band4.GetColorInterpretation(), GCI_AlphaBand);

    // Verify internal tiling is enabled
    int block_x, block_y;
    band1.GetBlockSize(&block_x, &block_y);
    EXPECT_EQ(block_x, 512) << "GeoTIFF should have 512×512 internal tile blocks";
    EXPECT_EQ(block_y, 512);

    // Output saved to TEST_DATA_OUTPUT_DIR for inspection
}
