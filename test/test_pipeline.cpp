#include <opencalibration/pipeline/pipeline.hpp>

#include <gdal/gdal_priv.h>
#include <gtest/gtest.h>

#include <chrono>

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
    p.set_geotiff_filename(output_path);

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
    GDALAllRegister();
    GDALDataset *dataset = (GDALDataset *)GDALOpen(output_path.c_str(), GA_ReadOnly);
    ASSERT_NE(dataset, nullptr) << "Failed to open GeoTIFF file";

    // Verify dimensions
    EXPECT_GT(dataset->GetRasterXSize(), 0) << "GeoTIFF width should be > 0";
    EXPECT_GT(dataset->GetRasterYSize(), 0) << "GeoTIFF height should be > 0";

    // Verify 4 bands (RGBA)
    EXPECT_EQ(dataset->GetRasterCount(), 4) << "GeoTIFF should have 4 bands (RGBA)";

    // Verify geotransform is set
    double geotransform[6];
    CPLErr err = dataset->GetGeoTransform(geotransform);
    EXPECT_EQ(err, CE_None) << "GeoTIFF should have a geotransform";
    EXPECT_GT(geotransform[1], 0) << "GSD (pixel width) should be > 0";
    EXPECT_LT(geotransform[5], 0) << "Pixel height should be negative (north-up orientation)";

    // Verify projection is set
    const char *projection = dataset->GetProjectionRef();
    EXPECT_NE(projection, nullptr) << "GeoTIFF should have a projection";
    EXPECT_GT(strlen(projection), 0) << "Projection WKT should not be empty";

    // Verify band color interpretation
    EXPECT_EQ(dataset->GetRasterBand(1)->GetColorInterpretation(), GCI_RedBand);
    EXPECT_EQ(dataset->GetRasterBand(2)->GetColorInterpretation(), GCI_GreenBand);
    EXPECT_EQ(dataset->GetRasterBand(3)->GetColorInterpretation(), GCI_BlueBand);
    EXPECT_EQ(dataset->GetRasterBand(4)->GetColorInterpretation(), GCI_AlphaBand);

    // Verify internal tiling is enabled
    int block_x, block_y;
    dataset->GetRasterBand(1)->GetBlockSize(&block_x, &block_y);
    EXPECT_EQ(block_x, 512) << "GeoTIFF should have 512×512 internal tile blocks";
    EXPECT_EQ(block_y, 512);

    GDALClose(dataset);

    // Output saved to TEST_DATA_OUTPUT_DIR for inspection
}
