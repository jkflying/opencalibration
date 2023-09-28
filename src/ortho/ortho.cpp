#include <opencalibration/ortho/ortho.hpp>

#include <jk/KDTree.h>
#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/performance/performance.hpp>
#include <opencalibration/surface/intersect.hpp>

#include <spdlog/spdlog.h>

namespace
{

std::array<double, 3> to_array(const Eigen::Vector3d &v)
{
    return {v.x(), v.y(), v.z()};
}
} // namespace

namespace opencalibration::orthomosaic
{

OrthoMosaic generateOrthomosaic(const std::vector<surface_model> &surfaces, const MeasurementGraph &graph)
{
    // calculate min/max x/y, and average z
    const double inf = std::numeric_limits<double>::infinity();
    double min_x = inf, min_y = inf, max_x = -inf, max_y = -inf;
    double mean_surface_z = 0;
    size_t count_z = 0;

    for (const auto &surface : surfaces)
    {
        for (auto iter = surface.mesh.cnodebegin(); iter != surface.mesh.cnodeend(); ++iter)
        {
            const auto &loc = iter->second.payload.location;

            mean_surface_z = (loc.z() + mean_surface_z * count_z) / (count_z + 1);
            min_x = std::min(min_x, loc.x());
            max_x = std::max(max_x, loc.x());
            min_y = std::min(min_y, loc.y());
            max_y = std::max(max_y, loc.y());
            count_z++;
        }
    }

    spdlog::info("x range [{}; {}]  y range [{}; {}]  mean surface {}", min_x, max_x, min_y, max_y, mean_surface_z);

    // calculate gsd size based on thumbnail resolution at average mesh/keypoint height
    double thumb_arc_pixel = 0;
    double mean_camera_z = 0;
    size_t thumb_count = 0;
    jk::tree::KDTree<size_t, 3> imageGPSLocations;
    for (auto iter = graph.cnodebegin(); iter != graph.cnodeend(); ++iter)
    {
        const auto &payload = iter->second.payload;
        imageGPSLocations.addPoint(to_array(payload.position), iter->first, false);
        const double h = 0.001;
        Eigen::Vector2d pixel = image_from_3d({0, 0, 1}, *payload.model);
        Eigen::Vector2d pixelShift = image_from_3d({h, 0, 1}, *payload.model);
        double arc_pixel = h / (pixel - pixelShift).norm();
        double thumb_scale =
            static_cast<double>(payload.thumbnail.size().height) / payload.metadata.camera_info.height_px;
        thumb_arc_pixel = (thumb_arc_pixel * thumb_count + arc_pixel / thumb_scale) / (thumb_count + 1);
        mean_camera_z = (mean_camera_z * thumb_count + payload.position.z()) / (thumb_count + 1);
        thumb_count++;
    }
    imageGPSLocations.splitOutstanding();

    spdlog::info("thumb arc pixel {}  mean camera z {}", thumb_arc_pixel, mean_camera_z);

    const double average_camera_elevation = mean_camera_z - mean_surface_z;
    const double mean_gsd = average_camera_elevation * thumb_arc_pixel;

    // from bounds and gsd, calculate image resolution
    const double image_width = (max_x - min_x) / mean_gsd;
    const double image_height = (max_y - min_y) / mean_gsd;

    cv::Size image_dimensions(static_cast<int>(image_width), static_cast<int>(image_height));

    spdlog::info("gsd {}  img dims {}x{}", mean_gsd, image_dimensions.width, image_dimensions.height);

    OrthoMosaic result;
    result.pixelValues = cv::Mat(image_dimensions, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    result.cameraUUIDLsb = cv::Mat(image_dimensions, CV_8UC1, 0);
    result.overlap = cv::Mat(image_dimensions, CV_8UC1, 0);

    // iterate over each pixel
#pragma omp parallel for
    for (int row = 0; row < image_dimensions.height; row++)
    {
        PerformanceMeasure p("Generate thumbnail");
        std::vector<MeshIntersectionSearcher> searchers;
        for (const auto &surface : surfaces)
        {
            searchers.emplace_back();
            if (!searchers.back().init(surface.mesh))
            {
                spdlog::error("Could not initialize searcher on mesh surface");
                searchers.pop_back();
            }
        }
        spdlog::debug("processing row {}", row);

        for (int col = 0; col < image_dimensions.width; col++)
        {
            spdlog::debug("processing col {}", col);

            const double x = col * mean_gsd + min_x;
            const double y = row * mean_gsd + min_y;

            // get height of pixel from mesh or nearest keypoint
            const ray_d intersectionRay{{0, 0, -1}, {x, y, average_camera_elevation}};
            double z = NAN;
            for (auto &searcher : searchers)
            {
                if (searcher.lastResult().type != MeshIntersectionSearcher::IntersectionInfo::INTERSECTION)
                {
                    if (!searcher.reinit())
                    {
                        continue;
                    }
                }

                auto result = searcher.triangleIntersect(intersectionRay);
                if (result.type == MeshIntersectionSearcher::IntersectionInfo::INTERSECTION)
                {
                    z = result.intersectionLocation.z();
                    break;
                }
            }

            Eigen::Vector3d sample_point(x, y, z);

            cv::Vec4b color(0, 0, 0, 0);
            uint8_t pixelSource =
                (row + col) % 2 == 0
                    ? -1
                    : static_cast<int>(
                          graph.size_nodes()); // give a checkerboard pattern of illegal values for background color

            auto closest5 = imageGPSLocations.searchKnn({x, y, average_camera_elevation}, 5);

            // get image vertically closest
            for (const auto &closest : closest5)
            {
                const auto *closestNode = graph.getNode(closest.payload);
                const auto &payload = closestNode->payload;

                // backproject 3D point onto thumbnail image, get color
                Eigen::Vector2d pixel =
                    image_from_3d(sample_point, *payload.model, payload.position, payload.orientation);

                const double thumb_scale =
                    static_cast<double>(payload.thumbnail.size().height) / payload.metadata.camera_info.height_px;
                Eigen::Vector2d thumb_pixel = pixel * thumb_scale;
                cv::Point2i cvPixel(thumb_pixel.x(), thumb_pixel.y());

                if (0 < cvPixel.x && cvPixel.x < payload.thumbnail.size().width && 0 < cvPixel.y &&
                    cvPixel.y < payload.thumbnail.size().height)
                {
                    auto pixel =
                        payload.thumbnail.at<cv::Vec3b>(cvPixel); // TODO: use a kernel to get subpixel accuracy
                    color = cv::Vec4b(pixel[0], pixel[1], pixel[2], 255);
                    pixelSource = closest.payload & 0xFF;
                    break;
                }
            }

            // assign color to thumbnail pixel
            result.pixelValues.at<cv::Vec4b>(row, col) = color;
            result.cameraUUIDLsb.at<uint8_t>(row, col) = pixelSource;
        }
    }

    return result;

    //     cv::imwrite("thumbnail.png", image);
    //     cv::imwrite("source.png", source);

    // TODO: some kind of color balancing of the different patches - maybe in LAB space?
    // TODO: laplacian or gradient domain blending of the differrent patches
}
} // namespace opencalibration::orthomosaic
