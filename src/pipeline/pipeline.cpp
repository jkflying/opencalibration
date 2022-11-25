#include <opencalibration/pipeline/pipeline.hpp>

#include <opencalibration/pipeline/link_stage.hpp>
#include <opencalibration/pipeline/load_stage.hpp>
#include <opencalibration/pipeline/relax_stage.hpp>

#include <opencalibration/combinatorics/interleave.hpp>
#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/performance/performance.hpp>
#include <opencalibration/surface/intersect.hpp>

#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

#include <chrono>
#include <iostream>

using namespace std::chrono_literals;
using fvec = std::vector<std::function<void()>>;

namespace
{

void run_parallel(fvec &funcs, int parallelism)
{
#pragma omp parallel for schedule(dynamic, 1) num_threads(parallelism)
    for (int i = 0; i < (int)funcs.size(); i++)
    {
        funcs[i]();
    }
}

} // namespace

namespace opencalibration
{
Pipeline::Pipeline(size_t batch_size, size_t parallelism)
    : usm::StateMachine<PipelineState, PipelineTransition>(State::INITIAL_PROCESSING), _load_stage(new LoadStage()),
      _link_stage(new LinkStage()), _relax_stage(new RelaxStage()), _step_callback([](const StepCompletionInfo &) {}),
      _batch_size(batch_size), _parallelism(parallelism == 0 ? omp_get_num_procs() : parallelism)
{
}

Pipeline::~Pipeline()
{
}

void Pipeline::add(const std::vector<std::string> &paths)
{

    std::lock_guard<std::mutex> guard(_queue_mutex);
    _add_queue.insert(_add_queue.end(), paths.begin(), paths.end());
    _queue_condition_variable.notify_all();
}

PipelineState Pipeline::chooseNextState(PipelineState currentState, Transition transition)
{
    // clang-format off
    USM_TABLE(currentState, State::COMPLETE,
        USM_STATE(transition, State::INITIAL_PROCESSING,
                  USM_MAP(Transition::NEXT, State::GLOBAL_RELAX));
        USM_STATE(transition, State::GLOBAL_RELAX,
                  USM_MAP(Transition::NEXT, State::CAMERA_PARAMETER_RELAX));
        USM_STATE(transition, State::CAMERA_PARAMETER_RELAX,
                  USM_MAP(Transition::NEXT, State::FINAL_GLOBAL_RELAX));
        USM_STATE(transition, State::FINAL_GLOBAL_RELAX,
                  USM_MAP(Transition::NEXT, State::GENERATE_THUMBNAIL));
        USM_STATE(transition, State::GENERATE_THUMBNAIL,
                  USM_MAP(Transition::NEXT, State::COMPLETE));
    );
    // clang-format on
}

Pipeline::Transition Pipeline::runCurrentState(PipelineState currentState)
{
    Transition t = Transition::ERROR;
    spdlog::debug("Running {}", toString(currentState));

    switch (currentState)
    {
    case State::INITIAL_PROCESSING:
        t = initial_processing();
        break;
    case State::GLOBAL_RELAX:
        t = global_relax();
        break;
    case State::CAMERA_PARAMETER_RELAX:
        t = camera_parameter_relax();
        break;
    case State::FINAL_GLOBAL_RELAX:
        t = final_global_relax();
        break;
    case State::COMPLETE:
        t = complete();
        break;
    case State::GENERATE_THUMBNAIL:
        t = generate_thumbnail();
        break;
    default:
        t = Transition::ERROR;
        break;
    }

    StepCompletionInfo info{_next_loaded_ids,    _next_linked_ids,  _next_relaxed_ids, _surfaces,
                            _graph.size_nodes(), _add_queue.size(), currentState,      stateRunCount()};
    _step_callback(info);

    return t;
}

Pipeline::Transition Pipeline::initial_processing()
{
    auto get_paths = [this](std::vector<std::string> &paths) -> bool {
        std::lock_guard<std::mutex> guard(_queue_mutex);
        while (_add_queue.size() > 0 && paths.size() < _batch_size)
        {
            paths.emplace_back(std::move(_add_queue.front()));
            _add_queue.pop_front();
        }
        return paths.size() > 0;
    };

    std::vector<std::string> paths;
    if (get_paths(paths) || _next_loaded_ids.size() > 0 || _next_linked_ids.size() > 0)
    {
        _previous_loaded_ids = std::move(_next_loaded_ids);
        _next_loaded_ids.clear();

        _previous_linked_ids = std::move(_next_linked_ids);
        _next_linked_ids.clear();

        _load_stage->init(_graph, paths);
        _link_stage->init(_graph, _imageGPSLocations, _previous_loaded_ids);
        _relax_stage->init(_graph, _previous_linked_ids, _imageGPSLocations, false, true,
                           {Option::ORIENTATION, Option::GROUND_PLANE});

        fvec funcs;
        {
            fvec load_funcs = _load_stage->get_runners();
            fvec link_funcs = _link_stage->get_runners(_graph);
            fvec relax_funcs = _relax_stage->get_runners(_graph);
            funcs = interleave<fvec>({load_funcs, link_funcs, relax_funcs});
        }

        run_parallel(funcs, _parallelism);

        _next_loaded_ids = _load_stage->finalize(_coordinate_system, _graph, _imageGPSLocations);
        _next_linked_ids = _link_stage->finalize(_graph);
        _next_relaxed_ids = _relax_stage->finalize(_graph);

        return Transition::REPEAT;
    }
    else
    {
        return Transition::NEXT;
    }
}

Pipeline::Transition Pipeline::global_relax()
{
    _relax_stage->init(_graph, {}, _imageGPSLocations, true, false, {Option::ORIENTATION, Option::GROUND_MESH});

    fvec relax_funcs = _relax_stage->get_runners(_graph);
    run_parallel(relax_funcs, _parallelism);
    _next_relaxed_ids = _relax_stage->finalize(_graph);
    _surfaces = _relax_stage->getSurfaceModels();

    if (stateRunCount() < 5)
        return Transition::REPEAT;
    else
        return Transition::NEXT;
}

Pipeline::Transition Pipeline::camera_parameter_relax()
{
    RelaxOptionSet options;
    switch (stateRunCount())
    {
    case 0:
    case 1:
        options = {Option::ORIENTATION, Option::POINTS_3D, Option::FOCAL_LENGTH};
        break;
    case 2:
        options = {Option::ORIENTATION, Option::POINTS_3D, Option::FOCAL_LENGTH, Option::LENS_DISTORTIONS_RADIAL,
                   Option::LENS_DISTORTIONS_RADIAL_BROWN2_PARAMETERIZATION};
        break;
    case 3:
        options = {Option::ORIENTATION, Option::POINTS_3D, Option::FOCAL_LENGTH, Option::LENS_DISTORTIONS_RADIAL,
                   Option::LENS_DISTORTIONS_RADIAL_BROWN24_PARAMETERIZATION};
        break;
    case 4:
        options = {Option::ORIENTATION, Option::POINTS_3D, Option::FOCAL_LENGTH, Option::LENS_DISTORTIONS_RADIAL,
                   Option::LENS_DISTORTIONS_RADIAL_BROWN246_PARAMETERIZATION};
        break;
    default:
        options = {Option::ORIENTATION,
                   Option::POINTS_3D,
                   Option::FOCAL_LENGTH,
                   Option::LENS_DISTORTIONS_RADIAL,
                   Option::LENS_DISTORTIONS_RADIAL_BROWN246_PARAMETERIZATION,
                   Option::LENS_DISTORTIONS_TANGENTIAL};
        break;
    }

    _relax_stage->init(_graph, {}, _imageGPSLocations, true, false, options);
    _relax_stage->trim_groups(1); // do it only with the largest cluster group

    fvec relax_funcs = _relax_stage->get_runners(_graph);
    run_parallel(relax_funcs, _parallelism);
    _next_relaxed_ids = _relax_stage->finalize(_graph);
    _surfaces = _relax_stage->getSurfaceModels();

    if (stateRunCount() < 5)
        return Transition::REPEAT;
    else
        return Transition::NEXT;
}

Pipeline::Transition Pipeline::final_global_relax()
{
    const bool lastIteration = stateRunCount() >= 3;

    _relax_stage->init(_graph, {}, _imageGPSLocations, true, lastIteration, {Option::ORIENTATION, Option::GROUND_MESH});

    fvec relax_funcs = _relax_stage->get_runners(_graph);
    run_parallel(relax_funcs, _parallelism);
    _next_relaxed_ids = _relax_stage->finalize(_graph);
    _surfaces = _relax_stage->getSurfaceModels();

    if (lastIteration)
        return Transition::NEXT;
    else
        return Transition::REPEAT;
}

Pipeline::Transition Pipeline::generate_thumbnail()
{

    // calculate min/max x/y, and average z
    const double inf = std::numeric_limits<double>::infinity();
    double min_x = inf, min_y = inf, max_x = -inf, max_y = -inf;
    double mean_surface_z = 0;
    size_t count_z = 0;

    for (const auto &surface : _surfaces)
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
    for (auto iter = _graph.cnodebegin(); iter != _graph.cnodeend(); ++iter)
    {
        const auto &payload = iter->second.payload;

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

    spdlog::info("thumb arc pixel {}  mean camera z {}", thumb_arc_pixel, mean_camera_z);

    const double average_camera_elevation = mean_camera_z - mean_surface_z;
    const double mean_gsd = average_camera_elevation * thumb_arc_pixel;

    // from bounds and gsd, calculate image resolution
    const double image_width = (max_x - min_x) / mean_gsd;
    const double image_height = (max_y - min_y) / mean_gsd;

    cv::Size image_dimensions(static_cast<int>(image_width), static_cast<int>(image_height));

    spdlog::info("gsd {}  img dims {}x{}", mean_gsd, image_dimensions.width, image_dimensions.height);

    cv::Mat image(image_dimensions, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    cv::Mat source(image_dimensions, CV_32S, -1);

    // iterate over each pixel
#pragma omp parallel for
    for (int row = 0; row < image_dimensions.height; row++)
    {
        PerformanceMeasure p("Generate thumbnail");
        std::vector<MeshIntersectionSearcher> searchers;
        for (const auto &surface : _surfaces)
        {
            searchers.emplace_back();
            if (!searchers.back().init(surface.mesh))
            {
                spdlog::error("Could not initialize searcher on mesh surface");
            }
        }
        spdlog::debug("processing row {}", row);

        for (int col = 0; col < image_dimensions.width; col++)
        {
            spdlog::debug("processing col {}", col);

            const double x = col * mean_gsd + min_x;
            const double y = row * mean_gsd + min_y;

            // get height of pixel from mesh or nearest keypoint
            const ray_d intersectinoRay{{0, 0, -1}, {x, y, average_camera_elevation}};
            double z = NAN;
            for (auto &searcher : searchers)
            {
                if (searcher.lastResult().type != MeshIntersectionSearcher::IntersectionInfo::INTERSECTION)
                {
                    searcher.reinit();
                }

                auto result = searcher.triangleIntersect(intersectinoRay);
                if (result.type == MeshIntersectionSearcher::IntersectionInfo::INTERSECTION)
                {
                    z = result.intersectionLocation.z();
                    break;
                }
            }

            Eigen::Vector3d sample_point(x, y, z);

            cv::Vec4b color(0, 0, 0, 0);
            int pixelSource =
                (row + col) % 2 == 0
                    ? -1
                    : static_cast<int>(
                          _graph.size_nodes()); // give a checkerboard pattern of illegal values for background color

            auto closest5 = _imageGPSLocations.searchKnn({x, y, average_camera_elevation}, 5);

            // get image vertically closest
            for (const auto &closest : closest5)
            {
                const auto *closestNode = _graph.getNode(closest.payload);
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
                    auto pixel = payload.thumbnail.at<cv::Vec3b>(cvPixel); // TODO: use a kernel to get subpixel accuracy
                    color = cv::Vec4b(pixel[0], pixel[1], pixel[2], 255);
                    pixelSource = closest.payload & 0xFF;
                    break;
                }
            }

            // assign color to thumbnail pixel
            image.at<cv::Vec4b>(row, col) = color;
            source.at<int>(row, col) = pixelSource;
        }
    }

    cv::imwrite("thumbnail.png", image);
    cv::imwrite("source.tiff", source);

    // TODO: some kind of color balancing of the different patches - maybe in LAB space?
    // TODO: laplacian or gradient domain blending of the differrent patches

    return Transition::NEXT;
}

Pipeline::Transition Pipeline::complete()
{
    return Transition::REPEAT;
}

std::string Pipeline::toString(PipelineState state)
{
    switch (state)
    {
    case State::INITIAL_PROCESSING:
        return "Initial Processing";
    case State::GLOBAL_RELAX:
        return "Global Relax";
    case State::CAMERA_PARAMETER_RELAX:
        return "Camera Parameter Relax";
    case State::FINAL_GLOBAL_RELAX:
        return "Final Global Relax";
    case State::GENERATE_THUMBNAIL:
        return "Generate Thumbnail";
    case State::COMPLETE:
        return "Complete";
    };

    return "Error";
}

} // namespace opencalibration
