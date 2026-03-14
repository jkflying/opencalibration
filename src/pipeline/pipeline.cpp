#include <opencalibration/pipeline/pipeline.hpp>

#include <cpl_vsi.h>
#include <opencalibration/pipeline/link_stage.hpp>
#include <opencalibration/pipeline/load_stage.hpp>
#include <opencalibration/pipeline/relax_stage.hpp>

#include <opencalibration/dense/dense_stereo.hpp>

#include <opencalibration/combinatorics/interleave.hpp>
#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/io/checkpoint.hpp>
#include <opencalibration/io/cv_raster_conversion.hpp>
#include <opencalibration/ortho/ortho.hpp>
#include <opencalibration/performance/performance.hpp>
#include <opencalibration/surface/expand_mesh.hpp>
#include <opencalibration/surface/intersect.hpp>
#include <opencalibration/surface/refine_mesh.hpp>

#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

#include <chrono>
#include <iostream>
#include <omp.h>

using namespace std::chrono_literals;
using fvec = std::vector<std::function<void()>>;

namespace
{

constexpr int MESH_REFINEMENT_MAX_ITERATIONS = 20;
constexpr int RELAX_MAX_ITERATIONS = 5;       // initial global relax, camera parameter relax
constexpr int FINAL_RELAX_MAX_ITERATIONS = 3; // final global relax

void run_parallel(fvec &funcs, int parallelism)
{
#pragma omp parallel for schedule(dynamic, 1) num_threads(parallelism)
    for (int i = 0; i < (int)funcs.size(); i++) // NOLINT(modernize-loop-convert)
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
    PipelineState s = State::COMPLETE;

    // clang-format off
    USM_TABLE(currentState, State::COMPLETE, s,
        USM_STATE(transition, State::INITIAL_PROCESSING,
                  USM_MAP(Transition::NEXT, State::MESH_REFINEMENT, s));
        USM_STATE(transition, State::MESH_REFINEMENT,
                  USM_MAP(Transition::NEXT, State::INITIAL_GLOBAL_RELAX, s));
        USM_STATE(transition, State::INITIAL_GLOBAL_RELAX,
                  USM_MAP(Transition::NEXT, State::CAMERA_PARAMETER_RELAX, s));
        USM_STATE(transition, State::CAMERA_PARAMETER_RELAX,
                  USM_MAP(Transition::NEXT, State::FINAL_GLOBAL_RELAX, s));
        USM_STATE(transition, State::FINAL_GLOBAL_RELAX,
                  USM_MAP(Transition::NEXT, State::GENERATE_THUMBNAIL, s));
        USM_STATE(transition, State::GENERATE_THUMBNAIL,
                  USM_MAP(Transition::NEXT, State::DENSIFY_MESH, s));
        USM_STATE(transition, State::DENSIFY_MESH,
                  USM_MAP(Transition::NEXT, State::DENSE_MESH_RELAX, s));
        USM_STATE(transition, State::DENSE_MESH_RELAX,
                  USM_MAP(Transition::NEXT, State::GENERATE_LAYERS, s));
        USM_STATE(transition, State::GENERATE_LAYERS,
                  USM_MAP(Transition::NEXT, State::COLOR_BALANCE, s));
        USM_STATE(transition, State::COLOR_BALANCE,
                  USM_MAP(Transition::NEXT, State::BLEND_LAYERS, s));
        USM_STATE(transition, State::BLEND_LAYERS,
                  USM_MAP(Transition::NEXT, State::COMPLETE, s));
    );
    // clang-format on

    return s;
}

Pipeline::Transition Pipeline::runCurrentState(PipelineState currentState)
{
    Transition t = Transition::ERROR;
    spdlog::debug("Running {}", toString(currentState));

    // clang-format off
    USM_TABLE(currentState, Transition::ERROR, t,
              USM_MAP(State::INITIAL_PROCESSING, initial_processing(), t);
              USM_MAP(State::INITIAL_GLOBAL_RELAX, initial_global_relax(), t);
              USM_MAP(State::CAMERA_PARAMETER_RELAX, camera_parameter_relax(), t);
              USM_MAP(State::FINAL_GLOBAL_RELAX, final_global_relax(), t);
              USM_MAP(State::MESH_REFINEMENT, mesh_refinement(), t);
              USM_MAP(State::COMPLETE, complete(), t);
              USM_MAP(State::GENERATE_THUMBNAIL, generate_thumbnail(), t);
              USM_MAP(State::DENSIFY_MESH, densify_mesh(), t);
              USM_MAP(State::DENSE_MESH_RELAX, dense_mesh_relax(), t);
              USM_MAP(State::GENERATE_LAYERS, generate_layers(), t);
              USM_MAP(State::COLOR_BALANCE, color_balance(), t);
              USM_MAP(State::BLEND_LAYERS, blend_layers(), t)
    );
    // clang-format on

    float local = 1.0f;
    switch (currentState)
    {
    case State::INITIAL_PROCESSING: {
        size_t total = _graph.size_nodes() + _add_queue.size();
        local = total > 0 ? float(_graph.size_nodes()) / float(total) : 1.f;
        break;
    }
    case State::MESH_REFINEMENT:
        local = std::min(1.f, float(stateRunCount()) / float(MESH_REFINEMENT_MAX_ITERATIONS));
        _emit_progress(toString(currentState), local, true);
        return t;
    case State::INITIAL_GLOBAL_RELAX:
        local = std::min(1.f, float(stateRunCount() + 1) / float(RELAX_MAX_ITERATIONS));
        _emit_progress("Optimizing camera poses (iter " + std::to_string(stateRunCount() + 1) + "/" +
                           std::to_string(RELAX_MAX_ITERATIONS + 1) + ")",
                       local, true);
        return t;
    case State::CAMERA_PARAMETER_RELAX:
        local = std::min(1.f, float(stateRunCount() + 1) / float(RELAX_MAX_ITERATIONS));
        _emit_progress("Optimizing camera parameters (iter " + std::to_string(stateRunCount() + 1) + "/" +
                           std::to_string(RELAX_MAX_ITERATIONS + 1) + ")",
                       local, true);
        return t;
    case State::FINAL_GLOBAL_RELAX:
        local = std::min(1.f, float(stateRunCount() + 1) / float(FINAL_RELAX_MAX_ITERATIONS));
        _emit_progress("Final global relaxation (iter " + std::to_string(stateRunCount() + 1) + "/" +
                           std::to_string(FINAL_RELAX_MAX_ITERATIONS + 1) + ")",
                       local, true);
        return t;
    case State::DENSE_MESH_RELAX:
        local = std::min(1.f, float(stateRunCount() + 1) / float(MESH_REFINEMENT_MAX_ITERATIONS));
        _emit_progress("Dense mesh relaxation (iter " + std::to_string(stateRunCount() + 1) + ")", local, true);
        return t;
    default:
        break;
    }
    _emit_progress(toString(currentState), local);

    return t;
}

void Pipeline::_emit_progress(std::string activity, float local_fraction, bool surfaces_updated,
                              std::optional<TileUpdate> tile_update)
{
    static const std::array<std::pair<PipelineState, float>, 11> stage_order = {{
        {State::INITIAL_PROCESSING, 0.20f},
        {State::MESH_REFINEMENT, 0.15f},
        {State::INITIAL_GLOBAL_RELAX, 0.12f},
        {State::CAMERA_PARAMETER_RELAX, 0.12f},
        {State::FINAL_GLOBAL_RELAX, 0.05f},
        {State::GENERATE_THUMBNAIL, 0.03f},
        {State::DENSIFY_MESH, 0.04f},
        {State::DENSE_MESH_RELAX, 0.03f},
        {State::GENERATE_LAYERS, 0.12f},
        {State::COLOR_BALANCE, 0.02f},
        {State::BLEND_LAYERS, 0.12f},
    }};

    PipelineState current = getState();
    float completed_weight = 0.f;
    float current_weight = 0.f;
    for (const auto &[state, weight] : stage_order)
    {
        if (state == current)
        {
            current_weight = weight;
            break;
        }
        completed_weight += weight;
    }

    StepCompletionInfo info{_next_loaded_ids,
                            _next_linked_ids,
                            _next_relaxed_ids,
                            _surfaces,
                            _graph.size_nodes(),
                            _add_queue.size(),
                            current,
                            stateRunCount(),
                            std::move(activity),
                            completed_weight + current_weight * local_fraction,
                            local_fraction,
                            surfaces_updated,
                            std::move(tile_update)};
    _step_callback(info);
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

        for (const auto &s : _relax_stage->getSurfaceModels())
        {
            _surfaces.push_back(s);
        }
    }

    USM_DECISION_TABLE(
        Transition::REPEAT,
        USM_MAKE_DECISION(_next_loaded_ids.size() == 0 && _next_linked_ids.size() == 0, Transition::NEXT));
}

Pipeline::Transition Pipeline::initial_global_relax()
{
    if (_skip_initial_global_relax)
    {
        spdlog::info("Skipping initial global relax stage");
        USM_DECISION_TABLE(Transition::NEXT, USM_MAKE_DECISION(_skip_initial_global_relax, Transition::NEXT));
    }

    _relax_stage->init(_graph, {}, _imageGPSLocations, true, false, {Option::ORIENTATION, Option::GROUND_MESH});

    fvec relax_funcs = _relax_stage->get_runners(_graph);
    run_parallel(relax_funcs, _parallelism);
    spdlog::info("global relaxed all");
    _next_relaxed_ids = _relax_stage->finalize(_graph);
    _surfaces = _relax_stage->getSurfaceModels();

    USM_DECISION_TABLE(Transition::REPEAT,
                       USM_MAKE_DECISION(stateRunCount() >= RELAX_MAX_ITERATIONS, Transition::NEXT));
}

Pipeline::Transition Pipeline::camera_parameter_relax()
{
    if (_skip_camera_param_relax)
    {
        spdlog::info("Skipping camera parameter relax stage");
        USM_DECISION_TABLE(Transition::NEXT, USM_MAKE_DECISION(_skip_camera_param_relax, Transition::NEXT));
    }

    RelaxOptionSet options;
    switch (stateRunCount())
    {
    case 0:
    case 1:
        options = {Option::ORIENTATION, Option::GROUND_MESH, Option::FOCAL_LENGTH};
        break;
    case 2:
        options = {Option::ORIENTATION, Option::GROUND_MESH, Option::FOCAL_LENGTH, Option::LENS_DISTORTIONS_RADIAL,
                   Option::LENS_DISTORTIONS_RADIAL_BROWN2_PARAMETERIZATION};
        break;
    case 3:
        options = {Option::ORIENTATION, Option::GROUND_MESH, Option::FOCAL_LENGTH, Option::LENS_DISTORTIONS_RADIAL,
                   Option::LENS_DISTORTIONS_RADIAL_BROWN24_PARAMETERIZATION};
        break;
    case 4:
        options = {Option::ORIENTATION,
                   Option::GROUND_MESH,
                   Option::FOCAL_LENGTH,
                   Option::PRINCIPAL_POINT,
                   Option::LENS_DISTORTIONS_RADIAL,
                   Option::LENS_DISTORTIONS_RADIAL_BROWN246_PARAMETERIZATION};
        break;
    default:
        options = {Option::ORIENTATION,
                   Option::GROUND_MESH,
                   Option::FOCAL_LENGTH,
                   Option::PRINCIPAL_POINT,
                   Option::LENS_DISTORTIONS_RADIAL,
                   Option::LENS_DISTORTIONS_RADIAL_BROWN246_PARAMETERIZATION};
        break;
    }

    _relax_stage->init(_graph, {}, _imageGPSLocations, true, false, options);
    _relax_stage->trim_groups(1); // do it only with the largest cluster group

    fvec relax_funcs = _relax_stage->get_runners(_graph);
    run_parallel(relax_funcs, _parallelism);
    _next_relaxed_ids = _relax_stage->finalize(_graph);
    _surfaces = _relax_stage->getSurfaceModels();

    USM_DECISION_TABLE(Transition::REPEAT,
                       USM_MAKE_DECISION(stateRunCount() >= RELAX_MAX_ITERATIONS, Transition::NEXT));
}

Pipeline::Transition Pipeline::final_global_relax()
{
    if (_skip_final_global_relax)
    {
        spdlog::info("Skipping final global relax stage");
        USM_DECISION_TABLE(Transition::NEXT, USM_MAKE_DECISION(_skip_final_global_relax, Transition::NEXT));
    }

    const bool lastIteration = stateRunCount() >= FINAL_RELAX_MAX_ITERATIONS;

    _relax_stage->init(_graph, {}, _imageGPSLocations, true, lastIteration, {Option::ORIENTATION, Option::GROUND_MESH});

    fvec relax_funcs = _relax_stage->get_runners(_graph);
    run_parallel(relax_funcs, _parallelism);
    _next_relaxed_ids = _relax_stage->finalize(_graph);
    _surfaces = _relax_stage->getSurfaceModels();

    USM_DECISION_TABLE(Transition::REPEAT,
                       USM_MAKE_DECISION(stateRunCount() >= FINAL_RELAX_MAX_ITERATIONS, Transition::NEXT));
}

Pipeline::Transition Pipeline::mesh_refinement()
{
    if (_skip_mesh_refinement)
    {
        spdlog::info("Skipping mesh refinement stage");
        USM_DECISION_TABLE(Transition::NEXT, USM_MAKE_DECISION(_skip_mesh_refinement, Transition::NEXT));
    }

    PerformanceMeasure p("Mesh refinement");

    const size_t maxPointsPerTriangle = 20;
    const double varianceGsdMultiplier = 2.0;
    const int maxIterations = MESH_REFINEMENT_MAX_ITERATIONS;
    const double baseGridFraction = 0.1;

    if (stateRunCount() == 0)
    {
        _mesh_refinement_grid_level = 0;
        _mesh_refinement_level_triangles = 0;
        // Build a single global minimal mesh covering all cameras upfront,
        // so that parallel clusters all share the same mesh structure
        point_cloud cameraLocations;
        for (auto it = _graph.cnodebegin(); it != _graph.cnodeend(); ++it)
        {
            if (it->second.payload.position.allFinite())
                cameraLocations.push_back(it->second.payload.position);
        }
        surface_model initialSurface;
        initialSurface.mesh = buildMinimalMesh(cameraLocations, _surfaces);
        _surfaces.clear();
        _surfaces.push_back(std::move(initialSurface));
        _relax_stage->setSurfaceModels(_surfaces);
    }

    const double gridFraction = baseGridFraction / std::pow(2.0, _mesh_refinement_grid_level);

    RelaxConfig config{{Option::ORIENTATION, Option::GROUND_MESH}};
    config.ground_mesh_grid_fraction = gridFraction;
    _relax_stage->init(_graph, {}, _imageGPSLocations, true, false, config);

    fvec relax_funcs = _relax_stage->get_runners(_graph);
    run_parallel(relax_funcs, _parallelism);
    _next_relaxed_ids = _relax_stage->finalize(_graph);
    _surfaces = _relax_stage->getSurfaceModels();

    if (_surfaces.empty())
    {
        spdlog::warn("No surfaces created during mesh optimization");
        USM_DECISION_TABLE(Transition::NEXT, );
    }

    double meanSurfaceZ = 0;
    size_t surfNodeCount = 0;
    for (const auto &surface : _surfaces)
    {
        for (auto it = surface.mesh.cnodebegin(); it != surface.mesh.cnodeend(); ++it)
        {
            meanSurfaceZ += it->second.payload.location.z();
            surfNodeCount++;
        }
    }
    if (surfNodeCount > 0)
        meanSurfaceZ /= surfNodeCount;

    double meanCameraZ = 0;
    double meanArcPerPixel = 0;
    double meanImageSize = 0;
    size_t camCount = 0;
    for (auto it = _graph.cnodebegin(); it != _graph.cnodeend(); ++it)
    {
        const auto &payload = it->second.payload;
        if (!payload.model || payload.model->focal_length_pixels <= 0 || !payload.position.allFinite())
            continue;
        meanCameraZ += payload.position.z();
        meanArcPerPixel += 1.0 / payload.model->focal_length_pixels;
        meanImageSize += static_cast<double>(std::max(payload.model->pixels_cols, payload.model->pixels_rows));
        camCount++;
    }

    double gsd = 0.01;
    double reducedGsd = 0.0;
    if (camCount > 0)
    {
        meanCameraZ /= camCount;
        meanArcPerPixel /= camCount;
        meanImageSize /= camCount;
        gsd = std::max(0.001, std::abs(meanCameraZ - meanSurfaceZ) * meanArcPerPixel);
        reducedGsd = std::sqrt(static_cast<double>(maxPointsPerTriangle) / 8.0) * gridFraction * meanImageSize * gsd;
    }
    const double minDistanceStddev = varianceGsdMultiplier * gsd;
    const double minDistanceVariance = minDistanceStddev * minDistanceStddev;
    spdlog::info("Mesh refinement level {}: GSD {:.4f}m, grid fraction {:.4f}, min triangle {:.4f}m",
                 _mesh_refinement_grid_level, gsd, gridFraction, reducedGsd);

    size_t trianglesAboveThreshold = 0;
    size_t maxPoints = 0;
    for (const auto &surface : _surfaces)
    {
        if (surface.mesh.size_nodes() == 0)
            continue;
        auto stats = countPointsPerTriangle(surface.mesh, surface.cloud);
        for (const auto &[key, s] : stats)
        {
            maxPoints = std::max(maxPoints, s.count);
            if (s.count > maxPointsPerTriangle && s.distanceVariance > minDistanceVariance)
                trianglesAboveThreshold++;
        }
    }

    spdlog::info("Mesh refinement iteration {}: max {} points/triangle, {} triangles above threshold", stateRunCount(),
                 maxPoints, trianglesAboveThreshold);

    bool levelConverged = (trianglesAboveThreshold == 0);

    if (!levelConverged && stateRunCount() >= (uint64_t)(maxIterations - 1))
    {
        spdlog::warn("Mesh refinement reached max iterations ({}), advancing grid level", maxIterations);
        levelConverged = true;
    }

    if (!levelConverged)
    {
        size_t totalRefined = 0;
        for (auto &surface : _surfaces)
        {
            if (surface.mesh.size_nodes() == 0)
                continue;
            totalRefined += refineByPointDensity(surface.mesh, surface.cloud, maxPointsPerTriangle, minDistanceVariance,
                                                 1, reducedGsd);
        }

        if (totalRefined == 0)
            levelConverged = true;
        else
        {
            _mesh_refinement_level_triangles += totalRefined;
            spdlog::info("Mesh refinement: created {} triangles", totalRefined);
            _relax_stage->setSurfaceModels(_surfaces);
            USM_DECISION_TABLE(Transition::REPEAT, );
        }
    }

    if (_mesh_refinement_level_triangles == 0)
    {
        spdlog::info("Mesh refinement complete: grid level {} (fraction {:.4f}) produced no new triangles",
                     _mesh_refinement_grid_level, gridFraction);
        USM_DECISION_TABLE(Transition::NEXT, );
    }

    _mesh_refinement_grid_level++;
    _mesh_refinement_level_triangles = 0;
    spdlog::info("Mesh refinement advancing to grid level {} (fraction {:.4f})", _mesh_refinement_grid_level,
                 baseGridFraction / std::pow(2.0, _mesh_refinement_grid_level));
    _relax_stage->setSurfaceModels(_surfaces);
    USM_DECISION_TABLE(Transition::REPEAT, );
}

Pipeline::Transition Pipeline::densify_mesh()
{
    if (!_generate_dense_mesh)
    {
        spdlog::info("Skipping dense mesh stage");
        USM_DECISION_TABLE(Transition::NEXT, USM_MAKE_DECISION(!_generate_dense_mesh, Transition::NEXT));
    }

    if (_surfaces.empty())
    {
        spdlog::warn("No surfaces available for dense mesh");
        USM_DECISION_TABLE(Transition::NEXT, USM_MAKE_DECISION(_surfaces.empty(), Transition::NEXT));
    }

    _emit_progress("Densifying mesh", 0.f);

    densifyMesh(_graph, _surfaces, [this](float progress) { _emit_progress("Densifying mesh", progress); });

    _emit_progress("Densifying mesh", 1.f);

    USM_DECISION_TABLE(Transition::NEXT, );
}

Pipeline::Transition Pipeline::dense_mesh_relax()
{
    if (!_generate_dense_mesh)
    {
        spdlog::info("Skipping dense mesh relax stage");
        USM_DECISION_TABLE(Transition::NEXT, USM_MAKE_DECISION(!_generate_dense_mesh, Transition::NEXT));
    }

    if (_surfaces.empty())
    {
        spdlog::warn("No surfaces available for dense mesh relax");
        USM_DECISION_TABLE(Transition::NEXT, USM_MAKE_DECISION(_surfaces.empty(), Transition::NEXT));
    }

    PerformanceMeasure p("Dense mesh refine");

    const size_t maxPointsPerTriangle = 20;
    const double varianceGsdMultiplier = 2.0;
    const double baseGridFraction = 0.05;

    double meanSurfaceZ = 0;
    size_t surfNodeCount = 0;
    for (const auto &surface : _surfaces)
    {
        for (auto it = surface.mesh.cnodebegin(); it != surface.mesh.cnodeend(); ++it)
        {
            meanSurfaceZ += it->second.payload.location.z();
            surfNodeCount++;
        }
    }
    if (surfNodeCount > 0)
        meanSurfaceZ /= surfNodeCount;

    double meanCameraZ = 0;
    double meanArcPerPixel = 0;
    double meanImageSize = 0;
    size_t camCount = 0;
    for (auto it = _graph.cnodebegin(); it != _graph.cnodeend(); ++it)
    {
        const auto &payload = it->second.payload;
        if (!payload.model || payload.model->focal_length_pixels <= 0 || !payload.position.allFinite())
            continue;
        meanCameraZ += payload.position.z();
        meanArcPerPixel += 1.0 / payload.model->focal_length_pixels;
        meanImageSize += static_cast<double>(std::max(payload.model->pixels_cols, payload.model->pixels_rows));
        camCount++;
    }

    double gsd = 0.01;
    double reducedGsd = 0.0;
    if (camCount > 0)
    {
        meanCameraZ /= camCount;
        meanArcPerPixel /= camCount;
        meanImageSize /= camCount;
        gsd = std::max(0.001, std::abs(meanCameraZ - meanSurfaceZ) * meanArcPerPixel);
        reducedGsd =
            std::sqrt(static_cast<double>(maxPointsPerTriangle) / 8.0) * baseGridFraction * meanImageSize * gsd;
    }
    const double minDistanceStddev = varianceGsdMultiplier * gsd;
    const double minDistanceVariance = minDistanceStddev * minDistanceStddev;

    size_t totalRefined = 0;
    for (auto &surface : _surfaces)
    {
        if (surface.mesh.size_nodes() == 0)
            continue;
        totalRefined +=
            refineByPointDensity(surface.mesh, surface.cloud, maxPointsPerTriangle, minDistanceVariance, 1, reducedGsd);
    }

    if (totalRefined > 0)
    {
        spdlog::info("Dense mesh refine: refined {} triangles, repeating", totalRefined);
        USM_DECISION_TABLE(Transition::REPEAT,
                           USM_MAKE_DECISION(stateRunCount() >= MESH_REFINEMENT_MAX_ITERATIONS, Transition::NEXT));
    }

    spdlog::info("Dense mesh refine complete after {} iterations", stateRunCount() + 1);
    USM_DECISION_TABLE(Transition::NEXT, );
}

Pipeline::Transition Pipeline::generate_thumbnail()
{
    if (!_generate_thumbnails || (_thumbnail_filename.empty() && _source_filename.empty() && _overlap_filename.empty()))
    {
        USM_DECISION_TABLE(Transition::NEXT, USM_MAKE_DECISION(!_generate_thumbnails, Transition::NEXT));
    }

    if (_surfaces.empty())
    {
        spdlog::warn("No surfaces available for thumbnail generation");
        USM_DECISION_TABLE(Transition::NEXT, USM_MAKE_DECISION(_surfaces.empty(), Transition::NEXT));
    }

    _emit_progress("Generating preview thumbnail", 0.f);

    auto thumbnail = orthomosaic::generateOrthomosaic(_surfaces, _graph);

    if (!_thumbnail_filename.empty())
        cv::imwrite(_thumbnail_filename, rasterToCv(thumbnail.pixelValues));
    if (!_source_filename.empty())
        cv::imwrite(_source_filename, rasterToCv(thumbnail.cameraUUID));
    if (!_overlap_filename.empty())
        cv::imwrite(_overlap_filename, rasterToCv(thumbnail.overlap));

    _emit_progress("Generating preview thumbnail", 1.f, true);

    USM_DECISION_TABLE(Transition::NEXT, );
}

Pipeline::Transition Pipeline::generate_layers()
{
    if (!_generate_geotiff || _geotiff_filename.empty())
    {
        USM_DECISION_TABLE(Transition::NEXT, USM_MAKE_DECISION(!_generate_geotiff, Transition::NEXT));
    }

    if (_surfaces.empty())
    {
        spdlog::warn("No surfaces available for layer generation");
        USM_DECISION_TABLE(Transition::NEXT, USM_MAKE_DECISION(_surfaces.empty(), Transition::NEXT));
    }

    _intermediate_layers_path = _geotiff_filename + ".layers.tif";
    _intermediate_cameras_path = _geotiff_filename + ".cameras.tif";
    _intermediate_dsm_path = !_dsm_filename.empty() ? _dsm_filename : _geotiff_filename + ".dsm.tif";

    orthomosaic::OrthoMosaicConfig config;
    config.max_output_megapixels = _orthomosaic_max_megapixels;

    TileProgressCallback tile_cb = [this](const TileUpdate &tu) {
        float local = float(tu.tile_index) / float(tu.total_tiles);
        _emit_progress("Generating layers", local, false, tu);
    };

    _correspondences =
        orthomosaic::generateLayeredGeoTIFF(_surfaces, _graph, _coordinate_system, _intermediate_layers_path,
                                            _intermediate_cameras_path, _intermediate_dsm_path, config, tile_cb);

    USM_DECISION_TABLE(Transition::NEXT, );
}

Pipeline::Transition Pipeline::color_balance()
{
    if (!_generate_geotiff || _geotiff_filename.empty())
    {
        USM_DECISION_TABLE(Transition::NEXT, USM_MAKE_DECISION(!_generate_geotiff, Transition::NEXT));
    }

    _emit_progress("Solving color balance", 0.f);

    ankerl::unordered_dense::map<size_t, orthomosaic::CameraPosition> camera_positions;
    for (const auto &corr : _correspondences)
    {
        for (size_t cam_id : {corr.camera_id_a, corr.camera_id_b})
        {
            if (camera_positions.count(cam_id) == 0)
            {
                const auto *node = _graph.getNode(cam_id);
                if (node)
                {
                    camera_positions[cam_id] = {node->payload.position.x(), node->payload.position.y()};
                }
            }
        }
    }

    _color_balance_result = orthomosaic::solveColorBalance(_correspondences, camera_positions);
    _correspondences.clear();

    _emit_progress("Solving color balance", 1.f);

    USM_DECISION_TABLE(Transition::NEXT, );
}

Pipeline::Transition Pipeline::blend_layers()
{
    if (!_generate_geotiff || _geotiff_filename.empty())
    {
        USM_DECISION_TABLE(Transition::NEXT, USM_MAKE_DECISION(!_generate_geotiff, Transition::NEXT));
    }

    orthomosaic::OrthoMosaicConfig config;
    config.max_output_megapixels = _orthomosaic_max_megapixels;

    TileProgressCallback tile_cb = [this](const TileUpdate &tu) {
        float local = float(tu.tile_index) / float(tu.total_tiles);
        _emit_progress("Blending layers", local, false, tu);
    };

    orthomosaic::blendLayeredGeoTIFF(_intermediate_layers_path, _intermediate_cameras_path, _intermediate_dsm_path,
                                     _geotiff_filename, _color_balance_result, _graph, _coordinate_system, config,
                                     tile_cb);

    _color_balance_result = {};

    // Clean up intermediate DSM if it was a temp file (not the user-requested output)
    if (_dsm_filename.empty() && !_intermediate_dsm_path.empty())
    {
        VSIUnlink(_intermediate_dsm_path.c_str());
    }

    if (!_textured_mesh_filename.empty() && !_geotiff_filename.empty())
    {
        orthomosaic::generateTexturedOBJ(_surfaces, _geotiff_filename, _textured_mesh_filename);
    }

    USM_DECISION_TABLE(Transition::NEXT, );
}

Pipeline::Transition Pipeline::complete()
{
    USM_DECISION_TABLE(Transition::REPEAT, );
}

std::string Pipeline::toString(PipelineState state)
{
    switch (state)
    {
    case State::INITIAL_PROCESSING:
        return "Initial Processing";
    case State::INITIAL_GLOBAL_RELAX:
        return "Initial global Relax";
    case State::CAMERA_PARAMETER_RELAX:
        return "Camera Parameter Relax";
    case State::FINAL_GLOBAL_RELAX:
        return "Final Global Relax";
    case State::MESH_REFINEMENT:
        return "Mesh Refinement";
    case State::GENERATE_THUMBNAIL:
        return "Generate Thumbnail";
    case State::DENSIFY_MESH:
        return "Densify Mesh";
    case State::DENSE_MESH_RELAX:
        return "Dense Mesh Relax";
    case State::GENERATE_LAYERS:
        return "Generate Layers";
    case State::COLOR_BALANCE:
        return "Color Balance";
    case State::BLEND_LAYERS:
        return "Blend Layers";
    case State::COMPLETE:
        return "Complete";
    };

    return "Error";
}

std::optional<PipelineState> Pipeline::fromString(const std::string &str)
{
    if (str == "INITIAL_PROCESSING" || str == "Initial Processing")
        return State::INITIAL_PROCESSING;
    if (str == "INITIAL_GLOBAL_RELAX" || str == "Initial global Relax")
        return State::INITIAL_GLOBAL_RELAX;
    if (str == "CAMERA_PARAMETER_RELAX" || str == "Camera Parameter Relax")
        return State::CAMERA_PARAMETER_RELAX;
    if (str == "FINAL_GLOBAL_RELAX" || str == "Final Global Relax")
        return State::FINAL_GLOBAL_RELAX;
    if (str == "MESH_REFINEMENT" || str == "Mesh Refinement")
        return State::MESH_REFINEMENT;
    if (str == "GENERATE_THUMBNAIL" || str == "Generate Thumbnail")
        return State::GENERATE_THUMBNAIL;
    if (str == "DENSIFY_MESH" || str == "Densify Mesh")
        return State::DENSIFY_MESH;
    if (str == "DENSE_MESH_RELAX" || str == "Dense Mesh Relax")
        return State::DENSE_MESH_RELAX;
    if (str == "GENERATE_LAYERS" || str == "Generate Layers" || str == "GENERATE_DSM" || str == "Generate DSM")
        return State::GENERATE_LAYERS;
    if (str == "COLOR_BALANCE" || str == "Color Balance")
        return State::COLOR_BALANCE;
    if (str == "BLEND_LAYERS" || str == "Blend Layers")
        return State::BLEND_LAYERS;
    if (str == "COMPLETE" || str == "Complete")
        return State::COMPLETE;

    return std::nullopt;
}

void Pipeline::rebuildGPSLocationsTree()
{
    _imageGPSLocations = jk::tree::KDTree<size_t, 2>();

    for (auto it = _graph.cnodebegin(); it != _graph.cnodeend(); ++it)
    {
        const auto &node = it->second;
        std::array<double, 2> gps_pos = {node.payload.position.x(), node.payload.position.y()};
        _imageGPSLocations.addPoint(gps_pos, it->first);
    }
}

bool Pipeline::saveCheckpoint(const std::string &checkpoint_dir)
{
    CheckpointData data;
    data.graph = _graph;
    data.surfaces = _surfaces;
    data.origin_latitude = _coordinate_system.getOriginLatitude();
    data.origin_longitude = _coordinate_system.getOriginLongitude();
    data.state = getState();
    data.state_run_count = stateRunCount();

    return opencalibration::saveCheckpoint(data, checkpoint_dir);
}

bool Pipeline::loadCheckpoint(const std::string &checkpoint_dir)
{
    CheckpointData data;

    if (!opencalibration::loadCheckpoint(checkpoint_dir, data))
    {
        return false;
    }

    _graph = std::move(data.graph);
    _surfaces = std::move(data.surfaces);

    if (data.origin_latitude != 0.0 || data.origin_longitude != 0.0)
    {
        _coordinate_system.setOrigin(data.origin_latitude, data.origin_longitude);
    }

    setCurrentState(data.state);
    setStateRunCount(data.state_run_count);

    rebuildGPSLocationsTree();

    spdlog::info("Loaded checkpoint with state: {}, run_count: {}", toString(data.state), data.state_run_count);

    return true;
}

bool Pipeline::resumeFromState(PipelineState target_state)
{
    PipelineState current = getState();

    if (static_cast<int>(target_state) > static_cast<int>(current))
    {
        spdlog::error("Cannot resume to state {} from {}: target state is later than current", toString(target_state),
                      toString(current));
        return false;
    }

    setCurrentState(target_state);
    setStateRunCount(0);

    spdlog::info("Resuming from state: {}", toString(target_state));

    return true;
}

} // namespace opencalibration
