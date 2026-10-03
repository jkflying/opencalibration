#include <opencalibration/pipeline/pipeline.hpp>

#include "link_stage.hpp"
#include "load_stage.hpp"
#include "relax_stage.hpp"

#include <opencalibration/combinatorics/interleave.hpp>
#include <opencalibration/dense/densify_multiview.hpp>
#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/io/checkpoint.hpp>
#include <opencalibration/io/cv_raster_conversion.hpp>
#include <opencalibration/ortho/color_balance.hpp>
#include <opencalibration/ortho/ortho.hpp>
#include <opencalibration/performance/performance.hpp>
#include <opencalibration/relax/relax_problem.hpp>
#include <opencalibration/surface/expand_mesh.hpp>
#include <opencalibration/surface/intersect.hpp>
#include <opencalibration/surface/refine_mesh.hpp>

#include <jk/KDTree.h>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>
#include <usm.hpp>

#include <atomic>
#include <chrono>
#include <deque>
#include <iostream>
#include <mutex>
#include <omp.h>

using namespace std::chrono_literals;
using fvec = std::vector<std::function<void()>>;

namespace
{

void logPhaseHotspots()
{
    auto &stats = opencalibration::FeatureSet::loadStats();
    static size_t loads = 0, features = 0, nanoseconds = 0, lock_wait = 0;
    const size_t now_loads = stats.loads, now_features = stats.features, now_nanoseconds = stats.nanoseconds,
                 now_lock_wait = stats.lock_wait_nanoseconds;
    if (now_loads != loads)
        spdlog::info(
            "Feature loads: {} ({:.0f} MB) in {:.1f}s summed over threads, {:.1f}s of it waiting for the store",
            now_loads - loads, (now_features - features) * sizeof(opencalibration::feature_2d) / 1048576.,
            (now_nanoseconds - nanoseconds) * 1e-9, (now_lock_wait - lock_wait) * 1e-9);
    loads = now_loads, features = now_features, nanoseconds = now_nanoseconds, lock_wait = now_lock_wait;

    if (const std::string totals = opencalibration::TopPerformanceTotalsSinceLastCall(10); !totals.empty())
        spdlog::info("Hotspots (summed thread time): {}", totals);
}

constexpr int MESH_REFINEMENT_MAX_ITERATIONS = 20;
constexpr int MESH_REFINEMENT_MAX_GRID_LEVEL = 2;
constexpr size_t MESH_REFINEMENT_FINAL_POINTS_PER_TRIANGLE = 25;
constexpr int RELAX_MAX_ITERATIONS = 5;       // initial global relax, camera parameter relax
constexpr int FINAL_RELAX_MAX_ITERATIONS = 3; // final global relax

void run_parallel(fvec &funcs, int parallelism)
{
    parallelism = std::max(1, std::min(parallelism, static_cast<int>(funcs.size())));
#pragma omp parallel for schedule(dynamic, 1) num_threads(parallelism)
    for (int i = 0; i < (int)funcs.size(); i++) // NOLINT(modernize-loop-convert)
    {
        funcs[i]();
    }
}

cv::Mat packCameraIdLow24BitsIntoRgb(const opencalibration::RasterLayer<int32_t> &cameraIds)
{
    cv::Mat packed(cameraIds.pixels.rows(), cameraIds.pixels.cols(), CV_8UC4);
    for (Eigen::Index row = 0; row < cameraIds.pixels.rows(); row++)
    {
        for (Eigen::Index col = 0; col < cameraIds.pixels.cols(); col++)
        {
            const uint32_t id = static_cast<uint32_t>(cameraIds.pixels(row, col));
            const uint8_t red = id >> 16, green = id >> 8, blue = id, alpha = 255;
            packed.at<cv::Vec4b>(row, col) = cv::Vec4b(blue, green, red, alpha);
        }
    }
    return packed;
}

struct MeshScale
{
    double gsd = 0.01;
    double meanImageSize = 0;
};

MeshScale estimateMeshScale(const opencalibration::MeasurementGraph &graph,
                            const std::vector<opencalibration::surface_model> &surfaces)
{
    double meanSurfaceZ = 0;
    size_t surfNodeCount = 0;
    for (const auto &surface : surfaces)
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
    for (auto it = graph.cnodebegin(); it != graph.cnodeend(); ++it)
    {
        const auto &payload = it->second.payload;
        if (!payload.model || payload.model->focal_length_pixels <= 0 || !payload.position.allFinite())
            continue;
        meanCameraZ += payload.position.z();
        meanArcPerPixel += 1.0 / payload.model->focal_length_pixels;
        meanImageSize += static_cast<double>(std::max(payload.model->pixels_cols, payload.model->pixels_rows));
        camCount++;
    }

    if (camCount == 0)
        return {};
    meanCameraZ /= camCount;
    meanArcPerPixel /= camCount;
    meanImageSize /= camCount;
    return {std::max(0.001, std::abs(meanCameraZ - meanSurfaceZ) * meanArcPerPixel), meanImageSize};
}

} // namespace

namespace opencalibration
{

enum class PipelineTransition
{
    REPEAT,
    NEXT,
    ERROR
};

struct Pipeline::Impl : public usm::StateMachine<PipelineState, PipelineTransition>
{
    using State = PipelineState;
    using Transition = PipelineTransition;

    MeasurementGraph graph;
    GeoCoord coordinate_system;
    std::vector<surface_model> surfaces;

    std::vector<size_t> previous_loaded_ids, previous_linked_ids, next_loaded_ids, next_linked_ids;
    std::vector<std::vector<size_t>> next_relaxed_ids;

    std::unique_ptr<LoadStage> load_stage;
    std::unique_ptr<LinkStage> link_stage;
    std::unique_ptr<RelaxStage> relax_stage;

    std::mutex queue_mutex;
    std::mutex progress_mutex;
    std::deque<std::string> add_queue;
    size_t queue_size()
    {
        std::lock_guard<std::mutex> guard(queue_mutex);
        return add_queue.size();
    }

    jk::tree::KDTree<size_t, 2> imageGPSLocations;

    StepCompletionCallback step_callback;

    size_t batch_size;
    size_t parallelism;
    bool generate_thumbnails = true;
    std::string thumbnail_filename;
    std::string source_filename;
    std::string overlap_filename;
    bool generate_geotiff = false;
    std::string geotiff_filename;
    std::string dsm_filename;
    std::string textured_mesh_filename;
    double orthomosaic_max_megapixels = 0.0;

    std::chrono::steady_clock::time_point stage_start = std::chrono::steady_clock::now();

    int mesh_refinement_grid_level = 0;
    int mesh_refinement_level_iterations = 0;
    size_t mesh_refinement_level_triangles = 0;

    bool skip_mesh_refinement = false;
    bool skip_initial_global_relax = true;
    bool skip_camera_param_relax = false;
    bool skip_final_global_relax = true;
    bool generate_dense_mesh = false;

    orthomosaic::ColorBalanceResult color_balance_result;

    Impl(size_t batch_size_, size_t parallelism_)
        : usm::StateMachine<PipelineState, PipelineTransition>(State::INITIAL_PROCESSING), load_stage(new LoadStage()),
          link_stage(new LinkStage()), relax_stage(new RelaxStage()), step_callback([](const StepCompletionInfo &) {}),
          batch_size(std::max<size_t>(batch_size_, 1)),
          parallelism(parallelism_ == 0 ? omp_get_num_procs() : parallelism_)
    {
        cv::setNumThreads(1);
    }

    PipelineState chooseNextState(PipelineState currentState, Transition transition) override;
    PipelineTransition runCurrentState(PipelineState currentState) override;

    Transition initial_processing();
    Transition initial_global_relax();
    Transition camera_parameter_relax();
    Transition final_global_relax();
    Transition mesh_refinement();
    Transition generate_thumbnail();
    Transition densify_mesh();
    Transition dense_mesh_relax();
    Transition generate_orthomosaic();
    Transition complete();

    void emit_progress(std::string activity, float local_fraction, bool surfaces_updated = false,
                       std::optional<TileUpdate> tile_update = std::nullopt);
    void rebuildGPSLocationsTree();

    void resetState(PipelineState state, uint64_t run_count = 0)
    {
        setCurrentState(state);
        setStateRunCount(run_count);
    }
};

// --- Pipeline public method delegations ---

Pipeline::Pipeline(size_t batch_size, size_t parallelism) : _impl(std::make_unique<Impl>(batch_size, parallelism))
{
}

Pipeline::~Pipeline() = default;
Pipeline::Pipeline(Pipeline &&) noexcept = default;
Pipeline &Pipeline::operator=(Pipeline &&) noexcept = default;

void Pipeline::add(const std::vector<std::string> &paths)
{
    std::lock_guard<std::mutex> guard(_impl->queue_mutex);
    _impl->add_queue.insert(_impl->add_queue.end(), paths.begin(), paths.end());
}

const MeasurementGraph &Pipeline::getGraph() const
{
    return _impl->graph;
}

const GeoCoord &Pipeline::getCoord() const
{
    return _impl->coordinate_system;
}

const std::vector<surface_model> &Pipeline::getSurfaces() const
{
    return _impl->surfaces;
}

void Pipeline::set_callback(const StepCompletionCallback &cb)
{
    _impl->step_callback = cb;
}

void Pipeline::set_generate_thumbnails(bool v)
{
    _impl->generate_thumbnails = v;
}

void Pipeline::set_thumbnail_filenames(const std::string &t, const std::string &s, const std::string &o)
{
    _impl->thumbnail_filename = t;
    _impl->source_filename = s;
    _impl->overlap_filename = o;
}

void Pipeline::set_geotiff_filename(const std::string &geotiff)
{
    _impl->geotiff_filename = geotiff;
    if (!geotiff.empty())
    {
        _impl->generate_geotiff = true;
    }
}

void Pipeline::set_dsm_filename(const std::string &dsm)
{
    _impl->dsm_filename = dsm;
    if (!dsm.empty())
    {
        _impl->generate_geotiff = true;
    }
}

void Pipeline::set_textured_mesh_filename(const std::string &path)
{
    _impl->textured_mesh_filename = path;
    if (!path.empty())
    {
        _impl->generate_geotiff = true;
    }
}

void Pipeline::set_orthomosaic_max_megapixels(double v)
{
    _impl->orthomosaic_max_megapixels = v;
}

bool Pipeline::set_project_dir(const std::string &dir)
{
    _impl->load_stage->store = ProjectStore::open(dir);
    return _impl->load_stage->store != nullptr;
}

void Pipeline::set_skip_mesh_refinement(bool v)
{
    _impl->skip_mesh_refinement = v;
}

void Pipeline::set_skip_initial_global_relax(bool v)
{
    _impl->skip_initial_global_relax = v;
}

void Pipeline::set_skip_camera_param_relax(bool v)
{
    _impl->skip_camera_param_relax = v;
}

void Pipeline::set_skip_final_global_relax(bool v)
{
    _impl->skip_final_global_relax = v;
}

void Pipeline::set_generate_dense_mesh(bool v)
{
    _impl->generate_dense_mesh = v;
}

void Pipeline::iterateOnce()
{
    _impl->iterateOnce();
}

PipelineState Pipeline::getState() const
{
    return _impl->getState();
}

bool Pipeline::saveCheckpoint(const std::string &checkpoint_dir)
{
    PerformanceMeasure p("Checkpoint save");
    CheckpointData data;
    data.graph = std::move(_impl->graph);
    data.surfaces = std::move(_impl->surfaces);
    struct MoveBack
    {
        Impl &impl;
        CheckpointData &data;
        ~MoveBack()
        {
            impl.graph = std::move(data.graph);
            impl.surfaces = std::move(data.surfaces);
        }
    } move_back{*_impl, data};
    data.origin_latitude = _impl->coordinate_system.getOriginLatitude();
    data.origin_longitude = _impl->coordinate_system.getOriginLongitude();
    data.state = _impl->getState();
    data.state_run_count = _impl->stateRunCount();
    data.color_balance = _impl->color_balance_result;

    return opencalibration::saveCheckpoint(data, checkpoint_dir);
}

bool Pipeline::loadCheckpoint(const std::string &checkpoint_dir, const std::string &stage)
{
    CheckpointData data;

    if (!opencalibration::loadCheckpoint(checkpoint_dir, data, stage))
    {
        return false;
    }

    _impl->graph = std::move(data.graph);
    _impl->surfaces = std::move(data.surfaces);
    _impl->color_balance_result = std::move(data.color_balance);
    _impl->relax_stage->setSurfaceModels(_impl->surfaces);

    if (data.origin_latitude != 0.0 || data.origin_longitude != 0.0)
    {
        _impl->coordinate_system.setOrigin(data.origin_latitude, data.origin_longitude);
    }

    _impl->resetState(data.state, data.state_run_count);

    _impl->rebuildGPSLocationsTree();

    spdlog::info("Loaded checkpoint with state: {}, run_count: {}", toString(data.state), data.state_run_count);

    return true;
}

bool Pipeline::resumeFromState(PipelineState target_state)
{
    PipelineState current = _impl->getState();

    if (static_cast<int>(target_state) > static_cast<int>(current))
    {
        spdlog::error("Cannot resume to state {} from {}: target state is later than current", toString(target_state),
                      toString(current));
        return false;
    }

    _impl->resetState(target_state);

    spdlog::info("Resuming from state: {}", toString(target_state));

    return true;
}

std::string Pipeline::toString(PipelineState state)
{
    switch (state)
    {
    case PipelineState::INITIAL_PROCESSING:
        return "Initial Processing";
    case PipelineState::INITIAL_GLOBAL_RELAX:
        return "Initial global Relax";
    case PipelineState::CAMERA_PARAMETER_RELAX:
        return "Camera Parameter Relax";
    case PipelineState::FINAL_GLOBAL_RELAX:
        return "Final Global Relax";
    case PipelineState::MESH_REFINEMENT:
        return "Mesh Refinement";
    case PipelineState::GENERATE_THUMBNAIL:
        return "Generate Thumbnail";
    case PipelineState::DENSIFY_MESH:
        return "Densify Mesh";
    case PipelineState::DENSE_MESH_RELAX:
        return "Dense Mesh Relax";
    case PipelineState::GENERATE_GEOTIFF:
        return "Generate GeoTIFF";
    case PipelineState::COMPLETE:
        return "Complete";
    };

    return "Error";
}

std::optional<PipelineState> Pipeline::fromString(const std::string &str)
{
    if (auto state = stringToPipelineState(str))
        return state;
    for (int i = 0; i <= static_cast<int>(PipelineState::COMPLETE); i++)
        if (toString(static_cast<PipelineState>(i)) == str)
            return static_cast<PipelineState>(i);
    return std::nullopt;
}

// --- Impl state machine methods ---

PipelineState Pipeline::Impl::chooseNextState(PipelineState currentState, Transition transition)
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
                  USM_MAP(Transition::NEXT, State::DENSIFY_MESH, s));
        USM_STATE(transition, State::DENSIFY_MESH,
                  USM_MAP(Transition::NEXT, State::DENSE_MESH_RELAX, s));
        USM_STATE(transition, State::DENSE_MESH_RELAX,
                  USM_MAP(Transition::NEXT, State::GENERATE_THUMBNAIL, s));
        USM_STATE(transition, State::GENERATE_THUMBNAIL,
                  USM_MAP(Transition::NEXT, State::GENERATE_GEOTIFF, s));
        USM_STATE(transition, State::GENERATE_GEOTIFF,
                  USM_MAP(Transition::NEXT, State::COMPLETE, s));
    );
    // clang-format on

    return s;
}

Pipeline::Impl::Transition Pipeline::Impl::runCurrentState(PipelineState currentState)
{
    Transition t = Transition::ERROR;
    spdlog::debug("Running {}", Pipeline::toString(currentState));
    if (stateRunCount() == 0)
        stage_start = std::chrono::steady_clock::now();

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
              USM_MAP(State::GENERATE_GEOTIFF, generate_orthomosaic(), t)
    );
    // clang-format on

    if (t != Transition::REPEAT)
    {
        spdlog::info("{} finished in {:.1f}s over {} iterations", Pipeline::toString(currentState),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start).count(),
                     stateRunCount() + 1);
        logPhaseHotspots();
    }

    float local = 1.0f;
    switch (currentState)
    {
    case State::INITIAL_PROCESSING: {
        size_t total = graph.size_nodes() + queue_size();
        local = total > 0 ? float(graph.size_nodes()) / float(total) : 1.f;
        break;
    }
    case State::MESH_REFINEMENT:
        local = std::min(1.f, float(stateRunCount()) / float(MESH_REFINEMENT_MAX_ITERATIONS));
        emit_progress(Pipeline::toString(currentState), local, true);
        return t;
    case State::INITIAL_GLOBAL_RELAX:
        local = std::min(1.f, float(stateRunCount() + 1) / float(RELAX_MAX_ITERATIONS));
        emit_progress("Optimizing camera poses (iter " + std::to_string(stateRunCount() + 1) + "/" +
                          std::to_string(RELAX_MAX_ITERATIONS + 1) + ")",
                      local, true);
        return t;
    case State::CAMERA_PARAMETER_RELAX:
        local = std::min(1.f, float(stateRunCount() + 1) / float(RELAX_MAX_ITERATIONS));
        emit_progress("Optimizing camera parameters (iter " + std::to_string(stateRunCount() + 1) + "/" +
                          std::to_string(RELAX_MAX_ITERATIONS + 1) + ")",
                      local, true);
        return t;
    case State::FINAL_GLOBAL_RELAX:
        local = std::min(1.f, float(stateRunCount() + 1) / float(FINAL_RELAX_MAX_ITERATIONS));
        emit_progress("Final global relaxation (iter " + std::to_string(stateRunCount() + 1) + "/" +
                          std::to_string(FINAL_RELAX_MAX_ITERATIONS + 1) + ")",
                      local, true);
        return t;
    case State::DENSE_MESH_RELAX:
        local = std::min(1.f, float(stateRunCount() + 1) / float(MESH_REFINEMENT_MAX_ITERATIONS));
        emit_progress("Dense mesh relaxation (iter " + std::to_string(stateRunCount() + 1) + ")", local, true);
        return t;
    default:
        break;
    }
    emit_progress(Pipeline::toString(currentState), local);

    return t;
}

void Pipeline::Impl::emit_progress(std::string activity, float local_fraction, bool surfaces_updated,
                                   std::optional<TileUpdate> tile_update)
{
    static const std::array<std::pair<PipelineState, float>, 9> stage_order = {{
        {State::INITIAL_PROCESSING, 0.20f},
        {State::MESH_REFINEMENT, 0.15f},
        {State::INITIAL_GLOBAL_RELAX, 0.12f},
        {State::CAMERA_PARAMETER_RELAX, 0.12f},
        {State::FINAL_GLOBAL_RELAX, 0.05f},
        {State::DENSIFY_MESH, 0.04f},
        {State::DENSE_MESH_RELAX, 0.03f},
        {State::GENERATE_THUMBNAIL, 0.05f},
        {State::GENERATE_GEOTIFF, 0.24f},
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

    StepCompletionInfo info{next_loaded_ids,    next_linked_ids,     next_relaxed_ids,
                            graph.size_nodes(), queue_size(),        current,
                            stateRunCount(),    std::move(activity), completed_weight + current_weight * local_fraction,
                            local_fraction,     surfaces_updated,    std::move(tile_update)};
    std::lock_guard<std::mutex> guard(progress_mutex);
    step_callback(info);
}

Pipeline::Impl::Transition Pipeline::Impl::initial_processing()
{
    auto get_paths = [this](std::vector<std::string> &paths) -> bool {
        std::lock_guard<std::mutex> guard(queue_mutex);
        while (add_queue.size() > 0 && paths.size() < batch_size)
        {
            paths.emplace_back(std::move(add_queue.front()));
            add_queue.pop_front();
        }
        return paths.size() > 0;
    };

    std::vector<std::string> paths;
    if (get_paths(paths) || next_loaded_ids.size() > 0 || next_linked_ids.size() > 0)
    {
        previous_loaded_ids = std::move(next_loaded_ids);
        next_loaded_ids.clear();

        previous_linked_ids = std::move(next_linked_ids);
        next_linked_ids.clear();

        load_stage->init(graph, paths);
        link_stage->init(graph, imageGPSLocations, previous_loaded_ids);
        relax_stage->init(graph, previous_linked_ids, imageGPSLocations, false, true,
                          {Option::ORIENTATION, Option::TRIANGULATED_RAYS});

        const auto batch_start = std::chrono::steady_clock::now();
        const auto elapsed = [batch_start] {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - batch_start).count();
        };
        const auto log_stage_done = [&elapsed](const char *name, fvec funcs) {
            auto remaining = std::make_shared<std::atomic<size_t>>(funcs.size());
            for (auto &f : funcs)
                f = [&elapsed, name, remaining, f = std::move(f)] {
                    f();
                    if (--*remaining == 0)
                        spdlog::debug("{} stage done after {:.2f}s", name, elapsed());
                };
            return funcs;
        };

        fvec funcs;
        {
            fvec load_funcs = log_stage_done("load", load_stage->get_runners());
            fvec link_funcs = log_stage_done("link", link_stage->get_runners(graph));
            fvec relax_funcs = log_stage_done("relax", relax_stage->get_runners(graph));
            spdlog::debug("Batch starting: {} load, {} link, {} relax runners", load_funcs.size(), link_funcs.size(),
                          relax_funcs.size());
            funcs = interleave<fvec>({load_funcs, link_funcs, relax_funcs});
        }

        run_parallel(funcs, parallelism);
        spdlog::debug("Batch done after {:.2f}s", elapsed());

        next_loaded_ids = load_stage->finalize(coordinate_system, graph, imageGPSLocations);
        next_linked_ids = link_stage->finalize(graph);
        next_relaxed_ids = relax_stage->finalize(graph);

        for (const auto &s : relax_stage->getSurfaceModels())
        {
            surfaces.push_back(s);
        }
    }

    USM_DECISION_TABLE(Transition::REPEAT,
                       USM_MAKE_DECISION(next_loaded_ids.size() == 0 && next_linked_ids.size() == 0, Transition::NEXT));
}

Pipeline::Impl::Transition Pipeline::Impl::initial_global_relax()
{
    if (skip_initial_global_relax)
    {
        spdlog::info("Skipping initial global relax stage");
        USM_DECISION_TABLE(Transition::NEXT, USM_MAKE_DECISION(skip_initial_global_relax, Transition::NEXT));
    }

    relax_stage->init(graph, {}, imageGPSLocations, true, false, {Option::ORIENTATION, Option::GROUND_MESH});

    fvec relax_funcs = relax_stage->get_runners(graph);
    run_parallel(relax_funcs, parallelism);
    spdlog::info("global relaxed all");
    next_relaxed_ids = relax_stage->finalize(graph);
    surfaces = relax_stage->getSurfaceModels();

    USM_DECISION_TABLE(Transition::REPEAT,
                       USM_MAKE_DECISION(stateRunCount() >= RELAX_MAX_ITERATIONS, Transition::NEXT));
}

Pipeline::Impl::Transition Pipeline::Impl::camera_parameter_relax()
{
    if (skip_camera_param_relax)
    {
        spdlog::info("Skipping camera parameter relax stage");
        USM_DECISION_TABLE(Transition::NEXT, USM_MAKE_DECISION(skip_camera_param_relax, Transition::NEXT));
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
    options.set(Option::POSITION, true);

    relax_stage->init(graph, {}, imageGPSLocations, true, false, options);
    relax_stage->trim_groups(1);

    fvec relax_funcs = relax_stage->get_runners(graph);
    run_parallel(relax_funcs, parallelism);
    next_relaxed_ids = relax_stage->finalize(graph);
    surfaces = relax_stage->getSurfaceModels();

    USM_DECISION_TABLE(Transition::REPEAT,
                       USM_MAKE_DECISION(stateRunCount() >= RELAX_MAX_ITERATIONS, Transition::NEXT));
}

Pipeline::Impl::Transition Pipeline::Impl::final_global_relax()
{
    if (skip_final_global_relax)
    {
        spdlog::info("Skipping final global relax stage");
        USM_DECISION_TABLE(Transition::NEXT, USM_MAKE_DECISION(skip_final_global_relax, Transition::NEXT));
    }

    const bool lastIteration = stateRunCount() >= FINAL_RELAX_MAX_ITERATIONS;

    relax_stage->init(graph, {}, imageGPSLocations, true, lastIteration,
                      {Option::ORIENTATION, Option::POSITION, Option::GROUND_MESH});

    fvec relax_funcs = relax_stage->get_runners(graph);
    run_parallel(relax_funcs, parallelism);
    next_relaxed_ids = relax_stage->finalize(graph);
    surfaces = relax_stage->getSurfaceModels();

    USM_DECISION_TABLE(Transition::REPEAT,
                       USM_MAKE_DECISION(stateRunCount() >= FINAL_RELAX_MAX_ITERATIONS, Transition::NEXT));
}

Pipeline::Impl::Transition Pipeline::Impl::mesh_refinement()
{
    if (skip_mesh_refinement)
    {
        spdlog::info("Skipping mesh refinement stage");
        USM_DECISION_TABLE(Transition::NEXT, USM_MAKE_DECISION(skip_mesh_refinement, Transition::NEXT));
    }

    PerformanceMeasure p("Mesh refinement");

    const double varianceGsdMultiplier = 2.0;
    const int maxIterations = MESH_REFINEMENT_MAX_ITERATIONS;
    const double baseGridFraction = 0.1;
    const double minRefinedFractionToContinue = 0.02;

    if (stateRunCount() == 0)
    {
        mesh_refinement_grid_level = 0;
        mesh_refinement_level_iterations = 0;
        mesh_refinement_level_triangles = 0;
        point_cloud cameraLocations;
        for (auto it = graph.cnodebegin(); it != graph.cnodeend(); ++it)
        {
            if (it->second.payload.position.allFinite())
                cameraLocations.push_back(it->second.payload.position);
        }
        surface_model initialSurface;
        initialSurface.mesh = buildMinimalMesh(cameraLocations, surfaces);
        surfaces.clear();
        surfaces.push_back(std::move(initialSurface));
        relax_stage->setSurfaceModels(surfaces);
    }

    const double gridFraction = baseGridFraction / std::pow(2.0, mesh_refinement_grid_level);
    const size_t maxPointsPerTriangle = MESH_REFINEMENT_FINAL_POINTS_PER_TRIANGLE
                                        << (MESH_REFINEMENT_MAX_GRID_LEVEL - mesh_refinement_grid_level);

    RelaxConfig config{{Option::ORIENTATION, Option::GROUND_MESH}};
    config.ground_mesh_grid_fraction = gridFraction;
    relax_stage->init(graph, {}, imageGPSLocations, true, false, config);

    fvec relax_funcs = relax_stage->get_runners(graph);
    run_parallel(relax_funcs, parallelism);
    next_relaxed_ids = relax_stage->finalize(graph);
    surfaces = relax_stage->getSurfaceModels();

    if (surfaces.empty())
    {
        spdlog::warn("No surfaces created during mesh optimization");
        USM_DECISION_TABLE(Transition::NEXT, );
    }

    const auto [gsd, meanImageSize] = estimateMeshScale(graph, surfaces);
    const double reducedGsd =
        std::sqrt(static_cast<double>(maxPointsPerTriangle) / 8.0) * gridFraction * meanImageSize * gsd;
    const double minDistanceStddev = varianceGsdMultiplier * gsd;
    const double minDistanceVariance = minDistanceStddev * minDistanceStddev;
    spdlog::info("Mesh refinement level {}: GSD {:.4f}m, grid fraction {:.4f}, min triangle {:.4f}m, max {} "
                 "points/triangle",
                 mesh_refinement_grid_level, gsd, gridFraction, reducedGsd, maxPointsPerTriangle);

    size_t trianglesAboveThreshold = 0;
    size_t maxPoints = 0;
    size_t triangleCount = 0;
    for (const auto &surface : surfaces)
    {
        if (surface.mesh.size_nodes() == 0)
            continue;
        auto stats = countPointsPerTriangle(surface.mesh, surface.cloud);
        triangleCount += stats.size();
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

    if (!levelConverged && ++mesh_refinement_level_iterations >= maxIterations)
    {
        spdlog::warn("Mesh refinement reached max iterations ({}) at grid level {}", maxIterations,
                     mesh_refinement_grid_level);
        levelConverged = true;
    }

    if (!levelConverged)
    {
        size_t totalRefined = 0;
        for (auto &surface : surfaces)
        {
            if (surface.mesh.size_nodes() == 0)
                continue;
            totalRefined += refineByPointDensity(surface.mesh, surface.cloud, maxPointsPerTriangle, minDistanceVariance,
                                                 1, reducedGsd);
        }

        mesh_refinement_level_triangles += totalRefined;
        spdlog::info("Mesh refinement: created {} triangles", totalRefined);
        levelConverged = totalRefined < minRefinedFractionToContinue * triangleCount;
        if (!levelConverged)
        {
            relax_stage->setSurfaceModels(surfaces);
            USM_DECISION_TABLE(Transition::REPEAT, );
        }
    }

    if (mesh_refinement_grid_level >= MESH_REFINEMENT_MAX_GRID_LEVEL ||
        mesh_refinement_level_triangles <= minRefinedFractionToContinue * triangleCount)
    {
        size_t vertexCount = 0;
        for (const auto &surface : surfaces)
            vertexCount += surface.mesh.size_nodes();
        spdlog::info("Mesh refinement complete: grid level {} (fraction {:.4f}) produced {} of {} triangles, mesh has "
                     "{} vertices",
                     mesh_refinement_grid_level, gridFraction, mesh_refinement_level_triangles, triangleCount,
                     vertexCount);
        USM_DECISION_TABLE(Transition::NEXT, );
    }

    mesh_refinement_grid_level++;
    mesh_refinement_level_iterations = 0;
    mesh_refinement_level_triangles = 0;
    spdlog::info("Mesh refinement advancing to grid level {} (fraction {:.4f})", mesh_refinement_grid_level,
                 baseGridFraction / std::pow(2.0, mesh_refinement_grid_level));
    relax_stage->setSurfaceModels(surfaces);
    USM_DECISION_TABLE(Transition::REPEAT, );
}

Pipeline::Impl::Transition Pipeline::Impl::densify_mesh()
{
    if (!generate_dense_mesh)
    {
        spdlog::info("Skipping dense mesh stage");
        USM_DECISION_TABLE(Transition::NEXT, USM_MAKE_DECISION(!generate_dense_mesh, Transition::NEXT));
    }

    if (surfaces.empty())
    {
        spdlog::warn("No surfaces available for dense mesh");
        USM_DECISION_TABLE(Transition::NEXT, USM_MAKE_DECISION(surfaces.empty(), Transition::NEXT));
    }

    emit_progress("Densifying mesh", 0.f);

    densifyMesh(graph, surfaces, [this](float progress) { emit_progress("Densifying mesh", progress); });

    emit_progress("Densifying mesh", 1.f);

    USM_DECISION_TABLE(Transition::NEXT, );
}

Pipeline::Impl::Transition Pipeline::Impl::dense_mesh_relax()
{
    if (!generate_dense_mesh)
    {
        spdlog::info("Skipping dense mesh relax stage");
        USM_DECISION_TABLE(Transition::NEXT, USM_MAKE_DECISION(!generate_dense_mesh, Transition::NEXT));
    }

    if (surfaces.empty())
    {
        spdlog::warn("No surfaces available for dense mesh relax");
        USM_DECISION_TABLE(Transition::NEXT, USM_MAKE_DECISION(surfaces.empty(), Transition::NEXT));
    }

    PerformanceMeasure p("Dense mesh setup");

    const size_t maxPointsPerTriangle = 20;
    const double varianceGsdMultiplier = 2.0;
    const double baseGridFraction = 0.05;

    const auto [gsd, meanImageSize] = estimateMeshScale(graph, surfaces);
    const double reducedGsd =
        std::sqrt(static_cast<double>(maxPointsPerTriangle) / 8.0) * baseGridFraction * meanImageSize * gsd;
    const double minDistanceStddev = varianceGsdMultiplier * gsd;
    const double minDistanceVariance = minDistanceStddev * minDistanceStddev;

    size_t totalRefined = 0;
    for (auto &surface : surfaces)
    {
        if (surface.mesh.size_nodes() == 0)
            continue;
        const double pointSigma = std::max(minDistanceStddev, estimatePointHeightSigma(surface.cloud));
        constexpr double HEIGHT_AGREEMENT_SIGMAS = 5;
        p.reset("Dense mesh filter points");
        surface_model agreeingSurface{
            filterPointsWithoutHeightAgreement(surface.cloud, HEIGHT_AGREEMENT_SIGMAS * pointSigma),
            std::move(surface.mesh)};
        auto countPoints = [](const std::vector<point_cloud> &clouds) {
            size_t count = 0;
            for (const auto &cloud : clouds)
                count += cloud.size();
            return count;
        };
        spdlog::info("Dense mesh relax: {} of {} points agree within {:.3f}m", countPoints(agreeingSurface.cloud),
                     countPoints(surface.cloud), HEIGHT_AGREEMENT_SIGMAS * pointSigma);
        auto fitHeights = [&agreeingSurface, pointSigma, &p] {
            p.reset("Dense mesh fit heights");
            RelaxProblem rp;
            rp.setupMeshHeightProblem(agreeingSurface, pointSigma);
            rp.solveMeshHeights();
            agreeingSurface.mesh = rp.getSurfaceModel().mesh;
        };
        const bool unrefinedMesh = stateRunCount() == 0;
        if (unrefinedMesh)
            fitHeights();
        p.reset("Dense mesh refine");
        const size_t refined = refineByPointDensity(agreeingSurface.mesh, agreeingSurface.cloud,
                                                    maxPointsPerTriangle, minDistanceVariance, 1, reducedGsd);
        totalRefined += refined;
        if (refined > 0)
            fitHeights();
        surface.mesh = std::move(agreeingSurface.mesh);
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

Pipeline::Impl::Transition Pipeline::Impl::generate_thumbnail()
{
    const bool save_thumbnail =
        generate_thumbnails && !(thumbnail_filename.empty() && source_filename.empty() && overlap_filename.empty());
    const bool need_color_balance = generate_geotiff && !geotiff_filename.empty();
    if (!save_thumbnail && !need_color_balance)
    {
        USM_DECISION_TABLE(Transition::NEXT, );
    }

    if (surfaces.empty())
    {
        spdlog::warn("No surfaces available for thumbnail generation");
        USM_DECISION_TABLE(Transition::NEXT, );
    }

    emit_progress("Generating preview thumbnail", 0.f);

    orthomosaic::OrthoMosaicConfig config;
    config.max_output_megapixels = orthomosaic_max_megapixels;
    auto thumbnail = orthomosaic::generateOrthomosaic(surfaces, graph, config);
    color_balance_result = std::move(thumbnail.color_balance);

    if (save_thumbnail)
    {
        if (!thumbnail_filename.empty())
            cv::imwrite(thumbnail_filename, rasterToCv(thumbnail.pixelValues));
        if (!source_filename.empty())
            cv::imwrite(source_filename, packCameraIdLow24BitsIntoRgb(thumbnail.cameraUUID));
        if (!overlap_filename.empty())
            cv::imwrite(overlap_filename, rasterToCv(thumbnail.overlap));
    }

    emit_progress("Generating preview thumbnail", 1.f, true);

    USM_DECISION_TABLE(Transition::NEXT, );
}

Pipeline::Impl::Transition Pipeline::Impl::generate_orthomosaic()
{
    if (!generate_geotiff || (geotiff_filename.empty() && dsm_filename.empty()))
    {
        USM_DECISION_TABLE(Transition::NEXT, USM_MAKE_DECISION(!generate_geotiff, Transition::NEXT));
    }

    if (surfaces.empty())
    {
        spdlog::warn("No surfaces available for GeoTIFF generation");
        USM_DECISION_TABLE(Transition::NEXT, USM_MAKE_DECISION(surfaces.empty(), Transition::NEXT));
    }

    if (color_balance_result.per_image_params.empty())
    {
        spdlog::warn("No color balance available, resume from GENERATE_THUMBNAIL to compute one");
    }

    orthomosaic::OrthoMosaicConfig config;
    config.max_output_megapixels = orthomosaic_max_megapixels;

    TileProgressCallback tile_cb = [this](const TileUpdate &tu) {
        float local = float(tu.tile_index) / float(tu.total_tiles);
        emit_progress("Generating orthomosaic", local, false, tu);
    };

    orthomosaic::generateGeoTIFF(surfaces, graph, coordinate_system, color_balance_result, geotiff_filename,
                                 dsm_filename, config, tile_cb);

    if (!textured_mesh_filename.empty() && !geotiff_filename.empty())
    {
        orthomosaic::generateTexturedOBJ(surfaces, geotiff_filename, textured_mesh_filename);
    }

    USM_DECISION_TABLE(Transition::NEXT, );
}

Pipeline::Impl::Transition Pipeline::Impl::complete()
{
    USM_DECISION_TABLE(Transition::REPEAT, );
}

void Pipeline::Impl::rebuildGPSLocationsTree()
{
    imageGPSLocations = jk::tree::KDTree<size_t, 2>();

    for (auto it = graph.cnodebegin(); it != graph.cnodeend(); ++it)
    {
        const auto &node = it->second;
        std::array<double, 2> gps_pos = {node.payload.position.x(), node.payload.position.y()};
        imageGPSLocations.addPoint(gps_pos, it->first);
    }
}

} // namespace opencalibration
