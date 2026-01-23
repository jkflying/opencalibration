#include <opencalibration/pipeline/pipeline.hpp>

#include <opencalibration/pipeline/link_stage.hpp>
#include <opencalibration/pipeline/load_stage.hpp>
#include <opencalibration/pipeline/relax_stage.hpp>

#include <opencalibration/combinatorics/interleave.hpp>
#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/io/checkpoint.hpp>
#include <opencalibration/io/cv_raster_conversion.hpp>
#include <opencalibration/ortho/ortho.hpp>
#include <opencalibration/performance/performance.hpp>
#include <opencalibration/surface/intersect.hpp>

#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

#include <chrono>
#include <iostream>
#include <omp.h>

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
    PipelineState s = State::COMPLETE;

    // clang-format off
    USM_TABLE(currentState, State::COMPLETE, s,
        USM_STATE(transition, State::INITIAL_PROCESSING,
                  USM_MAP(Transition::NEXT, State::INITIAL_GLOBAL_RELAX, s));
        USM_STATE(transition, State::INITIAL_GLOBAL_RELAX,
                  USM_MAP(Transition::NEXT, State::CAMERA_PARAMETER_RELAX, s));
        USM_STATE(transition, State::CAMERA_PARAMETER_RELAX,
                  USM_MAP(Transition::NEXT, State::FINAL_GLOBAL_RELAX, s));
        USM_STATE(transition, State::FINAL_GLOBAL_RELAX,
                  USM_MAP(Transition::NEXT, _generate_thumbnails ? State::GENERATE_THUMBNAIL :
                                           _generate_geotiff ? State::GENERATE_GEOTIFF : State::COMPLETE, s));
        USM_STATE(transition, State::GENERATE_THUMBNAIL,
                  USM_MAP(Transition::NEXT, _generate_geotiff ? State::GENERATE_GEOTIFF : State::COMPLETE, s));
        USM_STATE(transition, State::GENERATE_GEOTIFF,
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
              USM_MAP(State::COMPLETE, complete(), t);
              USM_MAP(State::GENERATE_THUMBNAIL, generate_thumbnail(), t);
              USM_MAP(State::GENERATE_GEOTIFF, generate_geotiff(), t)
    );
    // clang-format on

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
    }

    USM_DECISION_TABLE(
        Transition::REPEAT,
        USM_MAKE_DECISION(_next_loaded_ids.size() == 0 && _next_linked_ids.size() == 0, Transition::NEXT));
}

Pipeline::Transition Pipeline::initial_global_relax()
{
    _relax_stage->init(_graph, {}, _imageGPSLocations, true, false, {Option::ORIENTATION, Option::GROUND_MESH});

    fvec relax_funcs = _relax_stage->get_runners(_graph);
    run_parallel(relax_funcs, _parallelism);
    spdlog::info("global relaxed all");
    _next_relaxed_ids = _relax_stage->finalize(_graph);
    _surfaces = _relax_stage->getSurfaceModels();

    USM_DECISION_TABLE(Transition::REPEAT, USM_MAKE_DECISION(stateRunCount() >= 5, Transition::NEXT));
}

Pipeline::Transition Pipeline::camera_parameter_relax()
{
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

    USM_DECISION_TABLE(Transition::REPEAT, USM_MAKE_DECISION(stateRunCount() >= 5, Transition::NEXT));
}

Pipeline::Transition Pipeline::final_global_relax()
{
    const bool lastIteration = stateRunCount() >= 3;

    _relax_stage->init(_graph, {}, _imageGPSLocations, true, lastIteration, {Option::ORIENTATION, Option::GROUND_MESH});

    fvec relax_funcs = _relax_stage->get_runners(_graph);
    run_parallel(relax_funcs, _parallelism);
    _next_relaxed_ids = _relax_stage->finalize(_graph);
    _surfaces = _relax_stage->getSurfaceModels();

    USM_DECISION_TABLE(Transition::REPEAT, USM_MAKE_DECISION(stateRunCount() >= 3, Transition::NEXT));
}

Pipeline::Transition Pipeline::generate_thumbnail()
{
    if (_thumbnail_filename.empty() && _source_filename.empty() && _overlap_filename.empty())
    {
        USM_DECISION_TABLE(Transition::NEXT, );
    }

    if (_surfaces.empty())
    {
        spdlog::warn("No surfaces available for thumbnail generation");
        USM_DECISION_TABLE(Transition::NEXT, );
    }

    auto thumbnail = orthomosaic::generateOrthomosaic(_surfaces, _graph);

    if (!_thumbnail_filename.empty())
        cv::imwrite(_thumbnail_filename, rasterToCv(thumbnail.pixelValues));
    if (!_source_filename.empty())
        cv::imwrite(_source_filename, rasterToCv(thumbnail.cameraUUID));
    if (!_overlap_filename.empty())
        cv::imwrite(_overlap_filename, rasterToCv(thumbnail.overlap));

    USM_DECISION_TABLE(Transition::NEXT, );
}

Pipeline::Transition Pipeline::generate_geotiff()
{
    if (_geotiff_filename.empty() && _dsm_filename.empty())
    {
        USM_DECISION_TABLE(Transition::NEXT, );
    }

    if (!_geotiff_filename.empty())
    {
        orthomosaic::generateGeoTIFFOrthomosaic(_surfaces, _graph, _coordinate_system, _geotiff_filename);
    }

    if (!_dsm_filename.empty())
    {
        orthomosaic::generateDSMGeoTIFF(_surfaces, _graph, _coordinate_system, _dsm_filename);
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
    case State::GENERATE_THUMBNAIL:
        return "Generate Thumbnail";
    case State::GENERATE_GEOTIFF:
        return "Generate GeoTIFF";
    case State::COMPLETE:
        return "Complete";
    };

    return "Error";
}

PipelineState Pipeline::fromString(const std::string &str)
{
    if (str == "INITIAL_PROCESSING" || str == "Initial Processing")
        return State::INITIAL_PROCESSING;
    if (str == "INITIAL_GLOBAL_RELAX" || str == "Initial global Relax")
        return State::INITIAL_GLOBAL_RELAX;
    if (str == "CAMERA_PARAMETER_RELAX" || str == "Camera Parameter Relax")
        return State::CAMERA_PARAMETER_RELAX;
    if (str == "FINAL_GLOBAL_RELAX" || str == "Final Global Relax")
        return State::FINAL_GLOBAL_RELAX;
    if (str == "GENERATE_THUMBNAIL" || str == "Generate Thumbnail")
        return State::GENERATE_THUMBNAIL;
    if (str == "GENERATE_GEOTIFF" || str == "Generate GeoTIFF")
        return State::GENERATE_GEOTIFF;
    if (str == "COMPLETE" || str == "Complete")
        return State::COMPLETE;

    spdlog::warn("Unknown pipeline state: {}, defaulting to INITIAL_PROCESSING", str);
    return State::INITIAL_PROCESSING;
}

void Pipeline::rebuildGPSLocationsTree()
{
    _imageGPSLocations = jk::tree::KDTree<size_t, 3>();

    for (auto it = _graph.cnodebegin(); it != _graph.cnodeend(); ++it)
    {
        const auto &node = it->second;
        std::array<double, 3> gps_pos = {node.payload.position.x(), node.payload.position.y(),
                                         node.payload.position.z()};
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
