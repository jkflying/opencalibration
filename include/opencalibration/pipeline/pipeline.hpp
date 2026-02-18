#pragma once

#include <opencalibration/geo_coord/geo_coord.hpp>
#include <opencalibration/ortho/color_balance.hpp>
#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/pipeline_state.hpp>
#include <opencalibration/types/surface_model.hpp>

#include <jk/KDTree.h>
#include <usm.hpp>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace opencalibration
{
class LoadStage;
class LinkStage;
class RelaxStage;

enum class PipelineTransition
{
    REPEAT,
    NEXT,
    ERROR
};

class Pipeline : public usm::StateMachine<PipelineState, PipelineTransition>
{
  public:
    struct StepCompletionInfo
    {
        std::reference_wrapper<std::vector<size_t>> loaded_ids, linked_ids;
        std::reference_wrapper<std::vector<std::vector<size_t>>> relaxed_ids;
        std::reference_wrapper<std::vector<surface_model>> surfaces;
        size_t images_loaded, queue_size_remaining;
        PipelineState state;
        uint64_t state_iteration;
    };
    using StepCompletionCallback = std::function<void(const StepCompletionInfo &)>;

    Pipeline(size_t batch_size = 1, size_t parallelism = 0); // parallelism = 0 --> unlimited
    ~Pipeline();

    void add(const std::vector<std::string> &filename);

    // warning - not threadsafe, only access from callback or when finished
    [[nodiscard]] const MeasurementGraph &getGraph()
    {
        return _graph;
    }

    [[nodiscard]] const GeoCoord &getCoord() const
    {
        return _coordinate_system;
    }

    const std::vector<surface_model> &getSurfaces()
    {
        return _surfaces;
    }

    void set_callback(const StepCompletionCallback &step_complete)
    {
        _step_callback = step_complete;
    }

    void set_generate_thumbnails(bool generate)
    {
        _generate_thumbnails = generate;
    }

    void set_thumbnail_filenames(const std::string &thumbnail, const std::string &source, const std::string &overlap)
    {
        _thumbnail_filename = thumbnail;
        _source_filename = source;
        _overlap_filename = overlap;
    }

    void set_geotiff_filename(const std::string &geotiff)
    {
        _geotiff_filename = geotiff;
        if (!geotiff.empty())
        {
            _generate_geotiff = true;
        }
    }

    void set_dsm_filename(const std::string &dsm)
    {
        _dsm_filename = dsm;
        if (!dsm.empty())
        {
            _generate_geotiff = true; // DSM generation uses the same pipeline stage
        }
    }

    void set_orthomosaic_max_megapixels(double max_megapixels)
    {
        _orthomosaic_max_megapixels = max_megapixels;
    }

    // Stage configuration - allows skipping pipeline stages
    void set_skip_mesh_refinement(bool skip)
    {
        _skip_mesh_refinement = skip;
    }
    void set_skip_initial_global_relax(bool skip)
    {
        _skip_initial_global_relax = skip;
    }
    void set_skip_camera_param_relax(bool skip)
    {
        _skip_camera_param_relax = skip;
    }
    void set_skip_final_global_relax(bool skip)
    {
        _skip_final_global_relax = skip;
    }

    bool saveCheckpoint(const std::string &checkpoint_dir);
    bool loadCheckpoint(const std::string &checkpoint_dir);
    bool resumeFromState(PipelineState target_state);

    static std::string toString(PipelineState state);
    static PipelineState fromString(const std::string &str);

  protected:
    PipelineState chooseNextState(PipelineState currentState, PipelineTransition transition) override;
    PipelineTransition runCurrentState(PipelineState currentState) override;

  private:
    PipelineTransition initial_processing();
    PipelineTransition initial_global_relax();
    PipelineTransition camera_parameter_relax();
    PipelineTransition final_global_relax();
    PipelineTransition mesh_refinement();
    PipelineTransition generate_thumbnail();
    PipelineTransition generate_layers();
    PipelineTransition color_balance();
    PipelineTransition blend_layers();
    PipelineTransition complete();

    void rebuildGPSLocationsTree();

    std::vector<size_t> _previous_loaded_ids, _previous_linked_ids, _next_loaded_ids, _next_linked_ids;

    std::vector<std::vector<size_t>> _next_relaxed_ids;

    std::unique_ptr<LoadStage> _load_stage;
    std::unique_ptr<LinkStage> _link_stage;
    std::unique_ptr<RelaxStage> _relax_stage;

    std::condition_variable _queue_condition_variable;
    std::mutex _queue_mutex;
    std::deque<std::string> _add_queue;

    MeasurementGraph _graph;
    std::vector<surface_model> _surfaces;
    jk::tree::KDTree<size_t, 2> _imageGPSLocations;
    GeoCoord _coordinate_system;

    StepCompletionCallback _step_callback;

    size_t _batch_size;
    size_t _parallelism;
    bool _generate_thumbnails = true;
    std::string _thumbnail_filename = "";
    std::string _source_filename = "";
    std::string _overlap_filename = "";
    bool _generate_geotiff = false;
    std::string _geotiff_filename = "";
    std::string _dsm_filename = "";
    double _orthomosaic_max_megapixels = 0.0;

    // Stage skip flags
    bool _skip_mesh_refinement = false;
    bool _skip_initial_global_relax = false;
    bool _skip_camera_param_relax = false;
    bool _skip_final_global_relax = false;

    std::vector<orthomosaic::ColorCorrespondence> _correspondences;
    orthomosaic::ColorBalanceResult _color_balance_result;
    std::string _intermediate_layers_path;
    std::string _intermediate_cameras_path;
};
} // namespace opencalibration
