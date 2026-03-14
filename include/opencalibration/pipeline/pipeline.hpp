#pragma once

#include <opencalibration/geo_coord/geo_coord.hpp>
#include <opencalibration/pipeline/progress.hpp>
#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/pipeline_state.hpp>
#include <opencalibration/types/surface_model.hpp>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace opencalibration
{

class Pipeline
{
  public:
    Pipeline(size_t batch_size = 1, size_t parallelism = 0);
    ~Pipeline();
    Pipeline(Pipeline &&) noexcept;
    Pipeline &operator=(Pipeline &&) noexcept;

    void add(const std::vector<std::string> &filename);

    // Warning: not threadsafe, only access from callback or when finished
    [[nodiscard]] const MeasurementGraph &getGraph() const;
    [[nodiscard]] const GeoCoord &getCoord() const;
    [[nodiscard]] const std::vector<surface_model> &getSurfaces() const;

    void set_callback(const StepCompletionCallback &step_complete);
    void set_generate_thumbnails(bool generate);
    void set_thumbnail_filenames(const std::string &thumbnail, const std::string &source, const std::string &overlap);
    void set_geotiff_filename(const std::string &geotiff);
    void set_dsm_filename(const std::string &dsm);
    void set_textured_mesh_filename(const std::string &path);
    void set_orthomosaic_max_megapixels(double max_megapixels);
    void set_skip_mesh_refinement(bool skip);
    void set_skip_initial_global_relax(bool skip);
    void set_skip_camera_param_relax(bool skip);
    void set_skip_final_global_relax(bool skip);
    void set_generate_dense_mesh(bool generate);

    bool saveCheckpoint(const std::string &checkpoint_dir);
    bool loadCheckpoint(const std::string &checkpoint_dir);
    bool resumeFromState(PipelineState target_state);

    void iterateOnce();
    [[nodiscard]] PipelineState getState() const;

    static std::string toString(PipelineState state);
    static std::optional<PipelineState> fromString(const std::string &str);

  private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace opencalibration
