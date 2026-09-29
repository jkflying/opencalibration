#pragma once

#include <optional>
#include <string>

namespace opencalibration
{

// Declared in execution order: resumeFromState relies on it
enum class PipelineState
{
    INITIAL_PROCESSING,
    MESH_REFINEMENT,
    INITIAL_GLOBAL_RELAX,
    CAMERA_PARAMETER_RELAX,
    FINAL_GLOBAL_RELAX,
    DENSIFY_MESH,
    DENSE_MESH_RELAX,
    GENERATE_THUMBNAIL,
    GENERATE_GEOTIFF,
    COMPLETE
};

inline std::string pipelineStateToString(PipelineState state)
{
    switch (state)
    {
    case PipelineState::INITIAL_PROCESSING:
        return "INITIAL_PROCESSING";
    case PipelineState::INITIAL_GLOBAL_RELAX:
        return "INITIAL_GLOBAL_RELAX";
    case PipelineState::CAMERA_PARAMETER_RELAX:
        return "CAMERA_PARAMETER_RELAX";
    case PipelineState::FINAL_GLOBAL_RELAX:
        return "FINAL_GLOBAL_RELAX";
    case PipelineState::MESH_REFINEMENT:
        return "MESH_REFINEMENT";
    case PipelineState::GENERATE_THUMBNAIL:
        return "GENERATE_THUMBNAIL";
    case PipelineState::DENSIFY_MESH:
        return "DENSIFY_MESH";
    case PipelineState::DENSE_MESH_RELAX:
        return "DENSE_MESH_RELAX";
    case PipelineState::GENERATE_GEOTIFF:
        return "GENERATE_GEOTIFF";
    case PipelineState::COMPLETE:
        return "COMPLETE";
    }
    return "";
}

inline std::optional<PipelineState> stringToPipelineState(const std::string &str)
{
    if (str == "INITIAL_PROCESSING")
        return PipelineState::INITIAL_PROCESSING;
    if (str == "INITIAL_GLOBAL_RELAX")
        return PipelineState::INITIAL_GLOBAL_RELAX;
    if (str == "CAMERA_PARAMETER_RELAX")
        return PipelineState::CAMERA_PARAMETER_RELAX;
    if (str == "FINAL_GLOBAL_RELAX")
        return PipelineState::FINAL_GLOBAL_RELAX;
    if (str == "MESH_REFINEMENT")
        return PipelineState::MESH_REFINEMENT;
    if (str == "GENERATE_THUMBNAIL")
        return PipelineState::GENERATE_THUMBNAIL;
    if (str == "DENSIFY_MESH")
        return PipelineState::DENSIFY_MESH;
    if (str == "DENSE_MESH_RELAX")
        return PipelineState::DENSE_MESH_RELAX;
    if (str == "GENERATE_GEOTIFF" || str == "GENERATE_LAYERS" || str == "BLEND_LAYERS" || str == "COLOR_BALANCE" ||
        str == "GENERATE_DSM")
        return PipelineState::GENERATE_GEOTIFF;
    if (str == "COMPLETE")
        return PipelineState::COMPLETE;
    return std::nullopt;
}

} // namespace opencalibration
