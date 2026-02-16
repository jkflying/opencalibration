#pragma once

#include <optional>
#include <string>

namespace opencalibration
{

enum class PipelineState
{
    INITIAL_PROCESSING,
    INITIAL_GLOBAL_RELAX,
    CAMERA_PARAMETER_RELAX,
    FINAL_GLOBAL_RELAX,
    MESH_REFINEMENT,
    GENERATE_THUMBNAIL,
    GENERATE_LAYERS,
    COLOR_BALANCE,
    BLEND_LAYERS,
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
    case PipelineState::GENERATE_LAYERS:
        return "GENERATE_LAYERS";
    case PipelineState::COLOR_BALANCE:
        return "COLOR_BALANCE";
    case PipelineState::BLEND_LAYERS:
        return "BLEND_LAYERS";
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
    if (str == "GENERATE_LAYERS" || str == "GENERATE_GEOTIFF" || str == "GENERATE_DSM")
        return PipelineState::GENERATE_LAYERS;
    if (str == "COLOR_BALANCE")
        return PipelineState::COLOR_BALANCE;
    if (str == "BLEND_LAYERS")
        return PipelineState::BLEND_LAYERS;
    if (str == "COMPLETE")
        return PipelineState::COMPLETE;
    return std::nullopt;
}

} // namespace opencalibration
