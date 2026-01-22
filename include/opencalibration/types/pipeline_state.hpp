#pragma once

#include <string>

namespace opencalibration
{

enum class PipelineState
{
    INITIAL_PROCESSING,
    INITIAL_GLOBAL_RELAX,
    CAMERA_PARAMETER_RELAX,
    FINAL_GLOBAL_RELAX,
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
    case PipelineState::GENERATE_THUMBNAIL:
        return "GENERATE_THUMBNAIL";
    case PipelineState::GENERATE_GEOTIFF:
        return "GENERATE_GEOTIFF";
    case PipelineState::COMPLETE:
        return "COMPLETE";
    }
    return "UNKNOWN";
}

inline PipelineState stringToPipelineState(const std::string &str)
{
    if (str == "INITIAL_PROCESSING")
        return PipelineState::INITIAL_PROCESSING;
    if (str == "INITIAL_GLOBAL_RELAX")
        return PipelineState::INITIAL_GLOBAL_RELAX;
    if (str == "CAMERA_PARAMETER_RELAX")
        return PipelineState::CAMERA_PARAMETER_RELAX;
    if (str == "FINAL_GLOBAL_RELAX")
        return PipelineState::FINAL_GLOBAL_RELAX;
    if (str == "GENERATE_THUMBNAIL")
        return PipelineState::GENERATE_THUMBNAIL;
    if (str == "GENERATE_GEOTIFF")
        return PipelineState::GENERATE_GEOTIFF;
    if (str == "COMPLETE")
        return PipelineState::COMPLETE;
    return PipelineState::INITIAL_PROCESSING;
}

} // namespace opencalibration
