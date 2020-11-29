#pragma once

#include <vector>

namespace opencalibration
{
struct NodeLinks
{
    size_t node_id;
    std::vector<size_t> link_ids;
};
}
