#pragma once

#include <functional>
#include <vector>

#include <iostream>

namespace opencalibration
{

template <typename vec>
vec interleave(std::initializer_list<typename std::reference_wrapper<vec>> refs, bool full_dispersal = true)
{
    vec interleaved;
    int total_count = 0;
    struct vec_info
    {
        size_t original_size;
        size_t iterator_position;
    };
    std::vector<vec_info> counts(refs.size());
    size_t i = 0;
    for (const auto &r : refs)
    {
        counts[i] = {r.get().size(), 0};
        total_count += counts[i].original_size;
        i++;
    }
    interleaved.reserve(total_count);

    bool some_added;
    do
    {
        some_added = false;
        i = 0;
        for (auto &r : refs)
        {
            vec_info &info = counts[i];

            if (info.iterator_position < info.original_size &&
                // iter_pos / orig <= inter_size / total --> total * iter_pos <= inter_size * orig
                (!full_dispersal || total_count * info.iterator_position <= interleaved.size() * info.original_size))
            {
                interleaved.emplace_back(std::move(r.get()[info.iterator_position++]));
                some_added = true;
            }
            i++;
        }
    } while (some_added);
    return interleaved;
}

} // namespace opencalibration
