#pragma once

#include <cstdint>

namespace opencalibration
{

inline uint32_t xy2d(int order, int x, int y)
{
    uint32_t d = 0;
    for (int s = order / 2; s > 0; s /= 2)
    {
        int rx = (x & s) > 0 ? 1 : 0;
        int ry = (y & s) > 0 ? 1 : 0;
        d += s * s * ((3 * rx) ^ ry);
        if (ry == 0)
        {
            if (rx == 1)
            {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            std::swap(x, y);
        }
    }
    return d;
}

} // namespace opencalibration
