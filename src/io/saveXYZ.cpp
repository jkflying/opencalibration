#include <opencalibration/io/saveXYZ.hpp>

namespace opencalibration
{

bool toXYZ(const std::vector<surface_model> &surfaces, std::ostream &out)
{
    std::ostringstream buffer;

    for (const auto &s : surfaces)
    {
        for (const auto &c : s.cloud)
        {
            for (const auto &p : c)
            {
                buffer << p.x() << "," << p.y() << "," << p.z() << "\n";
            }
            out << buffer.str();
            buffer.str("");
            buffer.clear();
        }
    }
    return true;
}

} // namespace opencalibration
