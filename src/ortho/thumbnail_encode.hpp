#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <eigen3/Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace opencalibration::orthomosaic
{
namespace detail
{
inline std::string base64_encode(const uint8_t *data, size_t len)
{
    static constexpr char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    result.reserve(4 * ((len + 2) / 3));
    for (size_t i = 0; i < len; i += 3)
    {
        uint32_t n = static_cast<uint32_t>(data[i]) << 16;
        if (i + 1 < len)
            n |= static_cast<uint32_t>(data[i + 1]) << 8;
        if (i + 2 < len)
            n |= static_cast<uint32_t>(data[i + 2]);
        result += table[(n >> 18) & 0x3F];
        result += table[(n >> 12) & 0x3F];
        result += (i + 1 < len) ? table[(n >> 6) & 0x3F] : '=';
        result += (i + 2 < len) ? table[n & 0x3F] : '=';
    }
    return result;
}
} // namespace detail

inline std::string encodeThumbnailToBase64JPEG(const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> &b,
                                               const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> &g,
                                               const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> &r)
{
    int rows = static_cast<int>(b.rows());
    int cols = static_cast<int>(b.cols());
    cv::Mat bgr(rows, cols, CV_8UC3);
    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            bgr.at<cv::Vec3b>(y, x) = {b(y, x), g(y, x), r(y, x)};
        }
    }
    std::vector<uint8_t> buf;
    cv::imencode(".jpg", bgr, buf);
    return detail::base64_encode(buf.data(), buf.size());
}

} // namespace opencalibration::orthomosaic
