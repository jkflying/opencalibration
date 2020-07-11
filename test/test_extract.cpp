#include <opencalibration/oc_extract/extract_features.hpp>

#include <gtest/gtest.h>

TEST(extract, doesnt_crash)
{
    auto res = opencalibration::extract_features("something.jpg");
}
