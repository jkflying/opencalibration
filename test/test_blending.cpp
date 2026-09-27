#include <opencalibration/ortho/blending.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

using namespace opencalibration::orthomosaic;

TEST(Blending, compute_blend_weight_center)
{
    // GIVEN: A pixel at the center of a 100x100 image
    float weight = computeBlendWeight(50, 50, 100, 100, 10.0f);

    // THEN: Weight should be positive and relatively high
    EXPECT_GT(weight, 0.0f);
}

TEST(Blending, compute_blend_weight_edge)
{
    // GIVEN: A pixel at the edge of a 100x100 image
    float weight_edge = computeBlendWeight(0, 50, 100, 100, 10.0f);
    float weight_center = computeBlendWeight(50, 50, 100, 100, 10.0f);

    // THEN: Edge weight should be less than center weight
    EXPECT_LT(weight_edge, weight_center);
}

TEST(Blending, compute_blend_weight_proximity)
{
    // GIVEN: Same pixel position but different camera distances
    float weight_near = computeBlendWeight(50, 50, 100, 100, 5.0f);
    float weight_far = computeBlendWeight(50, 50, 100, 100, 50.0f);

    // THEN: Closer camera should have higher weight
    EXPECT_GT(weight_near, weight_far);
}
