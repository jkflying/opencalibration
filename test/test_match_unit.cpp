#include <opencalibration/match/match_features.hpp>

#include <gtest/gtest.h>

using namespace opencalibration;

TEST(match_unit, each_feature_matched_at_most_once)
{
    // GIVEN: two source features that are both closest to the same target feature
    auto featureWithBits = [](int first, int count) {
        feature_2d f;
        for (int i = first; i < first + count; i++)
            f.descriptor.set(i, true);
        return f;
    };
    const std::vector<feature_2d> set_1 = {featureWithBits(0, 100), featureWithBits(0, 102)};
    const std::vector<feature_2d> set_2 = {featureWithBits(0, 100), featureWithBits(200, 200)};

    // WHEN: matching all features
    const auto matches = match_features_subset(set_1, set_2, {0, 1}, {0, 1});

    // THEN: only the closer source feature keeps the shared target
    ASSERT_EQ(1u, matches.size());
    EXPECT_EQ(0u, matches[0].feature_index_1);
    EXPECT_EQ(0u, matches[0].feature_index_2);
}
