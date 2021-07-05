
#include <opencalibration/combinatorics/interleave.hpp>

#include <gtest/gtest.h>

TEST(interleave, works_with_empty)
{
    using vec = std::vector<std::string>;
    vec a = opencalibration::interleave<vec>({});
    EXPECT_EQ(a.size(), 0);
}

TEST(interleave, works_with_empty_subvectors)
{
    using vec = std::vector<std::string>;
    vec a;
    vec b;
    vec res = opencalibration::interleave<vec>({a, b});
    EXPECT_EQ(res.size(), 0);
}

TEST(interleave, works_with_some_data)
{
    using vec = std::vector<std::string>;
    vec a{"a0", "a1", "a2", "a3"};
    vec b{"b0", "b1", "b2", "b3", "b4", "b5"};
    vec c{"c0"};

    vec res = opencalibration::interleave<vec>({a, b, c});

    vec expected{"a0", "b0", "c0", "a1", "b1", "b2", "a2", "b3", "b4", "a3", "b5"};
    EXPECT_EQ(res.size(), expected.size());
    EXPECT_EQ(res, expected);
}
