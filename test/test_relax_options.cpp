#include <sstream>

#include <opencalibration/types/relax_options.hpp>

#include <gtest/gtest.h>

using namespace opencalibration;

TEST(relax_options, count)
{
    RelaxOptionSet empty;
    EXPECT_EQ(0, empty.count());

    RelaxOptionSet two{Option::ORIENTATION, Option::POSITION};
    EXPECT_EQ(2, two.count());
}

TEST(relax_options, excess)
{
    RelaxOptionSet big{Option::ORIENTATION, Option::POSITION, Option::FOCAL_LENGTH};
    RelaxOptionSet small{Option::ORIENTATION};
    EXPECT_EQ(2, big.excess(small));
    EXPECT_EQ(-2, small.excess(big));
}

TEST(relax_options, hasAll)
{
    RelaxOptionSet superset{Option::ORIENTATION, Option::POSITION, Option::FOCAL_LENGTH};
    RelaxOptionSet subset{Option::ORIENTATION, Option::POSITION};
    RelaxOptionSet disjoint{Option::GROUND_PLANE};

    EXPECT_TRUE(superset.hasAll(subset));
    EXPECT_FALSE(subset.hasAll(superset));
    EXPECT_FALSE(superset.hasAll(disjoint));
}

TEST(relax_options, operator_eq)
{
    RelaxOptionSet a{Option::ORIENTATION, Option::POSITION};
    RelaxOptionSet b{Option::ORIENTATION, Option::POSITION};
    RelaxOptionSet c{Option::ORIENTATION};

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
}
