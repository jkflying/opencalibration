#include <gtest/gtest.h>

#include <jk/KDTree.h>
#include <jk/KMeans.h>

TEST(kdtree, initializes)
{
    jk::tree::KDTree<int, 3> tree;
}

TEST(kdtree, finds_nearest)
{
    using tree_t = jk::tree::KDTree<std::string, 2>;
    using point_t = std::array<double, 2>;
    tree_t tree;
    tree.addPoint(point_t{{1, 2}}, "George");
    tree.addPoint(point_t{{1, 3}}, "Harold");
    tree.addPoint(point_t{{7, 7}}, "Melvin");

    // KNN search
    point_t lazyMonsterLocation{{6, 6}}; // this monster will always try to eat the closest people
    const std::size_t monsterHeads = 2;  // this monster can eat two people at once
    auto lazyMonsterVictims = tree.searchKnn(lazyMonsterLocation, monsterHeads);

    ASSERT_EQ(lazyMonsterVictims.size(), 2);
    EXPECT_EQ(lazyMonsterVictims[0].payload, "Melvin");
    EXPECT_DOUBLE_EQ(lazyMonsterVictims[0].distance, 2); // distance is squared
    EXPECT_EQ(lazyMonsterVictims[1].payload, "Harold");
    EXPECT_DOUBLE_EQ(lazyMonsterVictims[1].distance, 3 * 3 + 5 * 5); // distance is squared

    // ball search
    point_t stationaryMonsterLocation{{8, 8}}; // this monster doesn't move, so can only eat people that are close
    const double neckLength = 6.0;             // it can only reach within this range
    auto potentialVictims = tree.searchBall(stationaryMonsterLocation, neckLength * neckLength); // metric is SquaredL2
    ASSERT_EQ(potentialVictims.size(), 1);
    EXPECT_EQ(potentialVictims[0].payload, "Melvin");
    EXPECT_DOUBLE_EQ(potentialVictims[0].distance, 2); // distance is squared
}

TEST(kmeans, clusters)
{
    jk::tree::KMeans<size_t, 2> kmeans(20);
    for (size_t i = 0; i < 50; i++)
        for (size_t j = 0; j < 20; j++)
            kmeans.add({(double)i, (double)j}, 5 * i + j);
    // print clusters
    auto print_clusters = [&]() {
        std::cout << "clusters: " << kmeans.getClusters().front().points.size() << " - "
                  << kmeans.getClusters().back().points.size() << " entries" << std::endl;
    };

    print_clusters();

    for (int i = 0; i < 12; i++)
    {
        kmeans.iterate();
    }
    print_clusters();
}
