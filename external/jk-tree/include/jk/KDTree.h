#pragma once

/**
 * KDTree.h by Julian Kent
 * A C++11 KD-Tree with the following features:
 *     single file
 *     header only
 *     high performance K Nearest Neighbor and ball searches
 *     dynamic insertions
 *     simple API
 *     depends only on the STL
 *     templatable on your custom data type to store in the leaves. No need to keep a separate data structure!
 *     templatable on double, float etc
 *     templatable on L1, SquaredL2 or custom distance functor
 *     templated on number of dimensions for efficient inlining
 *     iterator support for easy traversal of all points
 *
 * -------------------------------------------------------------------
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not
 * distributed with this  file, you can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * A high level explanation of MPLv2: You may use this in any software provided you give attribution. You *must* make
 * available any changes you make to the source code of this file to anybody you distribute your software to.
 *
 * Upstreaming features and bugfixes are highly appreciated via:
 *
 * https://github.com/jkflying/bucket-pr-kdtree/tree/master/C%2B%2B
 *
 * For additional licensing rights, feature requests or questions, please contact Julian Kent <jkflying@gmail.com>
 *
 * -------------------------------------------------------------------
 *
 * Example usage:
 *
 * // setup
 * using tree_t = jk::tree::KDTree<std::string, 2>;
 * using point_t = std::array<double, 2>;
 * tree_t tree;
 * tree.addPoint(point_t{{1, 2}}, "George");
 * tree.addPoint(point_t{{1, 3}}, "Harold");
 * tree.addPoint(point_t{{7, 7}}, "Melvin");
 *
 * // KNN search
 * point_t lazyMonsterLocation{{6, 6}}; // this monster will always try to eat the closest people
 * const std::size_t monsterHeads = 2; // this monster can eat two people at once
 * auto lazyMonsterVictims = tree.searchKnn(lazyMonsterLocation, monsterHeads);
 * for (const auto& victim : lazyMonsterVictims)
 * {
 *     std::cout << victim.payload << " closest to lazy monster, with distance " << sqrt(victim.distance) << "!"
 *               << std::endl;
 * }
 *
 * // ball search
 * point_t stationaryMonsterLocation{{8, 8}}; // this monster doesn't move, so can only eat people that are close
 * const double neckLength = 6.0; // it can only reach within this range
 * auto potentialVictims = tree.searchBall(stationaryMonsterLocation, neckLength * neckLength); // metric is SquaredL2
 * std::cout << "Stationary monster can reach any of " << potentialVictims.size() << " people!" << std::endl;
 *
 * // hybrid KNN/ball search
 * auto actualVictims
 *     = tree.searchCapacityLimitedBall(stationaryMonsterLocation, neckLength * neckLength, monsterHeads);
 * std::cout << "The stationary monster will try to eat ";
 * for (const auto& victim : actualVictims)
 * {
 *     std::cout << victim.payload << " and ";
 * }
 * std::cout << "nobody else." << std::endl;
 *
 * Output:
 *
 * Melvin closest to lazy monster, with distance 1.41421!
 * Harold closest to lazy monster, with distance 5.83095!
 * Stationary monster can reach any of 1 people!
 * The stationary monster will try to eat Melvin and nobody else.
 *
 * -------------------------------------------------------------------
 *
 * Tuning tips:
 *
 * If you need to add a lot of points before doing any queries, set the optional `autosplit` parameter to false,
 * then call splitOutstanding(). This will reduce temporaries and result in a better balanced tree.
 *
 * Set the bucket size to be at least twice the K in a typical KNN query. If you have more dimensions, it is better to
 * have a larger bucket size. 32 is a good starting point. If possible use powers of 2 for the bucket size.
 *
 * If you experience linear search performance, check that you don't have a bunch of duplicate point locations. This
 * will result in the tree being unable to split the bucket the points are in, degrading search performance.
 *
 * The tree adapts to the parallel-to-axis dimensionality of the problem. Thus, if there is one dimension with a much
 * larger scale than the others, most of the splitting will happen on this dimension. This is achieved by trying to
 * keep the bounding boxes of the data in the buckets equal lengths in all axes.
 *
 * Random data performs worse than 'real world' data with structure. This is because real world data has tighter
 * bounding boxes, meaning more branches of the tree can be eliminated sooner.
 *
 * On pure random data, more than 7 dimensions won't be much faster than linear. However, most data isn't actually
 * random. The tree will adapt to any locally reduced dimensionality, which is found in most real world data.
 *
 * Hybrid ball/KNN searches are faster than either type on its own, because subtrees can be more aggresively eliminated.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <queue>
#include <set>
#include <vector>

namespace jk
{
namespace tree
{
    template <std::size_t Dim> struct Loop;
    struct L1;
    struct SquaredL2;

    template <class Payload,
              std::size_t Dimensions,
              std::size_t BucketSize = 32,
              class Distance = SquaredL2,
              typename Scalar = double>
    class KDTree
    {
    public:
        using distance_t = Distance;
        using scalar_t = Scalar;
        using payload_t = Payload;
        using point_t = std::array<Scalar, Dimensions>;
        static const std::size_t dimensions = Dimensions;
        static const std::size_t bucketSize = BucketSize;
        using tree_t = KDTree<Payload, Dimensions, BucketSize, Distance, Scalar>;

        struct LocationPayload
        {
            point_t location;
            Payload payload;
            bool operator==(const LocationPayload& other) const
            {
                return location == other.location && payload == other.payload;
            }
        };

        struct DistancePayload
        {
            Scalar distance;
            Payload payload;
            bool operator<(const DistancePayload& dp) const { return distance < dp.distance; }
        };

        class Iterator;
        class Searcher;
        enum class Strategy
        {
            Expand,
            Contract
        };

        KDTree()
            : m_nodes {{0, 0}}
            , m_nodeBounds {{std::numeric_limits<Scalar>::max(), std::numeric_limits<Scalar>::lowest()}}
        {
        }

        size_t size() const { return m_nodes.empty() ? 0 : m_nodes[0].m_entries; }

        void addPoint(const point_t& location, const Payload& payload, bool autosplit = true)
        {
            auto addOp = [&](std::size_t leafIdx)
            {
                std::size_t insertPos = m_nodes[leafIdx].m_dataEnd;
                m_data.insert(m_data.begin() + insertPos, LocationPayload {location, payload});
                m_nodes[leafIdx].m_dataEnd++;
                for (std::size_t i = 0; i < m_nodes.size(); ++i)
                {
                    if (i != leafIdx && m_nodes[i].m_splitDimension == Dimensions
                        && m_nodes[i].m_dataBegin >= insertPos)
                    {
                        m_nodes[i].m_dataBegin++;
                        m_nodes[i].m_dataEnd++;
                    }
                }
                return true;
            };
            std::size_t leafIdx = findLeaf(0, location, Strategy::Expand, addOp).first;
            if (m_nodes[leafIdx].shouldSplit() && m_nodes[leafIdx].m_entries % BucketSize == 0)
            {
                if (autosplit)
                    split(leafIdx);
                else if (!m_nodes[leafIdx].m_isWaitingForSplit)
                {
                    m_nodes[leafIdx].m_isWaitingForSplit = true;
                    m_waitingForSplit.push_back(leafIdx);
                }
            }
        }

        bool removePoint(const point_t& location, const Payload& payload)
        {
            auto removeOp = [&](std::size_t leafIdx)
            {
                Node& leaf = m_nodes[leafIdx];
                LocationPayload target {location, payload};
                for (std::size_t i = leaf.m_dataBegin; i < leaf.m_dataEnd; ++i)
                {
                    if (m_data[i] == target)
                    {
                        std::swap(m_data[i], m_data[leaf.m_dataEnd - 1]);
                        leaf.m_dataEnd--;
                        leaf.m_entries--;
                        return true;
                    }
                }
                return false;
            };
            return findLeaf(0, location, Strategy::Contract, removeOp).second;
        }

        void splitOutstanding()
        {
            std::vector<std::size_t> searchStack;
            std::swap(searchStack, m_waitingForSplit);
            while (!searchStack.empty())
            {
                std::size_t addNode = searchStack.back();
                searchStack.pop_back();
                m_nodes[addNode].m_isWaitingForSplit = false;
                if (m_nodes[addNode].m_splitDimension == Dimensions && m_nodes[addNode].shouldSplit() && split(addNode))
                {
                    searchStack.push_back(m_nodes[addNode].m_children.first);
                    searchStack.push_back(m_nodes[addNode].m_children.second);
                }
            }
        }

        void rebalance()
        {
            std::vector<LocationPayload> activeData;
            activeData.reserve(size());
            for (const auto& lp : *this)
                activeData.push_back(lp);
            m_data = std::move(activeData);
            m_nodes.assign(1, Node(0, m_data.size()));
            m_nodes[0].m_entries = m_data.size();
            m_nodeBounds.assign(1, bounds_t());
            m_nodeBounds[0].fill({std::numeric_limits<Scalar>::max(), std::numeric_limits<Scalar>::lowest()});
            m_waitingForSplit.clear();
            if (!m_data.empty())
            {
                recalculateLeafBounds(0);
                if (m_nodes[0].shouldSplit())
                {
                    m_waitingForSplit.push_back(0);
                    splitOutstanding();
                }
            }
        }

        std::vector<DistancePayload> searchKnn(const point_t& location, std::size_t maxPoints) const
        {
            return searcher().search(location, std::numeric_limits<Scalar>::max(), maxPoints);
        }
        std::vector<DistancePayload> searchBall(const point_t& location, Scalar maxRadius) const
        {
            return searcher().search(location, maxRadius, std::numeric_limits<std::size_t>::max());
        }
        std::vector<DistancePayload> searchCapacityLimitedBall(const point_t& location,
                                                               Scalar maxRadius,
                                                               std::size_t maxPoints) const
        {
            return searcher().search(location, maxRadius, maxPoints);
        }

        DistancePayload search(const point_t& location) const
        {
            DistancePayload result;
            result.distance = std::numeric_limits<Scalar>::infinity();
            if (size() > 0)
                searchRecursive(0, location, result);
            return result;
        }

        Searcher searcher() const { return Searcher(*this); }
        Iterator begin() const { return Iterator(this, 0, 0); }
        Iterator end() const { return Iterator(this, m_nodes.size(), 0); }

        class Iterator
        {
        public:
            using iterator_category = std::forward_iterator_tag;
            using value_type = LocationPayload;
            using difference_type = std::ptrdiff_t;
            using pointer = const LocationPayload*;
            using reference = const LocationPayload&;

            Iterator(const tree_t* tree, std::size_t nodeIndex, std::size_t pointIndex)
                : m_tree(tree), m_nodeIndex(nodeIndex), m_pointIndex(pointIndex)
            {
                advanceToNextValid();
            }
            Iterator() : m_tree(nullptr), m_nodeIndex(0), m_pointIndex(0) { }
            reference operator*() const { return m_tree->m_data[m_pointIndex]; }
            pointer operator->() const { return &m_tree->m_data[m_pointIndex]; }
            Iterator& operator++()
            {
                m_pointIndex++;
                advanceToNextValid();
                return *this;
            }
            Iterator operator++(int)
            {
                Iterator tmp = *this;
                ++(*this);
                return tmp;
            }
            bool operator==(const Iterator& other) const
            {
                if (m_tree != other.m_tree)
                    return false;
                if (m_tree == nullptr)
                    return true;
                return m_nodeIndex == other.m_nodeIndex && m_pointIndex == other.m_pointIndex;
            }
            bool operator!=(const Iterator& other) const { return !(*this == other); }

        private:
            const tree_t* m_tree;
            std::size_t m_nodeIndex, m_pointIndex;
            void advanceToNextValid()
            {
                if (!m_tree)
                    return;
                while (m_nodeIndex < m_tree->m_nodes.size())
                {
                    const auto& node = m_tree->m_nodes[m_nodeIndex];
                    if (node.m_splitDimension == Dimensions)
                    {
                        if (m_pointIndex < node.m_dataEnd)
                        {
                            if (m_pointIndex < node.m_dataBegin)
                                m_pointIndex = node.m_dataBegin;
                            return;
                        }
                    }
                    m_nodeIndex++;
                    m_pointIndex = 0;
                }
            }
        };

        class Searcher
        {
        public:
            Searcher(const tree_t& tree) : m_tree(tree) { }
            const std::vector<DistancePayload>& search(const point_t& location, Scalar maxRadius, std::size_t maxPoints)
            {
                m_results.clear();
                if (m_searchStack.capacity() == 0)
                    m_searchStack.reserve(1 + std::size_t(1.5 * std::log2(1 + m_tree.size() / BucketSize)));
                if (m_prioqueueCapacity < maxPoints && maxPoints < m_tree.size())
                {
                    std::vector<DistancePayload> container;
                    container.reserve(maxPoints);
                    m_prioqueue = std::priority_queue<DistancePayload, std::vector<DistancePayload>>(
                        std::less<DistancePayload>(), std::move(container));
                    m_prioqueueCapacity = maxPoints;
                }
                m_tree.searchCapacityLimitedBall(location, maxRadius, maxPoints, m_searchStack, m_prioqueue, m_results);
                m_prioqueueCapacity = std::max(m_prioqueueCapacity, m_results.size());
                return m_results;
            }

        private:
            const tree_t& m_tree;
            std::vector<std::size_t> m_searchStack;
            std::priority_queue<DistancePayload, std::vector<DistancePayload>> m_prioqueue;
            std::size_t m_prioqueueCapacity = 0;
            std::vector<DistancePayload> m_results;
        };

    private:
        struct Node
        {
            Node(std::size_t begin, std::size_t end) : m_dataBegin(begin), m_dataEnd(end) { }
            bool shouldSplit() const { return m_entries >= BucketSize; }
            void searchCapacityLimitedBall(const std::vector<LocationPayload>& data,
                                           const point_t& location,
                                           Scalar maxRadius,
                                           std::size_t K,
                                           std::priority_queue<DistancePayload>& results) const
            {
                std::size_t i = m_dataBegin;
                for (; results.size() < K && i < m_dataEnd; i++)
                {
                    Scalar distance = Distance::distance(location, data[i].location);
                    if (distance < maxRadius)
                        results.emplace(DistancePayload {distance, data[i].payload});
                }
                for (; i < m_dataEnd; i++)
                {
                    Scalar distance = Distance::distance(location, data[i].location);
                    if (distance < maxRadius && distance < results.top().distance)
                    {
                        results.pop();
                        results.emplace(DistancePayload {distance, data[i].payload});
                    }
                }
            }
            void queueChildren(const point_t& location, std::vector<std::size_t>& searchStack) const
            {
                if (location[m_splitDimension] < m_splitValue)
                {
                    searchStack.push_back(m_children.second);
                    searchStack.push_back(m_children.first);
                }
                else
                {
                    searchStack.push_back(m_children.first);
                    searchStack.push_back(m_children.second);
                }
            }
            std::size_t m_entries = 0, m_splitDimension = Dimensions;
            Scalar m_splitValue = 0;
            std::pair<std::size_t, std::size_t> m_children;
            std::size_t m_dataBegin = 0, m_dataEnd = 0;
            bool m_isWaitingForSplit = false;
        };

        struct Range
        {
            Scalar min, max;
            void expand(Scalar v)
            {
                if (v < min)
                    min = v;
                if (v > max)
                    max = v;
            }
            void expand(const Range& other)
            {
                if (other.min < min)
                    min = other.min;
                if (other.max > max)
                    max = other.max;
            }
        };
        using bounds_t = std::array<Range, Dimensions>;

        std::vector<Node> m_nodes;
        std::vector<bounds_t> m_nodeBounds;
        std::vector<std::size_t> m_waitingForSplit;
        std::vector<LocationPayload> m_data;

        void searchRecursive(std::size_t nodeIndex, const point_t& location, DistancePayload& best) const
        {
            const Node& node = m_nodes[nodeIndex];
            if (best.distance <= pointRectDist(nodeIndex, location))
                return;

            if (node.m_splitDimension == Dimensions)
            {
                for (std::size_t i = node.m_dataBegin; i < node.m_dataEnd; ++i)
                {
                    Scalar nodeDist = Distance::distance(location, m_data[i].location);
                    if (nodeDist < best.distance)
                        best = DistancePayload {nodeDist, m_data[i].payload};
                }
            }
            else
            {
                std::size_t nearChild = (location[node.m_splitDimension] < node.m_splitValue) ? node.m_children.first
                                                                                              : node.m_children.second;
                std::size_t farChild = (location[node.m_splitDimension] < node.m_splitValue) ? node.m_children.second
                                                                                             : node.m_children.first;
                searchRecursive(nearChild, location, best);
                searchRecursive(farChild, location, best);
            }
        }

        template <typename Op>
        std::pair<std::size_t, bool> findLeaf(std::size_t nodeIndex, const point_t& location, Strategy strategy, Op op)
        {
            if (m_nodes[nodeIndex].m_splitDimension == Dimensions)
            {
                bool modified = op(nodeIndex);
                if (modified)
                {
                    if (strategy == Strategy::Expand)
                        expandBounds(nodeIndex, location);
                    if (strategy == Strategy::Contract)
                        recalculateLeafBounds(nodeIndex);
                }
                return {nodeIndex, modified};
            }
            if (strategy == Strategy::Expand)
                expandBounds(nodeIndex, location);
            std::size_t nextNode = (location[m_nodes[nodeIndex].m_splitDimension] < m_nodes[nodeIndex].m_splitValue)
                ? m_nodes[nodeIndex].m_children.first
                : m_nodes[nodeIndex].m_children.second;
            auto res = findLeaf(nextNode, location, strategy, op);
            if (res.second && strategy == Strategy::Contract)
            {
                m_nodes[nodeIndex].m_entries--;
                updateInternalNodeBounds(nodeIndex);
            }
            return res;
        }

        void expandBounds(std::size_t nodeIndex, const point_t& location)
        {
            bounds_t& bounds = m_nodeBounds[nodeIndex];
            Loop<Dimensions>::run([&](std::size_t i) { bounds[i].expand(location[i]); });
            m_nodes[nodeIndex].m_entries++;
        }

        Scalar pointRectDist(std::size_t nodeIndex, const point_t& location) const
        {
            auto clamp = [](Scalar v, Range r) { return std::max(r.min, std::min(r.max, v)); };
            const bounds_t& bounds = m_nodeBounds[nodeIndex];
            point_t closestBoundsPoint;
            Loop<Dimensions>::run([&](std::size_t i) { closestBoundsPoint[i] = clamp(location[i], bounds[i]); });
            return Distance::distance(closestBoundsPoint, location);
        }

        void recalculateLeafBounds(std::size_t nodeIndex)
        {
            bounds_t& bounds = m_nodeBounds[nodeIndex];
            bounds.fill({std::numeric_limits<Scalar>::max(), std::numeric_limits<Scalar>::lowest()});
            const Node& node = m_nodes[nodeIndex];
            for (std::size_t i = node.m_dataBegin; i < node.m_dataEnd; ++i)
            {
                const auto& loc = m_data[i].location;
                Loop<Dimensions>::run([&](std::size_t d) { bounds[d].expand(loc[d]); });
            }
        }

        void updateInternalNodeBounds(std::size_t nodeIndex)
        {
            const Node& node = m_nodes[nodeIndex];
            const bounds_t& leftBounds = m_nodeBounds[node.m_children.first];
            const bounds_t& rightBounds = m_nodeBounds[node.m_children.second];
            bounds_t& bounds = m_nodeBounds[nodeIndex];
            Loop<Dimensions>::run(
                [&](std::size_t d)
                {
                    bounds[d] = leftBounds[d];
                    bounds[d].expand(rightBounds[d]);
                });
        }

        void searchCapacityLimitedBall(const point_t& location,
                                       Scalar maxRadius,
                                       std::size_t maxPoints,
                                       std::vector<std::size_t>& searchStack,
                                       std::priority_queue<DistancePayload, std::vector<DistancePayload>>& prioqueue,
                                       std::vector<DistancePayload>& results) const
        {
            std::size_t numSearchPoints = std::min(maxPoints, size());
            if (numSearchPoints > 0)
            {
                searchStack.push_back(0);
                while (!searchStack.empty())
                {
                    std::size_t nodeIndex = searchStack.back();
                    searchStack.pop_back();
                    const Node& node = m_nodes[nodeIndex];
                    Scalar minDist = pointRectDist(nodeIndex, location);
                    if (maxRadius > minDist
                        && (prioqueue.size() < numSearchPoints || prioqueue.top().distance > minDist))
                    {
                        if (node.m_splitDimension == Dimensions)
                            node.searchCapacityLimitedBall(m_data, location, maxRadius, numSearchPoints, prioqueue);
                        else
                            node.queueChildren(location, searchStack);
                    }
                }
                results.reserve(prioqueue.size());
                while (!prioqueue.empty())
                {
                    results.push_back(prioqueue.top());
                    prioqueue.pop();
                }
                std::reverse(results.begin(), results.end());
            }
        }

        bool split(std::size_t index)
        {
            if (m_nodes.capacity() < m_nodes.size() + 2)
            {
                m_nodes.reserve((m_nodes.capacity() + 1) * 2);
                m_nodeBounds.reserve((m_nodeBounds.capacity() + 1) * 2);
            }
            Node& splitNode = m_nodes[index];
            const bounds_t& splitBounds = m_nodeBounds[index];
            splitNode.m_splitDimension = Dimensions;
            Scalar width(0);
            Loop<Dimensions>::run(
                [&](std::size_t i)
                {
                    Scalar dWidth = splitBounds[i].max - splitBounds[i].min;
                    if (dWidth > width)
                    {
                        splitNode.m_splitDimension = i;
                        width = dWidth;
                    }
                });
            if (splitNode.m_splitDimension == Dimensions)
                return false;
            std::vector<Scalar> splitDimVals;
            splitDimVals.reserve(splitNode.m_entries);
            for (std::size_t i = splitNode.m_dataBegin; i < splitNode.m_dataEnd; ++i)
                splitDimVals.push_back(m_data[i].location[splitNode.m_splitDimension]);
            std::nth_element(
                splitDimVals.begin(), splitDimVals.begin() + splitDimVals.size() / 2 + 1, splitDimVals.end());
            std::nth_element(splitDimVals.begin(),
                             splitDimVals.begin() + splitDimVals.size() / 2,
                             splitDimVals.begin() + splitDimVals.size() / 2 + 1);
            splitNode.m_splitValue
                = (splitDimVals[splitDimVals.size() / 2] + splitDimVals[splitDimVals.size() / 2 + 1]) / Scalar(2);
            auto it = std::partition(m_data.begin() + splitNode.m_dataBegin,
                                     m_data.begin() + splitNode.m_dataEnd,
                                     [&](const LocationPayload& lp)
                                     { return lp.location[splitNode.m_splitDimension] < splitNode.m_splitValue; });
            std::size_t mid = std::distance(m_data.begin(), it);
            splitNode.m_children = std::make_pair(m_nodes.size(), m_nodes.size() + 1);
            m_nodes.emplace_back(splitNode.m_dataBegin, mid);
            m_nodeBounds.emplace_back();
            m_nodeBounds.back().fill({std::numeric_limits<Scalar>::max(), std::numeric_limits<Scalar>::lowest()});
            m_nodes.emplace_back(mid, splitNode.m_dataEnd);
            m_nodeBounds.emplace_back();
            m_nodeBounds.back().fill({std::numeric_limits<Scalar>::max(), std::numeric_limits<Scalar>::lowest()});
            std::size_t leftIndex = splitNode.m_children.first, rightIndex = splitNode.m_children.second;
            for (std::size_t i = m_nodes[leftIndex].m_dataBegin; i < m_nodes[leftIndex].m_dataEnd; ++i)
                expandBounds(leftIndex, m_data[i].location);
            for (std::size_t i = m_nodes[rightIndex].m_dataBegin; i < m_nodes[rightIndex].m_dataEnd; ++i)
                expandBounds(rightIndex, m_data[i].location);
            if (m_nodes[leftIndex].m_entries == 0)
            {
                splitNode.m_splitValue = 0;
                splitNode.m_splitDimension = Dimensions;
                splitNode.m_children = std::pair<std::size_t, std::size_t>(0, 0);
                m_nodes.pop_back();
                m_nodes.pop_back();
                m_nodeBounds.pop_back();
                m_nodeBounds.pop_back();
                return false;
            }
            return true;
        }
    };

    template <std::size_t Dim> struct Loop
    {
        template <typename F> static void run(F f)
        {
            Loop<Dim - 1>::run(f);
            f(Dim - 1);
        }
    };
    template <> struct Loop<0>
    {
        template <typename F> static void run(F) { }
    };

    struct L1
    {
        template <std::size_t Dimensions, typename Scalar>
        static Scalar distance(const std::array<Scalar, Dimensions>& location1,
                               const std::array<Scalar, Dimensions>& location2)
        {
            auto abs = [](Scalar v) { return v >= 0 ? v : -v; };
            Scalar dist = 0;
            Loop<Dimensions>::run([&](std::size_t i) { dist += abs(location1[i] - location2[i]); });
            return dist;
        }
    };

    struct SquaredL2
    {
        template <std::size_t Dimensions, typename Scalar>
        static Scalar distance(const std::array<Scalar, Dimensions>& location1,
                               const std::array<Scalar, Dimensions>& location2)
        {
            auto sqr = [](Scalar v) { return v * v; };
            Scalar dist = 0;
            Loop<Dimensions>::run([&](std::size_t i) { dist += sqr(location1[i] - location2[i]); });
            return dist;
        }
    };
}
}
