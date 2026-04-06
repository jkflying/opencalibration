#pragma once

#include <cstddef>
#include <numeric>
#include <vector>

namespace opencalibration
{

class UnionFind
{
  public:
    explicit UnionFind(size_t n) : _parent(n), _rank(n, 0)
    {
        std::iota(_parent.begin(), _parent.end(), 0);
    }

    size_t find(size_t x)
    {
        if (_parent[x] != x)
            _parent[x] = find(_parent[x]);
        return _parent[x];
    }

    void unite(size_t a, size_t b)
    {
        a = find(a);
        b = find(b);
        if (a == b)
            return;
        if (_rank[a] < _rank[b])
            std::swap(a, b);
        _parent[b] = a;
        if (_rank[a] == _rank[b])
            _rank[a]++;
    }

    [[nodiscard]] bool is_singleton(size_t x) const
    {
        return _parent[x] == x && _rank[x] == 0;
    }

  private:
    std::vector<size_t> _parent;
    std::vector<size_t> _rank;
};

} // namespace opencalibration
