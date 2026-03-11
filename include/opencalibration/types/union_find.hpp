#pragma once

#include <ankerl/unordered_dense.h>

#include <cstddef>
#include <cstdint>

namespace opencalibration
{

class UnionFind
{
  public:
    static uint64_t key(size_t a, size_t b)
    {
        return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
    }

    uint64_t find(uint64_t x)
    {
        auto it = _parent.find(x);
        if (it == _parent.end())
        {
            _parent[x] = x;
            _rank[x] = 0;
            return x;
        }
        if (it->second != x)
            it->second = find(it->second);
        return it->second;
    }

    void unite(uint64_t a, uint64_t b)
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

  private:
    ankerl::unordered_dense::map<uint64_t, uint64_t> _parent;
    ankerl::unordered_dense::map<uint64_t, uint64_t> _rank;
};

} // namespace opencalibration
