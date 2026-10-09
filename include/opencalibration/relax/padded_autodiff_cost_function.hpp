#pragma once

#include <ceres/autodiff_cost_function.h>
#include <ceres/sized_cost_function.h>

#include <algorithm>
#include <array>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

namespace opencalibration
{
namespace detail
{
constexpr int SIMD_PACKET_WIDTH = 4;

constexpr int paddingToWholePackets(int width)
{
    const bool too_narrow_to_vectorize = width <= SIMD_PACKET_WIDTH / 2;
    return too_narrow_to_vectorize ? 0 : (SIMD_PACKET_WIDTH - width % SIMD_PACKET_WIDTH) % SIMD_PACKET_WIDTH;
}

template <typename F, size_t NumBlocks> struct DropPadBlock
{
    std::unique_ptr<F> functor;

    template <typename... Args> bool operator()(Args... args) const
    {
        return call(std::forward_as_tuple(args...), std::make_index_sequence<NumBlocks>());
    }

  private:
    template <typename Tuple, size_t... I> bool call(const Tuple &args, std::index_sequence<I...>) const
    {
        return (*functor)(std::get<I>(args)..., std::get<std::tuple_size_v<Tuple> - 1>(args));
    }
};

template <typename F, int R, int... Ns> class JetPaddedCostFunction final : public ceres::SizedCostFunction<R, Ns...>
{
    static constexpr size_t NUM_BLOCKS = sizeof...(Ns);
    static constexpr int PAD = paddingToWholePackets((Ns + ...));

  public:
    explicit JetPaddedCostFunction(F *functor)
        : _padded(new DropPadBlock<F, NUM_BLOCKS>{std::unique_ptr<F>(functor)})
    {
    }

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
    {
        static constexpr std::array<double, PAD> padValues{};
        std::array<const double *, NUM_BLOCKS + 1> paddedParameters;
        std::copy_n(parameters, NUM_BLOCKS, paddedParameters.begin());
        paddedParameters.back() = padValues.data();
        if (jacobians == nullptr)
            return _padded.Evaluate(paddedParameters.data(), residuals, nullptr);

        std::array<double *, NUM_BLOCKS + 1> paddedJacobians;
        std::copy_n(jacobians, NUM_BLOCKS, paddedJacobians.begin());
        paddedJacobians.back() = nullptr;
        return _padded.Evaluate(paddedParameters.data(), residuals, paddedJacobians.data());
    }

  private:
    ceres::AutoDiffCostFunction<DropPadBlock<F, NUM_BLOCKS>, R, Ns..., PAD> _padded;
};
} // namespace detail

template <typename F, int R, int... Ns>
using PaddedAutoDiffCostFunction = std::conditional_t<detail::paddingToWholePackets((Ns + ...)) == 0,
                                                      ceres::AutoDiffCostFunction<F, R, Ns...>,
                                                      detail::JetPaddedCostFunction<F, R, Ns...>>;
} // namespace opencalibration
