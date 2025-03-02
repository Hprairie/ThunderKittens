/**
 * @file
 * @brief Indexing operations: between vectors.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {
    /**
     * @brief Applies a where operation to each element of a vector.
     *
     * @tparam T Vector type.
     * @param dst[out] Destination vector where the result is stored.
     * @param condition[in] Condition vector to indicate which source vector to use.
     * @param src1[in] Source vector to apply the operation, values selected at indices where condition is True.
     * @param src2[in] Source vector to apply the operation, values selected at indices where condition is False.
     */
    template<ducks::rv::all T>
    __device__ inline static void where(T &dst, const T &condition, const T &src1, const T &src2) {
        #pragma unroll
        for(int i = 0; i < dst.outer_dim; i++) {
            #pragma unroll
            for(int j = 0; j < dst.inner_dim; j++) {
                if constexpr (std::is_same_v<typename RV::layout, ortho_l> || std::is_same_v<typename RV::layout, align_l>) {
                    dst[i][j].x = condition[i][j].x ? src1[i][j].x : src2[i][j].x;
                    dst[i][j].y = condition[i][j].y ? src1[i][j].y : src2[i][j].y;

                }
                else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
                    dst[i][j] = condition[i][j] ? src1[i][j] : src2[i][j];
                }
            }
        }
    }
}