
/**
 * @file
 * @brief Scans on vectors stored in registers.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "common/util.cuh"
#include <numeric>

namespace kittens {
    /**
    * @brief Performs an inclusive associative scan operation on a register vector within a warp.
    *
    * This function applies a specified operation as a scan elements of a register vector `src` to a single value.
    * The result is stored in `dst`. If the `reset` parameter is true, the scan includes an initial value `src_accum`.
    *
    * @tparam op The operation to perform on the elements. Must provide a static `op` method.
    * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
    * @tparam reset A boolean flag indicating whether to include an initial value in the scan.
    * @param[out] dst The register vector to store the results of the associative scan.
    * @param[in] src The register vector to perform the associative scan.
    * @param[in] src_start The initial value to include in the scan if `reset` is false.
    */
    template<typename CombineFn, ducks::rv::naive_layout RV, bool reset>
    __device__ static inline void inclusive_warp_scan(
        RV &dst,
        const RV &src,
        const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum){
        using T = typename base_types::packing<typename RV::dtype>::unpacked_type;
        const int laneid = kittens::laneid();

        T base = reset ? T{} : src_accum;
        #pragma unroll
        for(int i = 0; i < RV::outer_dim; i++) {
            T temp = src[i][0];
            #pragma unroll
            for(int offset = 1; offset < kittens::WARP_THREADS; offset *= 2) {
                T shuffle = packed_shfl_up_sync(kittens::MASK_ALL, temp, offset);
                if (laneid >= offset) {
                    temp = CombineFn::template op<T>(temp, shuffle);
                }
            }
            dst[i][0] = CombineFn::template op<T>(base, temp);
            if (i < RV::outer_dim - 1) { // Prevents us from doing an extra shuffle
                base = packed_shfl_sync(kittens::MASK_ALL, dst[i][0], kittens::WARP_THREADS-1);
            }
        }
    }

    /**
    * @brief Performs an inclusive cumulative sum operation on a register vector within a warp.
    *
    * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
    * @param[out] dst The register vector to store the results of the cumsum.
    * @param[in] src The register vector to perform the cumsum.
    */
    template<ducks::rv::all RV>
    __device__ static inline void cumsum(RV &dst, const RV &src) {
        inclusive_warp_scan<base_ops::sum, RV, true>(
            dst, 
            src, 
            typename base_types::packing<typename RV::dtype>::unpacked_type{}
        );
    }

    /**
    * @brief Performs an inclusive cumulative sum operation on a register vector within a warp with an initial value.
    *
    * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
    * @param[out] dst The register vector to store the results of the cumsum.
    * @param[in] src The register vector to perform the cumsum.
    * @param[in] init The initial value to start the cumsum.
    */
    template<ducks::rv::all RV>
    __device__ static inline void cumsum(
        RV &dst, 
        const RV &src, 
        const typename base_types::packing<typename RV::dtype>::unpacked_type &init) {
        inclusive_warp_scan<base_ops::sum, RV, false>(dst, src, init);
    }

    /**
    * @brief Performs an inclusive cumulative product operation on a register vector within a warp.
    *
    * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
    * @param[out] dst The register vector to store the results of the cumprod.
    * @param[in] src The register vector to perform the cumprod.
    */
    template<ducks::rv::all RV>
    __device__ static inline void cumprod(RV &dst, const RV &src) {
        inclusive_warp_scan<base_ops::mul, RV, true>(
            dst, 
            src, 
            typename base_types::packing<typename RV::dtype>::unpacked_type{1}
        );
    }

    /**
    * @brief Performs an inclusive cumulative product operation on a register vector within a warp with an initial value.
    *
    * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
    * @param[out] dst The register vector to store the results of the cumprod.
    * @param[in] src The register vector to perform the cumprod.
    * @param[in] init The initial value to start the cumprod.
    */
    template<ducks::rv::all RV>
    __device__ static inline void cumprod(
        RV &dst, 
        const RV &src, 
        const typename base_types::packing<typename RV::dtype>::unpacked_type &init) {
        inclusive_warp_scan<base_ops::mul, RV, false>(dst, src, init);
    }
};