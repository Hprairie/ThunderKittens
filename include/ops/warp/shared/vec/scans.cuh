
/**
 * @file
 * @brief Warp-scope scans on shared vectors.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "common/util.cuh"

namespace kittens {
    /**
    * @brief Performs an inclusive associative scan operation on a shared vector within a warp.
    *
    * This function applies a specified operation as a scan elements of a register vector `src` to a single value.
    * The result is stored in `dst`. If the `reset` parameter is true, the scan includes an initial value `src_accum`.
    *
    * @tparam op The operation to perform on the elements. Must provide a static `op` method.
    * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
    * @tparam reset A boolean flag indicating whether to include an initial value in the scan.
    * @param[out] dst The shared vector to store the results of the associative scan.
    * @param[in] src The shared vector to perform the associative scan.
    * @param[in] src_start The initial value to include in the scan if `reset` is false.
    */
    template<typename CombineFn, ducks::sv::all SV, bool reset>
    __device__ static inline void inclusive_warp_scan(
        SV &dst,
        const SV &src,
        const typename base_types::packing<typename SV::dtype>::unpacked_type &src_accum){
        using T = typename base_types::packing<typename SV::dtype>::unpacked_type;
        const int laneid = kittens::laneid();

        T base = reset ? T{} : src_accum;
        #pragma unroll
        for (int section = 0; section < SV::length; section += kittens::WARP_THREADS){
            T temp = src[section + laneid];
            __syncwarp();
            #pragma unroll
            for (int offset = 1; offset < kittens::WARP_THREADS; offset *= 2){
                T shuffle = packed_shfl_up_sync(kittens::MASK_ALL, temp, offset);
                if (laneid >= offset) {
                    temp = CombineFn::template op<T>(temp, shuffle);
                }
            }
            dst[section + laneid] = CombineFn::template op<T>(base, temp);
            if (section + kittens::WARP_THREADS < SV::length) { // Prevents us from doing an extra sync
                __syncwarp();
                base = dst[section + kittens::WARP_THREADS - 1];
            }
        }
    }

    /**
    * @brief Performs an inclusive cumulative sum operation on a shared vector within a warp.
    *
    * @tparam SV The type of the shared vector. Must satisfy the `ducks::sv::all` concept.
    * @param[out] dst The shared vector to store the results of the cumsum.
    * @param[in] src The shared vector to perform the cumsum.
    */
    template<ducks::sv::all SV>
    __device__ static inline void cumsum(SV &dst, const SV &src) {
        inclusive_warp_scan<base_ops::sum, SV, true>(
            dst, 
            src, 
            typename base_types::packing<typename SV::dtype>::unpacked_type{}
        );
    }

    /**
    * @brief Performs an inclusive cumulative sum operation on a shared vector within a warp with an initial value.
    *
    * @tparam SV The type of the shared vector. Must satisfy the `ducks::sv::all` concept.
    * @param[out] dst The shared vector to store the results of the cumsum.
    * @param[in] src The shared vector to perform the cumsum.
    * @param[in] init The initial value to start the cumsum.
    */
    template<ducks::sv::all SV>
    __device__ static inline void cumsum(
        SV &dst, 
        const SV &src, 
        const typename base_types::packing<typename SV::dtype>::unpacked_type &init) {
        inclusive_warp_scan<base_ops::sum, SV, false>(dst, src, init);
    }

    /**
    * @brief Performs an inclusive cumulative product operation on a shared vector within a warp.
    *
    * @tparam SV The type of the shared vector. Must satisfy the `ducks::sv::all` concept.
    * @param[out] dst The shared vector to store the results of the cumprod.
    * @param[in] src The shared vector to perform the cumprod.
    */
    template<ducks::sv::all SV>
    __device__ static inline void cumprod(SV &dst, const SV &src) {
        inclusive_warp_scan<base_ops::mul, SV, true>(
            dst, 
            src, 
            typename base_types::packing<typename SV::dtype>::unpacked_type{1}
        );
    }

    /**
    * @brief Performs an inclusive cumulative product operation on a shared vector within a warp with an initial value.
    *
    * @tparam SV The type of the shared vector. Must satisfy the `ducks::sv::all` concept.
    * @param[out] dst The shared vector to store the results of the cumprod.
    * @param[in] src The shared vector to perform the cumprod.
    * @param[in] init The initial value to start the cumprod.
    */
    template<ducks::sv::all SV>
    __device__ static inline void cumprod(
        SV &dst, 
        const SV &src, 
        const typename base_types::packing<typename SV::dtype>::unpacked_type &init) {
        inclusive_warp_scan<base_ops::mul, SV, false>(dst, src, init);
    }
}