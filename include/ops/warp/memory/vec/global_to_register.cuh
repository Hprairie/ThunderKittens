/**
 * @file
 * @brief Functions for transferring data directly between global memory and registers and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/**
 * @brief Load data into a register vector from a source array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the source array.
 * @param[out] dst The destination register vector to load data into.
 * @param[in] src The source array in global memory to load data from.
 */
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void load(RV &dst, const GL &src, const COORD &idx) {
    using T2 = RV::dtype;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    U *src_ptr = (U*)&src[(idx.template unit_coord<-1, 3>())];
    int laneid = ::kittens::laneid();
    
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(auto w = 0; w < (dst.outer_dim+3)/4; w++) {
            int idx = w*64 + (laneid/4)*8 + 2*(laneid%4);
            int o_dim = w*4 + (laneid/4) / 2;
            int i_dim = (laneid/4) % 2;
            // this should be a maximally coalesced load.
            if(idx < dst.outer_dim*16)
                dst[o_dim][i_dim] = base_types::convertor<T2, U2>::convert(*(U2*)&src_ptr[idx]);
        }
        // now we need to do a bunch of shuffle_sync's to make sure everyone has everything they need.
        #pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            int leader = 8*(w%4) + (laneid%4); // repeats every 64 columns
            dst[w][0] = packed_shfl_sync(MASK_ALL, dst[w][0], leader);
            dst[w][1] = packed_shfl_sync(MASK_ALL, dst[w][1], leader+4);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        // really hoping https://stackoverflow.com/questions/15029765/is-coalescing-triggered-for-accessing-memory-in-reverse-order is still true
        // otherwise there will be some pain :/
        #pragma unroll
        for(auto w = 0; w < (dst.outer_dim+1)/2; w++) {
            int idx = w*32 + (laneid%4)*8 + (laneid/4);
            int o_dim = w*2 + (laneid%4) / 2;
            // this should be a maximally coalesced load.
            if(idx < dst.outer_dim*16) {
                T tmp = base_types::convertor<T, U>::convert(src_ptr[idx]);
                if(laneid%2==0) dst[o_dim][0].x = tmp;
                else dst[o_dim][0].y = tmp;
            }
        }
        // now we need to do a bunch of shuffle_sync's to make sure everyone has everything they need.
        #pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            int leader = (laneid/4)*4 + 2*(w%2); // repeats every 64 columns
            dst[w][0].x = __shfl_sync(MASK_ALL, dst[w][0].x, leader);
            dst[w][0].y = __shfl_sync(MASK_ALL, dst[w][0].y, leader+1);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        #pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            if(w < dst.outer_dim-1 || dst.length%32 == 0 || laneid<16) {
                dst[w][0] = base_types::convertor<T, U>::convert(src_ptr[w*32 + laneid]);
            }
        }
    }
}

/**
 * @brief Store data from a register vector to a destination array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register vector to store data from.
 */
template<typename op, ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void base_store(const GL &dst, const RV &src, const COORD &idx) {
    using T2 = RV::dtype;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;
    
    U *dst_ptr = (U*)&dst[(idx.template unit_coord<-1, 3>())];
    int laneid = ::kittens::laneid();
    
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(auto w = 0; w < (src.outer_dim+3)/4; w++) {
            int idx = w*64 + (laneid/4)*8 + 2*(laneid%4);
            int o_dim = w*4 + (laneid/4) / 2;
            int i_dim = (laneid/4) % 2;
            // this should be a maximally coalesced store. I hope!
            if(idx < src.outer_dim*16)
                // *(U2*)&dst_ptr[idx] = base_types::convertor<U2, T2>::convert(src[o_dim][i_dim]);
                op::op(*(U2*)&dst_ptr[idx], base_types::convertor<U2, T2>::convert(src[o_dim][i_dim]));
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        // really hoping https://stackoverflow.com/questions/15029765/is-coalescing-triggered-for-accessing-memory-in-reverse-order is still true
        // otherwise there will be some pain :/
        #pragma unroll
        for(auto w = 0; w < (src.outer_dim+1)/2; w++) {
            int idx = w*32 + (laneid%4)*8 + (laneid/4);
            int o_dim = w*2 + (laneid%4) / 2;
            // this should be a maximally coalesced load.
            if(idx < src.outer_dim*16) {
                U tmp;
                if(laneid%2==0) tmp = base_types::convertor<U, T>::convert(src[o_dim][0].x);
                else tmp = base_types::convertor<U, T>::convert(src[o_dim][0].y);
                // dst_ptr[idx] = tmp;
                op::op(dst_ptr[idx], tmp);
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        #pragma unroll
        for(auto w = 0; w < src.outer_dim; w++) {
            if(w < src.outer_dim-1 || src.length%32 == 0 || laneid<16) {
                // dst_ptr[w*32 + laneid] = base_types::convertor<U, T>::convert(src[w][0]);
                op::op(dst_ptr[w*32 + laneid], base_types::convertor<U, T>::convert(src[w][0]));
            }
        }
    }
}


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

/**
 * @brief Store data from a register vector to a destination array in global memory.
 *
 * @param dst The destination array in global memory to store data into.
 * @param src The source register vector to store data from.
 * @param idx The coordinate index for storing.
 */
 template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
 __device__ inline static void store(const GL &dst, const RV &src, const COORD &idx) {
     base_store<base_ops::assign, RV, GL, COORD>(dst, src, idx); 
 }

/**
 * @brief Atomically add data from a register vector to a destination array in global memory.
 *
 * @param dst The destination array in global memory to atomically add data into.
 * @param src The source register vector containing values to add.
 * @param idx The coordinate index for storing.
 */
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void atomic_add(const GL &dst, const RV &src, const COORD &idx) {
    base_store<base_ops::atomic_add, RV, GL, COORD>(dst, src, idx);
}

/**
 * @brief Atomically take maximum of data from a register vector and a destination array in global memory.
 *
 * @param dst The destination array in global memory to atomically update with maximum values.
 * @param src The source register vector containing values for maximum comparison.
 * @param idx The coordinate index for storing.
 */
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void atomic_max(const GL &dst, const RV &src, const COORD &idx) {
    base_store<base_ops::atomic_max, RV, GL, COORD>(dst, src, idx);
}

/**
 * @brief Atomically take minimum of data from a register vector and a destination array in global memory.
 *
 * @param dst The destination array in global memory to atomically update with minimum values.
 * @param src The source register vector containing values for minimum comparison.
 * @param idx The coordinate index for storing.
 */
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void atomic_min(const GL &dst, const RV &src, const COORD &idx) {
    base_store<base_ops::atomic_min, RV, GL, COORD>(dst, src, idx);
}

} // namespace kittens