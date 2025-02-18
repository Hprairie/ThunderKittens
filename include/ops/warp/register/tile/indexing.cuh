/**
 * @file
 * @brief Indexing operations: between tiles, and those which apply vectors to tiles.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {
    
/**
 * @brief Applies a where operation to each element of a tile.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param condition[in] Condition tile to indicate which source tile to use.
 * @param src1[in] Source tile to apply the operation, values selected at indices where condition is True.
 * @param src2[in] Source tile to apply the operation, values selected at indices where condition is False.
 */
template<ducks::rt::all T>
__device__ static inline void where(T &dst, const T &condition, const T &src1, const T &src2) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k].x = condition.tiles[i][j].data[k].x ? src1.tiles[i][j].data[k].x : src2.tiles[i][j].data[k].x;
                dst.tiles[i][j].data[k].y = condition.tiles[i][j].data[k].y ? src1.tiles[i][j].data[k].y : src2.tiles[i][j].data[k].y;
            }
        }
    }
}

/**
 * @brief Joins two tiles into one, along the height dimension. The order of the tiles is src1, src2.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src1[in] Source tile 1 to be joined with.
 * @param src2[in] Source tile 2 to be joined with.
 */
template<ducks::rt:all V, ducks::rt::all T>
__device__ static inline void join_h(T &dst, const T &src1, const T &src2) {
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::row_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(dst.height == src1.height + src2.height);
    static_assert(dst.width == src1.width == src2.width);

    #pragma unroll
    for(int i = 0; i < src1.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            dst.tiles[i][j] = src1.tiles[i][j];
        }
    }
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            dst.tiles[i + src1.height][j] = src2.tiles[i][j];
        }
    }
}

/**
 * @brief Joins two tiles into one, along the height dimension. The order of the tiles is src1, src2.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src1[in] Source tile 1 to be joined with.
 * @param src2[in] Source tile 2 to be joined with.
 */
template<ducks::rt:all V, ducks::rt::all T>
__device__ static inline void join_h(T &dst, const T &src1, const T &src2) {
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::row_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(dst.height == src1.height + src2.height);
    static_assert(dst.width == src1.width == src2.width);

    #pragma unroll
    for(int i = 0; i < src1.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            dst.tiles[i][j] = src1.tiles[i][j];
        }
    }
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            dst.tiles[i + src1.height][j] = src2.tiles[i][j];
        }
    }
}


}