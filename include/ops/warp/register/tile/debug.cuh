/**
 * @file
 * @brief Debug operations: between tiles, and those which apply vectors to tiles.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

    template<typename T>
    __device__ inline void print_packed(const T& val) {
        printf("%f ", static_cast<float>(val));
    }

    template<>
    __device__ inline void print_packed(const float2& val) {
        printf("%f %f ", val.x, val.y);
    }

    template<>
    __device__ inline void print_packed(const half2& val) {
        printf("%f %f ", __half2float(val.x), __half2float(val.y));
    }

    template<>
    __device__ inline void print_packed(const bf16_2& val) {
        printf("%f %f ", __bfloat162float(val.x), __bfloat162float(val.y));
    }

    #ifdef KITTENS_HOPPER
    template<>
    __device__ inline void print_packed(const fp8e4m3_4& val) {
        float tmp[4];
        val.to_float(tmp);
        printf("%f %f %f %f ", tmp[0], tmp[1], tmp[2], tmp[3]);
    }

    template<>
    __device__ inline void print_packed(const fp8e5m2_4& val) {
        float tmp[4];
        val.to_float(tmp);
        printf("%f %f %f %f ", tmp[0], tmp[1], tmp[2], tmp[3]);
    }
    #endif

    template<ducks::rt::all T>
    __device__ static inline void print(const T &tile) {
        // Header
        if (threadIdx.x == 0) {
            printf("Tile %dx%d:\n", tile.rows, tile.cols);
        }
        __syncwarp();
    
        // Iterate through tiles
        for(int i = 0; i < tile.height; i++) {
            for(int j = 0; j < tile.width; j++) {
                // Let each thread print in order
                for(int thread = 0; thread < 32; thread++) {
                    if (threadIdx.x == thread) {
                    printf("\nThread %d: [", threadIdx.x);
                    for(int k = 0; k < tile.packed_per_tile; k++) {
                        print_packed(tile.tiles[i][j].data[k]);
                    }
                    printf("]");
                    }
                    __syncwarp();
                }
                if (threadIdx.x == 0) {
                    printf("\n|\n");
                }
                __syncwarp();
            }
            if (threadIdx.x == 0) {
                printf("---\n");
            }
            __syncwarp();
        }
    }

}