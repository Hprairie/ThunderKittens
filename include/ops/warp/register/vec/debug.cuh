
/**
 * @file
 * @brief Debug operations: between vectors.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {
    template<ducks::rt::all T>
    __device__ static inline void print(const T &tile) {
        // Header
        if (threadIdx.x == 0) {
            printf("Tile %dx%d:\n", tile.rows, tile.cols);
        }
        __syncwarp();
    
        // Iterate through tiles
        for(int i = 0; i < tile.outer_dim; i++) {
            for(int j = 0; j < tile.inner_dim; j++) {
                // Let each thread print in order
                for(int thread = 0; thread < 32; thread++) {
                    if (threadIdx.x == thread) {
                    printf("\nThread %d: [", threadIdx.x);
                    for(int k = 0; k < tile.packed_per_tile; k++) {
                        base_ops::print(tile.tiles[i][j].data[k]);
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