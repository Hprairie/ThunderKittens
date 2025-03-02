/**
 * @file
 * @brief Debug operations: between tiles, and those which apply vectors to tiles.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {
/**
 * @brief Prints the contents of a tile to the console for debugging purposes.
 *
 * This function prints a formatted representation of a tile, showing its dimensions
 * and the data contained in each thread. The output is synchronized across threads
 * to ensure proper ordering of printed values.
 *
 * @tparam T Tile type.
 * @param tile[in] The tile to print.
 */
template<ducks::rt::all T>
__device__ static inline void print(const T &tile) {
    // Header
    if (threadIdx.x == 0) {
        printf("Tile [%dx%d]:\n", tile.rows, tile.cols);
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