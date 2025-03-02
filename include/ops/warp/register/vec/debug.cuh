
/**
 * @file
 * @brief Debug operations: between vectors.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {
/**
* @brief Prints the contents of a register vector to the console for debugging purposes.
*
* This function prints a formatted representation of a register vector, showing its length
* and the data contained in each thread. The output is synchronized across threads
* to ensure proper ordering of printed values.
*
* @tparam T Register vector type.
* @param vector[in] The register vector to print.
*/
template<ducks::rv::all T>
__device__ static inline void print(const T &vector) {
    if (threadIdx.x == 0) {
        printf("Vector [%d]:\n", vector.length);
    }
    __syncwarp();
    for(int i = 0; i < tile.outer_dim; i++) {
        for(int thread = 0; thread < 32; thread++) {
            if (threadIdx.x == thread) {
            printf("\nThread %d: [", threadIdx.x);
            for(int j = 0; j < tile.inner_dim; j++) {
                base_ops::print(vector[i][j]);
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