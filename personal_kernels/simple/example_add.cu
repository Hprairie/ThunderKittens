#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

#define NUM_THREADS (kittens::WARP_THREADS) // use 1 warp

#define _row 16
#define _col 16

// define global layout
using _gl  = gl<float, -1, -1, -1, -1, st_fl<_row, _col>>;

struct micro_globals {
    _gl x, o;
    // grid - number of thread blocks we are launching
    dim3 grid()  { return dim3(x.batch, x.depth); } 
    // block - number of threads in a thread block
    dim3 block() { return dim3(32); } 
    // Safe shared memory size for H100
    size_t dynamic_shared_memory() { return 224000; } 
};

// define kernel
__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const __grid_constant__ micro_globals g) {

    // shared memory
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_fl<_row, _col> (&x_s) = al.allocate<st_fl<_row, _col>>();
    st_fl<_row, _col> (&o_s) = al.allocate<st_fl<_row, _col>>();

    // register memory 
    rt_fl<_row, _col> x_reg_fl;

    // load from HBM to shared
    load(x_s, g.x, {blockIdx.x, blockIdx.y, 0, 0});
    __syncthreads();

    // load from shared to register
    load(x_reg_fl, x_s);
    __syncthreads();

    // x (dst) = x (src b) + x (src a)
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        print(x_reg_fl);
    }
    add(x_reg_fl, x_reg_fl, x_reg_fl);
    __syncthreads();

    // store from register to shared
    store(o_s, x_reg_fl);
    __syncthreads();

    // store from shared to HBM
    store(g.o, o_s, {blockIdx.x, blockIdx.y, 0, 0});
    __syncthreads();
}

// Launch Kernel
void dispatch_micro(micro_globals g) {
    unsigned long mem_size = 50480; 
    cudaFuncSetAttribute(
        micro_tk,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    cudaDeviceSynchronize();
}


PYBIND11_MODULE(simple_tk, m) {
    m.doc() = "simple_tk python module";
    // For wrapping kernels directly.
    py::bind_kernel<micro_tk>(m, "micro_tk", &micro_globals::x, &micro_globals::o); 
    // For host functions that wrap the kernel, this will be called from Python
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::x, &micro_globals::o); 
}