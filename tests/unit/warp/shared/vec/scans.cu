#include "scans.cuh"

#ifdef TEST_WARP_SHARED_VEC_SCANS

template<typename T>
struct vec_shared_scan {
    using dtype = T;
    template<int S, int NW>
    using valid = std::bool_constant<NW == 1 && S<=64 && sizeof(dtype) != 1>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_vec_scan_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_vec_scan_gmem=half" :
                                                                                         "shared_vec_scan_gmem=float";
    template<int S, int NW, gl_t GL>
    __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        float sum = 0.0f;
        for(int i = 0; i < o_ref.size(); i++) {
            sum += i_ref[i];
            o_ref[i] = sum;
        }
    }

    template<int S, int NW, gl_t GL>
    __device__ static void device_func(const GL &input, const GL &output) {
        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator al((int*)&__shm[0]); 
        kittens::col_vec<kittens::st<dtype, 16*S, 16*S>> &vec_in = al.allocate<kittens::col_vec<kittens::st<dtype, 16*S, 16*S>>>();
        kittens::col_vec<kittens::st<dtype, 16*S, 16*S>> &vec_out = al.allocate<kittens::col_vec<kittens::st<dtype, 16*S, 16*S>>>();
        
        kittens::load(vec_in, input, {});
        kittens::cumsum(vec_out, vec_in);
        kittens::store(output, vec_out, {});
    }
};

void warp::shared::vec::scans::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/shared/vec/scans tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    sweep_gmem_type_1d_warp<vec_shared_scan, SIZE>::run(results);
}

#endif