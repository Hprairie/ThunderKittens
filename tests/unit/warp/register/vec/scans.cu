#include "scans.cuh"

#ifdef TEST_WARP_REGISTER_VEC_SCANS

struct vec_scan {
    template<int S, int NW, kittens::ducks::rv_layout::all L>
    using valid = std::bool_constant<NW == 1 && S<=64>; // this is warp-level
    
    static inline const std::string test_identifier = "reg_vec_scan";

    template<int S, int NW, gl_t GL, kittens::ducks::rv_layout::all L>
    __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        float sum = 0.0f;
        for(int i = 0; i < o_ref.size(); i++) {
            sum += i_ref[i];
            o_ref[i] = sum;
        }
    }

    template<int S, int NW, gl_t GL, kittens::ducks::rv_layout::all L>
    __device__ static void device_func(const GL &input, const GL &output) {
        kittens::rv_fl<16*S, L> vec_in, vec_out;
        kittens::load(vec_in, input, {});
        kittens::cumsum(vec_out, vec_in);
        kittens::store(output, vec_out, {});
    }
};

void warp::reg::vec::scans::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/vec/scans tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    // sweep_size_1d_warp<vec_scan, SIZE, kittens::ducks::rv_layout::align>::run(results);
    // sweep_size_1d_warp<vec_scan, SIZE, kittens::ducks::rv_layout::ortho>::run(results);
    sweep_size_1d_warp<vec_scan, SIZE, kittens::ducks::rv_layout::naive>::run(results);
}

#endif