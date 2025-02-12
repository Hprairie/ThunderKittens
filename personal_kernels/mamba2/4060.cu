#include "kittens.cuh"
#include "ops/group/group.cuh"
#include "prototype.cuh"


using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcsf;

template <int S_BLOCK_L, int S_BLOCK_D, int R_BLOCK_L, int R_BLOCK_D>
struct mamba2_layout {
    // Define shared tile layouts
    using base_tile = st_bf<S_BLOCK_L, S_BLOCK_D>;
    using base_vec = sv_fl<S_BLOCK_L>;

    // Define register tile layouts
    using consumer_tile_bf = rt_bf<R_BLOCK_L, R_BLOCK_D>;
    using consumer_tile_fl = rt_fl<R_BLOCK_L, R_BLOCK_D>;

    // Define global layouts
    using global_tile_layout = gl<bf16, 1, 1, -1, -1, base_tile>;
    using global_vec_layout = gl<float, 1, 1, -1, -1, base_vec>;

    // Define global variables 
    struct globals {global_tile_layout b, c, x, o; global_vec_layout a;};

    // Define shared memory layouts
    struct input_block {
        base_tile b, c, x[2];
        base_vec a[2], padding[6];
    };
    struct ouput_block {
        base_tile o[2];
    };
    struct scratch_block {
        base_tile cb[2], b[2];
        base_vec a_cumsum[2], padding[6];
    };
    struct finish_block {};

    // Define registers
    struct common_state {int batch, head;};
    struct producer_state {};
    struct consumer_state {
        consumer_tile_fl o_reg, att_block, local_decay, cb;
        consumer_tile_bf attn_block_mma, c_reg, b_reg;
    };
};

template <int S_BLOCK_L, int S_BLOCK_D, int R_BLOCK_L, int R_BLOCK_D>
struct mamba2_fwd_template {
    // TK specific parameters (note that these are defaults)
	static constexpr int 
        FORCE_ALIGN=1024,
        DEBUG=0,
        NUM_BLOCKS=1,
        NUM_CONSUMER_WARPS=8, 
        NUM_PRODUCER_WARPS=4,
        INPUT_PIPE_STAGES=2, 
        OUTPUT_PIPE_STAGES=2, 
        PRODUCER_BARRIER_ARRIVALS=1, 
        CONSUMER_BARRIER_ARRIVALS=2;

    // Personal parameters
    static constexpr int
        NUM_HEADS_PROCESSED=NUM_CONSUMER_WARPS/4;

    using layout = mamba2_layout<S_BLOCK_L, S_BLOCK_D, R_BLOCK_L, R_BLOCK_D>;
    
    // Kernel Notes: We launch (132, 1, 1) and use persistent kernels

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        // Calculate the batch and head index for our current task
        int task_id = args.task_iter * gridDim.x + blockIdx.x;
        args.common.batch = task_id / (args.globals.x.depth / NUM_HEADS_PROCESSED); // batch = id / heads
        args.common.head = (task_id - args.common.batch * (args.globals.x.depth / NUM_HEADS_PROCESSED)) * NUM_HEADS_PROCESSED; // head = id - batch * heads (int rounding)
        args.num_iters = args.common.batch < args.globals.x.batch ? args.globals.x.rows / layout::base_tile::rows : -1;
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers();
        };

        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::warpid() == args.iter % 4){
                // TODO: Fix expected inputs for variable number of heads processed
                tma::expect(args.inputs_arrived, args.input.b, args.input.c, args.input.x[0], args.input.a[0], args.input.x[1], args.input.a[1]);
                tma::load_async(args.input.b, args.globals.b, {args.common.batch, 0, args.iter, 0}, args.inputs_arrived);
                tma::load_async(args.input.c, args.globals.c, {args.common.batch, 0, args.iter, 0}, args.inputs_arrived);
                #pragma unroll
                for (int i = 0; i < NUM_HEADS_PROCESSED; i++){
                    tma::load_async(args.input.x[i], args.globals.x, {args.common.batch, args.common.head + i, args.iter, 0}, args.inputs_arrived);
                    tma::load_async(args.input.a[i], args.globals.a, {args.common.batch, args.common.head + i, 0, args.iter}, args.inputs_arrived);
                }
            }
        };

        __device__ static void store(producer_store_args<layout> args) {

            if (warpgroup::warpid() == args.iter % 4){

                #pragma unroll
                for (int i = 0; i < NUM_HEADS_PROCESSED; i++){
                    tma::store_async(args.globals.o, args.output.o[i], {args.common.batch, args.common.head + i, args.iter, 0});
                }
                tma::store_async_read_wait();
                __syncwarp();
                if (warpgroup::laneid() == 0) arrive(args.outputs_finished);
                __syncwarp();
            }

        };
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<NUM_CONSUMER_WARPS/WARPGROUP_WARPS>();
        };

        __device__ static void compute(consumer_compute_args<layout> args) {
            int warpgroupid = warpgroup::groupid();

            warpgroup::sync(warpgroupid);
            warpgroup::copy(args.scratch.a_cumsum[warpgroupid], args.input.a[warpgroupid])
            warpgroup::sync(warpgroupid);

            if (warpgroup::warpid() == 0){
                cumsum(args.scratch.a_cumsum[warpgroupid], args.scratch.a_cumsum[warpgroupid]);
            }
            warpgroup::sync(warpgroupid); // Wait until cumsum is done

            // Calculate the decay
            warpgroup::load(args.state.local_decay, args.input.a[warpgroupid]);
            float decay = args.scratch.a_cumsum[warpgroupid][layout::base_vec::length - 1];
            sub(args.state.local_decay, decay);
            exp(args.state.local_decay, args.state.local_decay);

            // Add attention mask

            // Calculate the attention block
            warpgroup::load(args.state.c_reg, args.input.c);
            warpgroup::mm_ABt(args.state.att_block, args.state.c_reg, args.input.b);
            warpgroup::mma_async_wait();
            mul(args.state.att_block, args.state.att_block, args.state.local_decay);
            warpgroup::mm_AB(args.state.o_reg, args.state.att_block, args.input.x[warpgroupid]);
            warpgroup::mma_async_wait();

            // Multiply by the decay
            warpgroup::mma_async_wait();

        };

        __device__ static void finish(consumer_finish_args<layout> args) {
            if (warpgroup::laneid() == 0) arrive(args.finish_finished);
            __syncwarp();
        };
    };
};