#pragma once

#include "util.cuh"
#include <pybind11/pybind11.h>

namespace kittens {
namespace py {

template<typename T> struct from_object {
    static T make(pybind11::object obj) {
        return obj.cast<T>();
    }
};
template<ducks::gl::all GL> struct from_object<GL> {
    static GL make(pybind11::object obj) {
        // Check if argument is a torch.Tensor
        if (pybind11::hasattr(obj, "__class__") && 
            obj.attr("__class__").attr("__name__").cast<std::string>() == "Tensor") {
        
            // Check if tensor is contiguous
            if (!obj.attr("is_contiguous")().cast<bool>()) {
                throw std::runtime_error("Tensor must be contiguous");
            }
            if (obj.attr("device").attr("type").cast<std::string>() == "cpu") {
                throw std::runtime_error("Tensor must be on CUDA device");
            }
            
            // Get shape, pad with 1s if needed
            std::array<int, 4> shape = {1, 1, 1, 1};
            auto py_shape = obj.attr("shape").cast<pybind11::tuple>();
            size_t dims = py_shape.size();
            if (dims > 4) {
                throw std::runtime_error("Expected Tensor.ndim <= 4");
            }
            for (size_t i = 0; i < dims; ++i) {
                shape[4 - dims + i] = pybind11::cast<int>(py_shape[i]);
            }
            
            // Get data pointer using data_ptr()
            uint64_t data_ptr = obj.attr("data_ptr")().cast<uint64_t>();
            
            // Create GL object using make_gl
            return make_gl<GL>(data_ptr, shape[0], shape[1], shape[2], shape[3]);
        }
        throw std::runtime_error("Expected a torch.Tensor");
    }
};

template<typename T> concept has_dynamic_shared_memory = requires(T t) { { t.dynamic_shared_memory() } -> std::convertible_to<int>; };
template<typename T> concept has_inputs = requires { T::inputs(); };

template<typename> struct trait;
template<typename MT, typename T> struct trait<MT T::*> { using member_type = MT; using type = T; };
template<typename> using object = pybind11::object;
template<auto kernel, typename TGlobal> static void bind_kernel(auto m, auto name, auto TGlobal::*... member_ptrs) {
    m.def(name, [](object<decltype(member_ptrs)>... args) {
        TGlobal __g__ {from_object<typename trait<decltype(member_ptrs)>::member_type>::make(args)...};
        if constexpr (has_dynamic_shared_memory<TGlobal>) {
            int __dynamic_shared_memory__ = (int)__g__.dynamic_shared_memory();
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, __dynamic_shared_memory__);
            kernel<<<__g__.grid(), __g__.block(), __dynamic_shared_memory__>>>(__g__);
        } else {
            kernel<<<__g__.grid(), __g__.block()>>>(__g__);
        }
    });
}
// With stream
template<auto kernel, typename TGlobal>
static void bind_kernel_stream(auto m, auto name, auto TGlobal::*... member_ptrs) {
    m.def((std::string(name) + "_stream").c_str(), [](object<decltype(member_ptrs)>... args, uintptr_t stream) {
        TGlobal __g__ {from_object<typename trait<decltype(member_ptrs)>::member_type>::make(args)...};
        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream));
        if constexpr (has_dynamic_shared_memory<TGlobal>) {
            int __dynamic_shared_memory__ = (int)__g__.dynamic_shared_memory();
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, __dynamic_shared_memory__);
            kernel<<<__g__.grid(), __g__.block(), __dynamic_shared_memory__, cuda_stream>>>(__g__);
        } else {
            kernel<<<__g__.grid(), __g__.block(), 0, cuda_stream>>>(__g__);
        }
    });
}

// With grid/block dimensions
template<auto kernel, typename TGlobal>
static void bind_kernel_grid(auto m, auto name, auto TGlobal::*... member_ptrs) {
    m.def((std::string(name) + "_grid").c_str(), [](object<decltype(member_ptrs)>... args, 
                            int grid_x, int grid_y, int grid_z,
                            int block_x, int block_y, int block_z) {
        TGlobal __g__ {from_object<typename trait<decltype(member_ptrs)>::member_type>::make(args)...};
        dim3 grid(grid_x, grid_y, grid_z);
        dim3 block(block_x, block_y, block_z);
        if constexpr (has_dynamic_shared_memory<TGlobal>) {
            int __dynamic_shared_memory__ = (int)__g__.dynamic_shared_memory();
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, __dynamic_shared_memory__);
            kernel<<<grid, block, __dynamic_shared_memory__>>>(__g__);
        } else {
            kernel<<<grid, block>>>(__g__);
        }
    });
}

// With stream and grid/block dimensions
template<auto kernel, typename TGlobal>
static void bind_kernel_stream_grid(auto m, auto name, auto TGlobal::*... member_ptrs) {
    m.def((std::string(name) + "_stream_grid").c_str(), [](object<decltype(member_ptrs)>... args,
                                    int grid_x, int grid_y, int grid_z,
                                    int block_x, int block_y, int block_z,
                                    uintptr_t stream) {
        TGlobal __g__ {from_object<typename trait<decltype(member_ptrs)>::member_type>::make(args)...};
        dim3 grid(grid_x, grid_y, grid_z);
        dim3 block(block_x, block_y, block_z);
        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream));
        if constexpr (has_dynamic_shared_memory<TGlobal>) {
            int __dynamic_shared_memory__ = (int)__g__.dynamic_shared_memory();
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, __dynamic_shared_memory__);
            kernel<<<grid, block, __dynamic_shared_memory__, cuda_stream>>>(__g__);
        } else {
            kernel<<<grid, block, 0, cuda_stream>>>(__g__);
        }
    });
}

// Convenience function to bind all variants
template<auto kernel, typename TGlobal>
static void bind_kernel_all(auto m, auto name, auto TGlobal::*... member_ptrs) {
    bind_kernel<kernel, TGlobal>(m, name, member_ptrs...);
    bind_kernel_stream<kernel, TGlobal>(m, name, member_ptrs...);
    bind_kernel_grid<kernel, TGlobal>(m, name, member_ptrs...);
    bind_kernel_stream_grid<kernel, TGlobal>(m, name, member_ptrs...);
}

// Version that uses inputs() for all variants
template<auto kernel, typename TGlobal>
static void bind_kernel_all(auto m, auto name) {
    static_assert(has_inputs<TGlobal>, "TGlobal must provide a static inputs() method");
    
    std::apply([&](auto... member_ptrs) {
        bind_kernel_all<kernel, TGlobal>(m, name, member_ptrs...);
    }, TGlobal::inputs());
}
template<auto function, typename TGlobal> static void bind_function(auto m, auto name, auto TGlobal::*... member_ptrs) {
    m.def(name, [](object<decltype(member_ptrs)>... args) {
        TGlobal __g__ {from_object<typename trait<decltype(member_ptrs)>::member_type>::make(args)...};
        function(__g__);
    });
}


template<auto kernel, typename TGlobal> static void bind_kernel(auto m, auto name) {
    static_assert(has_inputs<TGlobal>, "TGlobal must provide a static inputs() method");
    
    std::apply([&](auto... member_ptrs) {
        bind_kernel<kernel>(m, name, member_ptrs...);
    }, TGlobal::inputs());
}
template<auto function, typename TGlobal> static void bind_function(auto m, auto name) {
    static_assert(has_inputs<TGlobal>, "TGlobal must provide a static inputs() method");
    
    std::apply([&](auto... member_ptrs) {
        bind_function<function>(m, name, member_ptrs...);
    }, TGlobal::inputs());
}
} // namespace py
} // namespace kittens
