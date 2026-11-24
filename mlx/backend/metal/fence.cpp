// Copyright © 2024 Apple Inc.
#include "mlx/fence.h"

#include <optional>

#include "mlx/backend/metal/device.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

// The fence kernels, compiled at run time. fence.metal needs Metal 3.2
// (`coherent(system)`, `thread_scope_system`), and builds that compile the
// kernel library without it (SwiftPM, which has no per-file Metal language
// setting) leave these kernels out of default.metallib. Building them here
// with the device's newest language version makes MLX_METAL_FAST_SYNCH work
// in every build. When default.metallib has them too, this copy is unused
// duplication, not a conflict.
constexpr const char* fence_kernels_source = R"FENCE(
// Copyright © 2024 Apple Inc.

#pragma METAL internals : enable

#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
#endif
namespace metal {
constexpr constant metal::thread_scope thread_scope_system =
    static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__);
}

#include <metal_atomic>

[[kernel]] void input_coherent(
    volatile coherent(system) device uint* input [[buffer(0)]],
    const constant uint& size [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  if (index < size) {
    input[index] = input[index];
  }
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device,
      metal::memory_order_seq_cst,
      metal::thread_scope_system);
}

// single thread kernel to update timestamp
[[kernel]] void fence_update(
    volatile coherent(system) device uint* timestamp [[buffer(0)]],
    constant uint& value [[buffer(1)]]) {
  timestamp[0] = value;
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device,
      metal::memory_order_seq_cst,
      metal::thread_scope_system);
}

// single thread kernel to spin wait for timestamp value
[[kernel]] void fence_wait(
    volatile coherent(system) device uint* timestamp [[buffer(0)]],
    constant uint& value [[buffer(1)]]) {
  while (1) {
    metal::atomic_thread_fence(
        metal::mem_flags::mem_device,
        metal::memory_order_seq_cst,
        metal::thread_scope_system);
    if (timestamp[0] >= value) {
      break;
    }
  }
}
)FENCE";

MTL::ComputePipelineState* get_fence_kernel(
    metal::Device& d,
    const std::string& name) {
  auto lib = d.get_library(
      "mlx_fence_kernels", [] { return std::string(fence_kernels_source); });
  return d.get_kernel(name, lib);
}

} // namespace

struct FenceImpl {
  FenceImpl(Stream stream) {
    auto d = metal::device(stream.device).mtl_device();
    if (!d->supportsFamily(MTL::GPUFamilyMetal3)) {
      use_fast = false;
    } else if (__builtin_available(macOS 15, iOS 18, *)) {
      use_fast = env::metal_fast_synch();
    }

    if (!use_fast) {
      event = std::make_unique<Event>(stream);
    } else {
      fence.emplace(allocator::malloc(sizeof(uint32_t)));
      cpu_value()[0] = 0;
    }
  }

  bool use_fast{false};
  uint32_t count{0};
  std::optional<allocator::Data> fence;
  std::unique_ptr<Event> event;

  MTL::Buffer* buffer() {
    return static_cast<MTL::Buffer*>(fence->buffer().ptr());
  }

  std::atomic_uint* cpu_value() {
    return static_cast<std::atomic_uint*>(buffer()->contents());
  }
};

Fence::Fence(Stream stream) {
  fence_ = std::make_shared<FenceImpl>(stream);
}

void Fence::wait(Stream stream, const array& x, uint32_t value) {
  auto& f = *static_cast<FenceImpl*>(fence_.get());

  if (!f.use_fast) {
    auto& event = *f.event;
    event.set_value(value);
    event.wait(stream);
    return;
  }

  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [fence_ = fence_, value]() mutable {
      auto& f = *static_cast<FenceImpl*>(fence_.get());
      while (f.cpu_value()[0] < value) {
      }
    });
    return;
  }

  auto& d = metal::device(stream.device);
  auto& compute_encoder = metal::get_command_encoder(stream);

  compute_encoder.end_encoding();

  // Register outputs to ensure that no kernels which depends on the
  // output starts before this one is done
  compute_encoder.register_output_array(x);

  auto kernel = get_fence_kernel(d, "fence_wait");
  MTL::Size kernel_dims = MTL::Size(1, 1, 1);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_buffer(f.buffer(), 0);
  compute_encoder.set_bytes(value, 1);
  compute_encoder.dispatch_threads(kernel_dims, kernel_dims);
  compute_encoder.end_encoding(true);

  compute_encoder.get_command_buffer()->addCompletedHandler(
      [fence_ = fence_](MTL::CommandBuffer* cbuf) {});
}

uint32_t Fence::update(Stream stream, const array& x, bool cross_device) {
  auto& f = *static_cast<FenceImpl*>(fence_.get());
  f.count++;

  if (!f.use_fast) {
    f.event->set_value(f.count);
    f.event->signal(stream);
    return f.count;
  }

  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [fence_ = fence_, count = f.count]() mutable {
      auto& f = *static_cast<FenceImpl*>(fence_.get());
      f.cpu_value()[0] = count;
    });
    return f.count;
  }

  auto& d = metal::device(stream.device);
  auto& compute_encoder = metal::get_command_encoder(stream);

  // Launch input visibility kernels
  if (cross_device) {
    auto kernel = get_fence_kernel(d, "input_coherent");
    uint32_t nthreads = (x.data_size() * x.itemsize() + sizeof(uint32_t) - 1) /
        sizeof(uint32_t);
    MTL::Size group_dims = MTL::Size(1024, 1, 1);
    MTL::Size grid_dims = MTL::Size((nthreads + 1024 - 1) / 1024, 1, 1);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x, 0);
    compute_encoder.set_bytes(nthreads, 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  // Barrier on previous kernels
  compute_encoder.barrier();

  // Launch value update kernel
  auto kernel = get_fence_kernel(d, "fence_update");
  MTL::Size kernel_dims = MTL::Size(1, 1, 1);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_buffer(f.buffer(), 0);
  compute_encoder.set_bytes(f.count, 1);
  compute_encoder.dispatch_threads(kernel_dims, kernel_dims);

  compute_encoder.get_command_buffer()->addCompletedHandler(
      [fence_ = fence_](MTL::CommandBuffer* cbuf) {});
  return f.count;
}

} // namespace mlx::core
