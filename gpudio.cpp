#include <cassert>

// https://betterprogramming.pub/optimizing-parallel-reduction-in-metal-for-apple-m1-8e8677b49b01

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include "Metal.hpp"

#define N (48000)

int main() {
  using NS::StringEncoding::UTF8StringEncoding;

  auto* device = MTL::CreateSystemDefaultDevice();
  auto queue = device->newCommandQueue();
  const char* code = R"(
#include <metal_stdlib>
using namespace metal;
kernel void add_arrays(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] + inB[index];
    //result[index] = 3.0;
}
)";
  NS::Error* error = nullptr;
  auto* library =
    device->newLibrary(NS::String::string(code, UTF8StringEncoding), nullptr, &error);
  if (!library) {
    __builtin_printf("%s", error->localizedDescription()->utf8String());
    assert(false);
  }
  auto* function =
    library->newFunction(NS::String::string("add_arrays", NS::UTF8StringEncoding));
  if (!function) {
    printf("FAIL\n");
    assert(false);
  }

  auto* pipeline = device->newComputePipelineState(function, &error);
  if (!pipeline) {
    __builtin_printf("%s", error->localizedDescription()->utf8String());
    assert(false);
  }
  function->release();
  library->release();

  auto* buffer_a = device->newBuffer(N * 4, MTL::ResourceStorageModeManaged);
  auto* buffer_b = device->newBuffer(N * 4, MTL::ResourceStorageModeManaged);
  auto* buffer_result = device->newBuffer(N * 4, MTL::ResourceStorageModeManaged);
  {
    float a[N], b[N], result[N];
    for (int i = 0; i < N; i++) {
      a[i] = 1.0f;
      b[i] = 1.0f;
      result[i] = 1.0f;
    }
    memcpy(buffer_a->contents(), a, N * 4);
    memcpy(buffer_b->contents(), b, N * 4);
    memcpy(buffer_result->contents(), result, N * 4);
  }

  auto* command = queue->commandBuffer();
  if (!command) {
    __builtin_printf("Could not make command\n");
    assert(false);
  }
  auto* encoder = command->computeCommandEncoder();
  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(buffer_a, 0, 0);
  encoder->setBuffer(buffer_b, 0, 1);
  encoder->setBuffer(buffer_result, 0, 2);
  auto grid_size = MTL::Size(N, 1, 1);
  auto maximum = pipeline->maxTotalThreadsPerThreadgroup();
  if (maximum > N) {
    maximum = N;
  }
  //printf("%lu\n", maximum);
  MTL::Size thread_group_size(maximum, 1, 1);
  encoder->dispatchThreads(grid_size, thread_group_size);
  encoder->endEncoding();
  command->commit();
  command->waitUntilCompleted(); // block; important

  auto* a = static_cast<float*>(buffer_a->contents());
  auto* b = static_cast<float*>(buffer_b->contents());
  auto* result = static_cast<float*>(buffer_result->contents());

  for (unsigned long index = 0; index < N; index++)
  {
      if (result[index] != (a[index] + b[index]))
      {
          printf("Compute ERROR: index=%lu result=%g vs %g=a+b (a=%g, b=%g)\n",
                 index, result[index], a[index] + b[index], a[index], b[index]);
          assert(result[index] == (a[index] + b[index]));
      }
  }
  printf("Compute results as expected\n");
}
