#include "..\base.h"

#include <stdio.h>

#define ARRAY_SIZE 8
static u32 InitialArray[ARRAY_SIZE] = {0};
static u32 ResultArray[ARRAY_SIZE] = {0};

void Init() {
	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		InitialArray[i] = i;
	}
}

#if 0
__global__ void Pooling(u32 *A) {
	u32 Index = threadIdx.x;
	u32 Sum = A[Index];
	if (Index >= 1) Sum += A[Index - 1];
	if (Index >= 2) Sum += A[Index - 2];
	A[Index] = Sum;
}
#elif 0
__global__ void Pooling(u32 *A) {
	u32 Index = threadIdx.x;

	__shared__ u32 SharedMemory[8];
	SharedMemory[threadIdx.x] = A[Index];
	u32 Sum = SharedMemory[threadIdx.x];
	__syncthreads();
	if (threadIdx.x >= 1) Sum += SharedMemory[threadIdx.x - 1];
	if (threadIdx.x >= 2) Sum += SharedMemory[threadIdx.x - 2];

	A[Index] = Sum;
}
#else
__global__ void Pooling(u32 *A) {
	u32 Index = threadIdx.x;
	u32 SharedValue = A[Index];
	u32 Sum = SharedValue;
	Sum += __shfl_up_sync(0xFFFFFFFF, SharedValue, 1) * (Index >= 1);
	Sum += __shfl_up_sync(0xFFFFFFFF, SharedValue, 2) * (Index >= 2);
	A[Index] = Sum;
}
#endif

s32 main() {
	Init();

	u32 *GPUArray = 0;
	u32 SizeInBytes = sizeof(InitialArray);
	cudaMalloc(&GPUArray, SizeInBytes);
	cudaMemcpy(GPUArray, InitialArray, SizeInBytes, cudaMemcpyHostToDevice);

	Pooling<<<1, 8>>>(GPUArray);
	cudaMemcpy(ResultArray, GPUArray, SizeInBytes, cudaMemcpyDeviceToHost);

	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		u32 Expected = InitialArray[i];
		if (i >= 1) Expected += InitialArray[i - 1];
		if (i >= 2) Expected += InitialArray[i - 2];
		u32 Actual = ResultArray[i];
		if (Expected == Actual) {
			printf(ANSI_COLOR_GREEN "Expected: %u -- Actual: %u\n", Expected, Actual);
		} else {
			printf(ANSI_COLOR_RED "Expected: %u -- Actual: %u\n", Expected, Actual);
		}
	}
	printf(ANSI_COLOR_RESET);
}
