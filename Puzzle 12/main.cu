#include "..\base.h"

#include <stdio.h>

#define THREADS_PER_BLOCK 256

static f32 InitialArray[1024 * 32] = {0};
static f32 Result = 0.0f;

static void Init() {
	random_state RandomState = { 0xB40148552A2E3491ULL };
	for (u32 i = 0; i < ARRAY_LEN(InitialArray); ++i) {
		InitialArray[i] = RandomFloat(&RandomState);
	}
}

__device__ f32 ReduceAdd(f32 Value) {
	u32 Mask = 0xFFFFFFFF;
	Value += __shfl_down_sync(Mask, Value, 16);
	Value += __shfl_down_sync(Mask, Value, 8);
	Value += __shfl_down_sync(Mask, Value, 4);
	Value += __shfl_down_sync(Mask, Value, 2);
	Value += __shfl_down_sync(Mask, Value, 1);
	return Value;
}

__global__ void PrefixSum(f32 *A, f32 *Out) {

	__shared__ f32 SharedSums[32];

	u32 GlobalIndex = blockIdx.x * blockDim.x + threadIdx.x;
	f32 ReducedSum = ReduceAdd(A[GlobalIndex]);

	u32 WarpIndex = threadIdx.x / 32;
	u32 WarpCount = blockDim.x / 32;
	if (threadIdx.x % 32 == 0 && WarpIndex < WarpCount) {
		SharedSums[WarpIndex] = ReducedSum;
	}
	__syncthreads();

	if (WarpIndex == 0) {
		f32 Result = 0.0f;
		if (threadIdx.x < WarpCount) {
			Result = SharedSums[threadIdx.x];
		}
		Result = ReduceAdd(Result);
		if (threadIdx.x == 0) {
			Out[blockIdx.x] = Result;
		}
	}
}

s32 main() {
	Init();

	f32 *GPUArrayA = 0, *GPUArrayB = 0, *GPUArrayOut = 0;
	cudaMalloc(&GPUArrayA, sizeof(InitialArray));
	cudaMalloc(&GPUArrayB, ARRAY_LEN(InitialArray) / THREADS_PER_BLOCK);
	cudaMemset(GPUArrayB, 0, ARRAY_LEN(InitialArray) / THREADS_PER_BLOCK);
	cudaMalloc(&GPUArrayOut, sizeof(f32));
	cudaMemcpy(GPUArrayA, InitialArray, sizeof(InitialArray), cudaMemcpyHostToDevice);

	u32 BlockCount = ARRAY_LEN(InitialArray) / THREADS_PER_BLOCK;
	PrefixSum<<<BlockCount, THREADS_PER_BLOCK>>>(GPUArrayA, GPUArrayB);
	PrefixSum<<<1, BlockCount>>>(GPUArrayB, GPUArrayOut);
	cudaMemcpy(&Result, GPUArrayOut, sizeof(Result), cudaMemcpyDeviceToHost);

	f32 ExpectedResult = 0.0f;
	for (u32 i = 0; i < ARRAY_LEN(InitialArray); ++i) {
		ExpectedResult += InitialArray[i];
	}

	if (Result == ExpectedResult) {
		printf(ANSI_COLOR_GREEN "Expected: %.5f | Actual: %.5f\n", ExpectedResult, Result);
	} else {
		printf(ANSI_COLOR_RED "Expected: %.5f | Actual: %.5f\n", ExpectedResult, Result);
	}

	printf(ANSI_COLOR_RESET);
}
