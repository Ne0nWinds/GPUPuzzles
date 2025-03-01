#include "..\base.h"

#include <stdio.h>

__device__ f32 ReduceAdd(f32 Value) {
	u32 Mask = 0xFFFFFFFF;
	Value += __shfl_down_sync(Mask, Value, 16);
	Value += __shfl_down_sync(Mask, Value, 8);
	Value += __shfl_down_sync(Mask, Value, 4);
	Value += __shfl_down_sync(Mask, Value, 2);
	Value += __shfl_down_sync(Mask, Value, 1);
	return Value;
}

#if 0
#define ARRAY_SIZE 32
// Thread-level dot product
__global__ void Dot(f32 *A, f32 *B, f32 *Out) {

	f32 ElementwiseProduct = A[threadIdx.x] * B[threadIdx.x];
	f32 ReducedSum = ReduceAdd(ElementwiseProduct);

	if (threadIdx.x == 0) {
		*Out = ReducedSum;
	}
}
#else
#define ARRAY_SIZE 2048
// Global dot product
__global__ void Dot(f32 *A, f32 *B, f32 *Out) {

	__shared__ f32 WarpLevelResults[32];

	u32 GlobalIndex = blockIdx.x * blockDim.x + threadIdx.x;
	f32 ElementwiseProduct = A[GlobalIndex] * B[GlobalIndex];
	f32 ReducedSum = ReduceAdd(ElementwiseProduct);

	u32 WarpIndex = threadIdx.x / 32;
	u32 WarpCount = blockDim.x / 32;
	if (threadIdx.x % 32 == 0 && WarpIndex < WarpCount) {
		WarpLevelResults[WarpIndex] = ReducedSum;
	}
	__syncthreads();

	if (WarpIndex == 0) {
		f32 Result = 0.0f;
		if (threadIdx.x < WarpCount) {
			Result = WarpLevelResults[threadIdx.x];
		}
		Result = ReduceAdd(Result);
		if (threadIdx.x == 0) {
			atomicAdd(Out, Result);
		}
	}
}
#endif

static f32 InitialArray1[ARRAY_SIZE] = {0};
static f32 InitialArray2[ARRAY_SIZE] = {0};
static f32 Result = 0.0f;

static void Init() {
	random_state RandomState = { 0xB40148552A2E3491ULL };
	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		InitialArray1[i] = RandomFloat(&RandomState);
		InitialArray2[i] = RandomFloat(&RandomState);
	}
}


s32 main() {
	Init();

	f32 *GPUArrayA = 0, *GPUArrayB = 0, *GPUResult = 0;
	cudaMalloc(&GPUArrayA, sizeof(InitialArray1));
	cudaMalloc(&GPUArrayB, sizeof(InitialArray2));
	cudaMalloc(&GPUResult, sizeof(Result));
	cudaMemset(GPUResult, 0, sizeof(f32));
	cudaMemcpy(GPUArrayA, InitialArray1, sizeof(InitialArray1), cudaMemcpyHostToDevice);
	cudaMemcpy(GPUArrayB, InitialArray2, sizeof(InitialArray2), cudaMemcpyHostToDevice);

	u32 ThreadCount = (ARRAY_SIZE >= 1024) ? 1024 : ARRAY_SIZE;
	u32 BlockCount = (ARRAY_SIZE + (1023)) / 1024;
	Dot<<<BlockCount, ThreadCount>>>(GPUArrayA, GPUArrayB, GPUResult);
	cudaMemcpy(&Result, GPUResult, sizeof(f32), cudaMemcpyDeviceToHost);

	f32 ExpectedResult = 0.0f;
	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		ExpectedResult += InitialArray1[i] * InitialArray2[i];
	}
	f32 ActualResult = Result;
	if (ExpectedResult == ActualResult) {
		printf(ANSI_COLOR_GREEN "Expected: %.2f -- Actual: %.2f\n", ExpectedResult, ActualResult);
	} else {
		printf(ANSI_COLOR_RED "Expected: %.2f -- Actual: %.2f\n", ExpectedResult, ActualResult);
	}

	printf(ANSI_COLOR_RESET);
}
