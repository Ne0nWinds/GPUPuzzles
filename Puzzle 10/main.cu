#include "..\base.h"

#include <stdio.h>

static random_state RandomState = { 0xB40148552A2E3491 };
#define ARRAY_SIZE 8 * 1024 * 1024
static f32 InitialArray1[ARRAY_SIZE] = {0};
static f32 InitialArray2[ARRAY_SIZE] = {0};
static f32 ResultArray[ARRAY_SIZE / 8] = {0};

void InitRandomIntegers() {
	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		InitialArray1[i] = RandomInt(&RandomState) % 32;
		InitialArray2[i] = RandomInt(&RandomState) % 32;
	}
}

__global__ void Dot(f32 *In1, f32 *In2, f32 *Out, u32 Length) {

#if 0
	__shared__ u32 SharedMemory[16];
	u32 GlobalIndex = blockIdx.x * 8 + threadIdx.x;
	SharedMemory[threadIdx.x + 0] = In1[GlobalIndex];
	SharedMemory[threadIdx.x + 8] = In2[GlobalIndex];
	__syncthreads();

	f32 Result = 0.0f;
	for (u32 i = 0; i < 8; ++i) {
		Result += SharedMemory[i] * SharedMemory[i + 8];
	}
	Out[GlobalIndex] = Result;
#elif 0
	__shared__ u32 SharedMemory[16];
	u32 GlobalIndex = blockIdx.x * 8 + threadIdx.x;
	SharedMemory[threadIdx.x + 0] = In1[GlobalIndex];
	SharedMemory[threadIdx.x + 8] = In2[GlobalIndex];
	__syncthreads();

	SharedMemory[threadIdx.x] = SharedMemory[threadIdx.x] * SharedMemory[threadIdx.x + 8];
	__syncthreads();
	SharedMemory[threadIdx.x] += SharedMemory[threadIdx.x + 4];
	__syncthreads();
	SharedMemory[threadIdx.x] += SharedMemory[threadIdx.x + 2];
	__syncthreads();
	Out[blockIdx.x] = SharedMemory[threadIdx.x] + SharedMemory[threadIdx.x + 1];
#else
	u32 GlobalIndex = blockIdx.x * 8 + threadIdx.x;
	f32 Value = In1[GlobalIndex] * In2[GlobalIndex];
	for (u32 Offset = 4; Offset > 0; Offset /= 2) {
		Value += __shfl_down_sync(0xFFFF'FFFF, Value, Offset);
	}
	Out[blockIdx.x] = Value;
#endif
}

s32 main() {
	InitRandomIntegers();
	puts("===");

	f32 *GPUArray1 = 0, *GPUArray2 = 0, *GPUArray3 = 0;
	cudaMalloc(&GPUArray1, sizeof(InitialArray1));
	cudaMalloc(&GPUArray2, sizeof(InitialArray2));
	cudaMalloc(&GPUArray3, sizeof(ResultArray));
	cudaMemset(GPUArray3, 0, sizeof(f32));
	cudaMemcpy(GPUArray1, InitialArray1, sizeof(InitialArray1), cudaMemcpyHostToDevice);
	cudaMemcpy(GPUArray2, InitialArray2, sizeof(InitialArray2), cudaMemcpyHostToDevice);

	dim3 threadDimension(8, 1);
	dim3 blockDimension(ARRAY_SIZE / 8, 1);
	Dot<<<blockDimension, threadDimension>>>(GPUArray1, GPUArray2, GPUArray3, ARRAY_SIZE);
	cudaMemcpy(ResultArray, GPUArray3, sizeof(ResultArray), cudaMemcpyDeviceToHost);

	puts("===");

#if 0
	for (u32 i = 0; i < ARRAY_SIZE; i += 8) {
		f32 ExpectedResult = 0.0f;
		for (u32 j = 0; j < 8; j += 1) {
			ExpectedResult += InitialArray1[i + j] * InitialArray2[i + j];
		}
		f32 ActualResult = ResultArray[i / 8];
		printf("Expected: %.2f -- Actual: %.2f\n", ExpectedResult, ActualResult);
	}
#endif
	printf(ANSI_COLOR_RESET);
}
