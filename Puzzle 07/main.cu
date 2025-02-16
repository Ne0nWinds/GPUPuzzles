#include "..\base.h"

#include <stdio.h>

#define BLOCKS 4
#define THREADS_PER_BLOCK 4
#define ARRAY_SIZE (BLOCKS * THREADS_PER_BLOCK)
static u32 InitialArray[ARRAY_SIZE] = {0};
static u32 ResultArray[ARRAY_SIZE] = {0};

void Init() {
	random_state RandomState = { 0xB40148552A2E3491ULL };
	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		InitialArray[i] = RandomInt(&RandomState) % 32;
	}
}

__global__ void Blocks2D(u32 *A, uint2 ArrayDimensions) {
	u32 X = blockIdx.x * blockDim.x + threadIdx.x;
	u32 Y = blockIdx.y * blockDim.y + threadIdx.y;

	A[Y * ArrayDimensions.x + X] += 10;
}

s32 main() {
	Init();

	u32 *GPUArray = 0;
	cudaMalloc(&GPUArray, ARRAY_SIZE * sizeof(u32));
	cudaMemcpy(GPUArray, InitialArray, ARRAY_SIZE * sizeof(u32), cudaMemcpyHostToDevice);

	dim3 BlockCount = dim3(2, 2);
	dim3 ThreadsPerBlock = dim3(2, 2);
	Blocks2D<<<BlockCount, ThreadsPerBlock>>>(GPUArray, make_uint2(4, 4));
	cudaMemcpy(ResultArray, GPUArray, ARRAY_SIZE * sizeof(u32), cudaMemcpyDeviceToHost);

	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		u32 A = InitialArray[i];
		u32 ActualResult = ResultArray[i];
		u32 ExpectedResult = 10 + A;
		if (ExpectedResult == ActualResult) {
			printf(ANSI_COLOR_GREEN "%.2u + 10 = %u\n", A, ActualResult);
		} else {
			printf(ANSI_COLOR_RED "%.2u + 10 = %u\n", A, ActualResult);
		}
	}
	printf(ANSI_COLOR_RESET);
}
