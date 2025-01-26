#include "..\base.h"

#include <stdio.h>

static random_state RandomState = { 0xB40148552A2E3491 };
#define ARRAY_SIZE 32 * 32
static u32 InitialArray[ARRAY_SIZE] = {0};
static u32 ResultArray[ARRAY_SIZE] = {0};

void InitRandomIntegers() {
	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		InitialArray[i] = i;
	}
}

__global__ void Map2D(u32 *In, u32 *Out, uint2 GridDimension, u32 Length) {
	u32 X = threadIdx.x + blockIdx.x * blockDim.x;
	u32 Y = threadIdx.y + blockIdx.y * blockDim.y;
	u32 Index = Y * GridDimension.x + X;
	Out[Index] = In[Index] + 10;
}

s32 main() {
	InitRandomIntegers();
	puts("===");

	u32 *GPUArray1 = 0, *GPUArray2 = 0;
	cudaMalloc(&GPUArray1, ARRAY_SIZE * sizeof(int));
	cudaMalloc(&GPUArray2, ARRAY_SIZE * sizeof(int));
	cudaMemcpy(GPUArray1, InitialArray, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);

	dim3 threadDimension(8, 8);
	dim3 blockDimension(4, 4);
	Map2D<<<blockDimension, threadDimension>>>(GPUArray1, GPUArray2, make_uint2(32, 32), ARRAY_SIZE);
	cudaMemcpy(ResultArray, GPUArray2, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	puts("===");

	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		u32 A = InitialArray[i];
		u32 ExpectedResult = A + 10;
		u32 ActualResult = ResultArray[i];
		if (ExpectedResult == ActualResult) {
			printf(ANSI_COLOR_GREEN "%.2u + 10 = %u\n" ANSI_COLOR_RESET, A, ActualResult);
		} else {
			printf(ANSI_COLOR_RED "%.2u + 10 = %u\n" ANSI_COLOR_RESET, A, ActualResult);
		}
	}
}
