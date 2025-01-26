
#include "..\base.h"

#include <stdio.h>

static random_state RandomState = { 0xB40148552A2E3491 };
#define ARRAY_SIZE 8
static u32 InitialArray[ARRAY_SIZE] = {0};
static u32 ResultArray[ARRAY_SIZE * 2] = {0};

void InitRandomIntegers() {
	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		InitialArray[i] = i;
	}
}

__global__ void Blocks(u32 *In, u32 *Out, u32 Length) {
	__shared__ u32 SharedArray[4];
	u32 Index = blockIdx.x * blockDim.x + threadIdx.x;
	u32 LocalIndex = threadIdx.x;

	if (Index < Length) {
		SharedArray[LocalIndex] = In[Index];
		__syncthreads();
	}

	Out[Index] = SharedArray[LocalIndex] + 10;
}

s32 main() {
	InitRandomIntegers();
	puts("===");

	u32 *GPUArray1 = 0, *GPUArray2 = 0;
	cudaMalloc(&GPUArray1, ARRAY_SIZE * sizeof(s32));
	cudaMalloc(&GPUArray2, sizeof(ResultArray));
	cudaMemset(GPUArray2, 0, sizeof(ResultArray));
	cudaMemcpy(GPUArray1, InitialArray, ARRAY_SIZE * sizeof(s32), cudaMemcpyHostToDevice);

	dim3 threadDimension(4, 1);
	dim3 blockDimension(2, 1);
	Blocks<<<blockDimension, threadDimension>>>(GPUArray1, GPUArray2, ARRAY_SIZE);
	cudaMemcpy(ResultArray, GPUArray2, sizeof(ResultArray), cudaMemcpyDeviceToHost);

	puts("===");

	u32 i = 0;
	for (; i < ARRAY_SIZE; ++i) {
		u32 A = InitialArray[i];
		u32 ExpectedResult = A + 10;
		u32 ActualResult = ResultArray[i];
		if (ExpectedResult == ActualResult) {
			printf(ANSI_COLOR_GREEN "%.2u + 10 = %u\n", A, ActualResult);
		} else {
			printf(ANSI_COLOR_RED "%.2u + 10 = %u\n", A, ActualResult);
		}
	}
	for (; i < ARRAY_SIZE * 2; ++i) {
		if (ResultArray[i] != 0) {
			printf(ANSI_COLOR_RED "Wrote past the end of the array at index: %u", i);
			break;
		}
	}
	printf(ANSI_COLOR_RESET);
}
