#include "..\base.h"

#include <stdio.h>

static inline s32 S32_Max(s32 A, s32 B) {
	return (A > B) ? A : B;
}

static random_state RandomState = { 0xB40148552A2E3491 };
#define ARRAY_SIZE 8
static u32 InitialArray[ARRAY_SIZE] = {0};
static u32 ResultArray[ARRAY_SIZE * 2] = {0};

void InitRandomIntegers() {
	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		InitialArray[i] = i;
	}
}

__global__ void Pooling(u32 *In, u32 *Out, u32 Length) {
	__shared__ u32 SharedMemory[8];
	u32 Index = blockIdx.x * blockDim.x + threadIdx.x;

#if 0
	u32 Sum = 0;
	for (u32 Offset = ((s32)Index - 2 > 0) ? (s32)Index - 2 : 0; Offset <= Index; ++Offset) {
		Sum += In[Offset];
	}
#else
	u32 Sum = 0;
	SharedMemory[threadIdx.x] = In[Index];
	__syncthreads();

	for (u32 Offset = ((s32)Index - 2 > 0) ? (s32)Index - 2 : 0; Offset <= Index; ++Offset) {
		Sum += SharedMemory[Offset];
	}
	Out[Index] = Sum;
#endif
}

s32 main() {
	InitRandomIntegers();
	puts("===");

	u32 *GPUArray1 = 0, *GPUArray2 = 0;
	cudaMalloc(&GPUArray1, ARRAY_SIZE * sizeof(s32));
	cudaMalloc(&GPUArray2, sizeof(ResultArray));
	cudaMemset(GPUArray2, 0, sizeof(ResultArray));
	cudaMemcpy(GPUArray1, InitialArray, ARRAY_SIZE * sizeof(s32), cudaMemcpyHostToDevice);

	dim3 threadDimension(8, 1);
	dim3 blockDimension(1, 1);
	Pooling<<<blockDimension, threadDimension>>>(GPUArray1, GPUArray2, ARRAY_SIZE);
	cudaMemcpy(ResultArray, GPUArray2, sizeof(ResultArray), cudaMemcpyDeviceToHost);

	puts("===");

	u32 i = 0;
	for (; i < ARRAY_SIZE; ++i) {
		u32 ExpectedResult = 0;
		for (u32 j = S32_Max((s32)i - 2, 0); j <= i; ++j) {
			ExpectedResult += InitialArray[j];
		}
		u32 ActualResult = ResultArray[i];
		if (ExpectedResult == ActualResult) {
			printf(ANSI_COLOR_GREEN "Expected: %u -- Actual: %u\n", ExpectedResult, ActualResult);
		} else {
			printf(ANSI_COLOR_RED "Expected: %u -- Actual: %u\n", ExpectedResult, ActualResult);
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
