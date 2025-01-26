#include "..\base.h"

#include <stdio.h>

static random_state RandomState = { 0xB40148552A2E3491 };
#define THREAD_SIZE 256
#define ARRAY_SIZE 0x7FFFF00
static f32 InitialArray[ARRAY_SIZE] = {0};
static f32 ResultArray[ARRAY_SIZE / 32] = {0};

static void InitRandomIntegers() {
	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		InitialArray[i] = RandomInt(&RandomState) % 32;
	}
}

__global__ void HorizontalAdd(f32 *In, f32 *Out) {
	u32 Index = threadIdx.x + blockIdx.x * THREAD_SIZE;

	f32 Result = In[Index];

	for (u32 Shift = 16; Shift > 0; Shift /= 2) {
		Result += __shfl_down_sync(0xFFFF'FFFF, Result, Shift);
	}
	if ((threadIdx.x & (32 - 1)) == 0) {
		u32 OutIndex = threadIdx.x / 32 + blockIdx.x * (THREAD_SIZE / 32);
		Out[OutIndex] = Result;
	}
}

s32 main() {
	InitRandomIntegers();

	f32 *GPUArray1 = 0, *GPUArray2 = 0;
	cudaMalloc(&GPUArray1, sizeof(InitialArray));
	cudaMalloc(&GPUArray2, sizeof(ResultArray));
	cudaMemcpy(GPUArray1, InitialArray, sizeof(InitialArray), cudaMemcpyHostToDevice);

	dim3 threadDimension(THREAD_SIZE, 1);
	dim3 blockDimension(ARRAY_SIZE / THREAD_SIZE, 1);
	HorizontalAdd<<<blockDimension, threadDimension>>>(GPUArray1, GPUArray2);
	cudaMemcpy(ResultArray, GPUArray2, sizeof(ResultArray), cudaMemcpyDeviceToHost);

#if 0
	bool FoundError = false;
	for (u32 i = 0; i < ARRAY_SIZE; i += 32) {
		f32 ExpectedResult = 0.0f;
		for (u32 j = 0; j < 32; ++j) {
			ExpectedResult += InitialArray[i + j];
		}
		f32 ActualResult = ResultArray[i / 32];
		if (ExpectedResult == ActualResult) {
			// printf(ANSI_COLOR_GREEN "Expected: %.2f -- Actual: %.2f\n", ExpectedResult, ActualResult);
		} else {
			printf(ANSI_COLOR_RED "Expected: %.2f -- Actual: %.2f at Index: %u\n", ExpectedResult, ActualResult, i / 32);
			printf("");
			FoundError = true;
			break;
		}
	}
	if (!FoundError) {
		puts(ANSI_COLOR_GREEN "Test Passed!");
	}
#endif
	printf(ANSI_COLOR_RESET);
}
