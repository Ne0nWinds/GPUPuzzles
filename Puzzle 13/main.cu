#include "..\base.h"

#include <stdio.h>

// static random_state RandomState = { 0xB40148552A2E3491 };
#define COLUMN_SIZE 4
#define ROW_SIZE 6
#define ARRAY_SIZE ROW_SIZE * COLUMN_SIZE
static f32 InitialArray[ARRAY_SIZE] = {0};
static f32 ResultArray[ROW_SIZE] = {0};

static void InitRandomIntegers() {
	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		InitialArray[i] = i;
	}
}

__global__ void AxisSum(f32 *In, f32 *Out) {
	u32 Value = In[blockIdx.x * ROW_SIZE + threadIdx.x];
	Value += __shfl_down_sync(__activemask(), Value, 4);
	Value += __shfl_down_sync(__activemask(), Value, 2);
	Value += __shfl_down_sync(__activemask(), Value, 1);
	if (threadIdx.x == 0) {
		Out[blockIdx.x] = Value;
	}
}

s32 main() {
	InitRandomIntegers();

	f32 *GPUArray1 = 0, *GPUArray2 = 0;
	cudaMalloc(&GPUArray1, sizeof(InitialArray));
	cudaMalloc(&GPUArray2, sizeof(ResultArray));
	cudaMemcpy(GPUArray1, InitialArray, sizeof(InitialArray), cudaMemcpyHostToDevice);

	dim3 threadDimension(ROW_SIZE, 1);
	dim3 blockDimension(COLUMN_SIZE, 1);
	AxisSum<<<blockDimension, threadDimension>>>(GPUArray1, GPUArray2);
	cudaMemcpy(ResultArray, GPUArray2, sizeof(ResultArray), cudaMemcpyDeviceToHost);

#if 1
	bool FoundError = false;
	for (u32 i = 0; i < ARRAY_SIZE; i += ROW_SIZE) {
		f32 ExpectedResult = 0.0f;
		for (u32 j = 0; j < ROW_SIZE; ++j) {
			ExpectedResult += InitialArray[i + j];
		}
		f32 ActualResult = ResultArray[i / ROW_SIZE];
		if (ExpectedResult == ActualResult) {
			printf(ANSI_COLOR_GREEN "Expected: %.2f -- Actual: %.2f\n", ExpectedResult, ActualResult);
		} else {
			printf(ANSI_COLOR_RED "Expected: %.2f -- Actual: %.2f at Index: %u\n", ExpectedResult, ActualResult, i / ROW_SIZE);
			FoundError = true;
			// break;
		}
	}
	if (!FoundError) {
		puts(ANSI_COLOR_GREEN "Test Passed!");
	}
#endif
	printf(ANSI_COLOR_RESET);
}
