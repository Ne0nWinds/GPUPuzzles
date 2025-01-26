#include "..\base.h"

#include <stdio.h>

static random_state RandomState = { 0xB40148552A2E3491 };
#define MATRIX_SIZE 4
static f32 MatrixA[MATRIX_SIZE * MATRIX_SIZE] = {0};
static f32 MatrixB[MATRIX_SIZE * MATRIX_SIZE] = {0};
static f32 MatrixOut[MATRIX_SIZE * MATRIX_SIZE] = {0};

static void Init() {
	for (u32 i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i) {
		MatrixA[i] = (f32)(RandomInt(&RandomState) % 32) / 32.0f;
		MatrixB[i] = (f32)(RandomInt(&RandomState) % 32) / 32.0f;
	}
}

__global__ void MatrixMultiply(f32 *InA, f32 *InB, f32 *Out) {
#if 0
	// Reference
	u32 Column = threadIdx.x % MATRIX_SIZE;
	u32 Row = threadIdx.x / MATRIX_SIZE;
	f32 Result = 0.0f;
	for (u32 i = 0; i < MATRIX_SIZE; ++i) {
		Result = fma(InA[Row * MATRIX_SIZE + i], InB[i * MATRIX_SIZE + Column], Result);
	}
	Out[threadIdx.x] = Result;
#elif 1
	// 4x4 Special Case
	u32 Column = threadIdx.x % 4;
	u32 Row = threadIdx.x / 4;
	f32 ValueA = InA[threadIdx.x];
	f32 ValueB = InB[threadIdx.x];

	f32 Result = 0.0f;
	Result = fma(__shfl_sync(0xFFFF, ValueA, Row * 4 + 0), __shfl_sync(0xFFFF, ValueB, 0 * 4 + Column), Result);
	Result = fma(__shfl_sync(0xFFFF, ValueA, Row * 4 + 1), __shfl_sync(0xFFFF, ValueB, 1 * 4 + Column), Result);
	Result = fma(__shfl_sync(0xFFFF, ValueA, Row * 4 + 2), __shfl_sync(0xFFFF, ValueB, 2 * 4 + Column), Result);
	Result = fma(__shfl_sync(0xFFFF, ValueA, Row * 4 + 3), __shfl_sync(0xFFFF, ValueB, 3 * 4 + Column), Result);
	Out[threadIdx.x] = Result;
#endif
}

s32 main() {
	Init();

	f32 *GPUArray1 = 0, *GPUArray2 = 0, *GPUArray3 = 0;
	cudaMalloc(&GPUArray1, sizeof(MatrixA));
	cudaMalloc(&GPUArray2, sizeof(MatrixB));
	cudaMalloc(&GPUArray3, sizeof(MatrixOut));
	cudaMemcpy(GPUArray1, MatrixA, sizeof(MatrixA), cudaMemcpyHostToDevice);
	cudaMemcpy(GPUArray2, MatrixB, sizeof(MatrixB), cudaMemcpyHostToDevice);
	cudaMemset(GPUArray3, 0, sizeof(MatrixOut));

	dim3 threadDimension(MATRIX_SIZE * MATRIX_SIZE, 1);
	dim3 blockDimension(1, 1);
	MatrixMultiply<<<blockDimension, threadDimension>>>(GPUArray1, GPUArray2, GPUArray3);
	cudaMemcpy(MatrixOut, GPUArray3, sizeof(MatrixOut), cudaMemcpyDeviceToHost);

#if 1
	bool FoundError = false;
	for (u32 i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i += 1) {
		f32 ExpectedResult = 0.0f;
		u32 Column = i % MATRIX_SIZE;
		u32 Row = i / MATRIX_SIZE;
		for (u32 j = 0; j < MATRIX_SIZE; ++j) {
			ExpectedResult += MatrixA[Row * MATRIX_SIZE + j] * MatrixB[j * MATRIX_SIZE + Column];
		}
		f32 ActualResult = MatrixOut[i];
		if (ExpectedResult == ActualResult) {
			printf(ANSI_COLOR_GREEN "Expected: %.2f -- Actual: %.2f\n", ExpectedResult, ActualResult);
		} else {
			printf(ANSI_COLOR_RED "Expected: %.2f -- Actual: %.2f at Index: %u\n", ExpectedResult, ActualResult, i);
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
