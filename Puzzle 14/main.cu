#include "..\base.h"

#include <stdio.h>
#include <assert.h>

static random_state RandomState = { 0xB40148552A2E3491 };

__device__ f32 ReduceAdd(f32 Value) {
	u32 Mask = 0xFFFFFFFF;
	Value += __shfl_down_sync(Mask, Value, 16);
	Value += __shfl_down_sync(Mask, Value, 8);
	Value += __shfl_down_sync(Mask, Value, 4);
	Value += __shfl_down_sync(Mask, Value, 2);
	Value += __shfl_down_sync(Mask, Value, 1);
	return Value;
}

#define MATRIX_SIZE 2048

static f32 MatrixA[MATRIX_SIZE * MATRIX_SIZE] = {0};
static f32 MatrixB[MATRIX_SIZE * MATRIX_SIZE] = {0};
static f32 MatrixOut[MATRIX_SIZE * MATRIX_SIZE] = {0};

static constexpr void Init() {
	for (u32 i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i) {
		MatrixA[i] = RandomFloat(&RandomState);
		MatrixB[i] = RandomFloat(&RandomState);
		// MatrixA[i] = i;
		// MatrixB[i] = i;
	}
}

#if 0
__global__ void MatrixMultiply(f32 *InA, f32 *InB, u32 MatrixSize, f32 *Out) {
	const u32 BlockSize = 16;
	const u32 X = blockIdx.x * BlockSize + threadIdx.x;
	const u32 Y = blockIdx.y * BlockSize + threadIdx.y;

	float Result = 0.0f;
	for (u32 I = 0; I < MatrixSize; ++I) {
		f32 A = InA[Y * MatrixSize + I];
		f32 B = InB[I * MatrixSize + X];
		Result = fma(A, B, Result);
	}
	Out[Y * MATRIX_SIZE + X] = Result;
}
#else
__global__ void MatrixMultiply(f32 *InA, f32 *InB, u32 MatrixSize, f32 *Out) {
	const u32 X = blockIdx.x * 16 + threadIdx.x;
	const u32 Y = blockIdx.y * 16 + threadIdx.y;

	__shared__ f32 SharedMemoryA[16][16];
	__shared__ f32 SharedMemoryB[16][16];

	f32 Result = 0.0f;
	for (u32 Offset = 0; Offset < MatrixSize; Offset += 16) {
		SharedMemoryA[threadIdx.y][threadIdx.x] = InA[Y * MatrixSize + (Offset + threadIdx.x)];
		SharedMemoryB[threadIdx.y][threadIdx.x] = InB[(Offset + threadIdx.y) * MatrixSize + X];
		__syncthreads();

		#pragma unroll
		for (u32 I = 0; I < 16; ++I) {
			f32 A = SharedMemoryA[threadIdx.y][I];
			f32 B = SharedMemoryB[I][threadIdx.x];
			Result = fma(A, B, Result);
		}
		__syncthreads();
	}

	Out[Y * MATRIX_SIZE + X] = Result;
}
#endif

s32 main() {
	Init();

	f32 *GPUArray1 = 0, *GPUArray2 = 0, *GPUArray3 = 0;
	cudaMalloc(&GPUArray1, sizeof(MatrixA));
	cudaMalloc(&GPUArray2, sizeof(MatrixB));
	cudaMalloc(&GPUArray3, sizeof(MatrixOut));
	cudaMemcpy(GPUArray1, MatrixA, sizeof(MatrixA), cudaMemcpyHostToDevice);
	cudaMemcpy(GPUArray2, MatrixB, sizeof(MatrixB), cudaMemcpyHostToDevice);
	cudaMemset(GPUArray3, 0, sizeof(MatrixOut));

	dim3 GridDimension((MATRIX_SIZE + 15) / 16, (MATRIX_SIZE + 15) / 16);
	MatrixMultiply<<<GridDimension, dim3(16, 16)>>>(GPUArray1, GPUArray2, MATRIX_SIZE, GPUArray3);
	cudaMemcpy(MatrixOut, GPUArray3, sizeof(MatrixOut), cudaMemcpyDeviceToHost);

#if 0
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
			// printf(ANSI_COLOR_GREEN "Expected: %.2f -- Actual: %.2f\n", ExpectedResult, ActualResult);
		} else {
			printf(ANSI_COLOR_RED "Expected: %.2f -- Actual: %.2f at Index: %u\n", ExpectedResult, ActualResult, i);
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
