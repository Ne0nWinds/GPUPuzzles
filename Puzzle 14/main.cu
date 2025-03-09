#include "..\base.h"

#include <stdio.h>
#include <assert.h>


#define MATRIX_SIZE 2048

static f32 MatrixA[MATRIX_SIZE * MATRIX_SIZE] = {0};
static f32 MatrixB[MATRIX_SIZE * MATRIX_SIZE] = {0};
static f32 MatrixOut[MATRIX_SIZE * MATRIX_SIZE] = {0};

static constexpr void Init() {
	random_state RandomState = { 0xB40148552A2E3491ULL };
	for (u32 i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i) {
		MatrixA[i] = RandomFloat(&RandomState);
		MatrixB[i] = RandomFloat(&RandomState);
	}
}

#if 1
__global__ void MatrixMultiply(f32 *A, f32 *B, f32 *C, u32 MatrixSize) {
	const u32 X = blockIdx.x * blockDim.x + threadIdx.x;
	const u32 Y = blockIdx.y * blockDim.y + threadIdx.y;

	float Result = 0.0f;
	for (u32 I = 0; I < MatrixSize; ++I) {
		f32 ValueA = A[Y * MatrixSize + I];
		f32 ValueB = B[I * MatrixSize + X];
		Result = fma(ValueA, ValueB, Result);
	}
	C[Y * MATRIX_SIZE + X] = Result;
}
#else
__global__ void MatrixMultiply(f32 *A, f32 *B, f32 *C, u32 MatrixSize) {
	const u32 X = blockIdx.x * 16 + threadIdx.x;
	const u32 Y = blockIdx.y * 16 + threadIdx.y;

	__shared__ f32 SharedMemoryA[16][16];
	__shared__ f32 SharedMemoryB[16][16];

	f32 Result = 0.0f;
	for (u32 Offset = 0; Offset < MatrixSize; Offset += 16) {
		SharedMemoryA[threadIdx.y][threadIdx.x] = A[Y * MatrixSize + (Offset + threadIdx.x)];
		SharedMemoryB[threadIdx.y][threadIdx.x] = B[(Offset + threadIdx.y) * MatrixSize + X];
		__syncthreads();

		for (u32 I = 0; I < 16; ++I) {
			f32 ValueA = SharedMemoryA[threadIdx.y][I];
			f32 ValueB = SharedMemoryB[I][threadIdx.x];
			Result = fma(ValueA, ValueB, Result);
		}
		__syncthreads();
	}

	C[Y * MATRIX_SIZE + X] = Result;
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

	const dim3 BlockSize(16, 16);
	const dim3 GridSize(MATRIX_SIZE / BlockSize.x, MATRIX_SIZE / BlockSize.y);
	MatrixMultiply<<<GridSize, BlockSize>>>(GPUArray1, GPUArray2, GPUArray3, MATRIX_SIZE);
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
			printf(ANSI_COLOR_GREEN "Expected: %.2f | Actual: %.2f\n", ExpectedResult, ActualResult);
		} else {
			printf(ANSI_COLOR_RED "Expected: %.2f | Actual: %.2f at Index: %u\n", ExpectedResult, ActualResult, i);
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
