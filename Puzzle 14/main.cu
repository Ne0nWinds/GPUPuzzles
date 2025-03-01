#include "..\base.h"

#include <stdio.h>
#include <assert.h>

static random_state RandomState = { 0xB40148552A2E3491 };

#define MATRIX_SIZE 32

static f32 MatrixA[MATRIX_SIZE * MATRIX_SIZE] = {0};
static f32 MatrixB[MATRIX_SIZE * MATRIX_SIZE] = {0};
static f32 MatrixOut[MATRIX_SIZE * MATRIX_SIZE] = {0};

static void Init() {
	for (u32 i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i) {
		MatrixA[i] = (f32)(RandomInt(&RandomState) % 128) / 128.0f;
		MatrixB[i] = (f32)(RandomInt(&RandomState) % 128) / 128.0f;
		// MatrixA[i] = i;
		// MatrixB[i] = i;
	}
}

__global__ void MatrixMultiply1(f32 *InA, f32 *InB, f32 *Out) {
	const u32 X = blockIdx.x * blockDim.x + threadIdx.x;
	const u32 Y = blockIdx.y * blockDim.y + threadIdx.y;
	if (X < MATRIX_SIZE && Y < MATRIX_SIZE) {
		float Result = 0.0f;
		for (u32 i = 0; i < MATRIX_SIZE; ++i) {
			f32 A = InA[Y * MATRIX_SIZE + i];
			f32 B = InB[i * MATRIX_SIZE + X];
			Result = fma(A, B, Result);
		}
		Out[Y * MATRIX_SIZE + X] = Result;
	}
}

__global__ void MatrixMultiply3(f32 *InA, f32 *InB, f32 *Out) {
	const u32 BlockSize = 16;
	const u32 Row = blockIdx.y * BlockSize;
	const u32 Column = blockIdx.x * BlockSize;
	const u32 ThreadRow = threadIdx.y;
	const u32 ThreadColumn = threadIdx.x;

	__shared__ f32 As[BlockSize * BlockSize];
	__shared__ f32 Bs[BlockSize * BlockSize];

	InA += Row * MATRIX_SIZE;
	InB += Column;
	Out += Row * MATRIX_SIZE + Column;

	float Result = 0.0f;

	for (u32 Index = 0; Index < MATRIX_SIZE; Index += BlockSize) {
		As[ThreadRow * BlockSize + ThreadColumn] = InA[ThreadRow * MATRIX_SIZE + ThreadColumn];
		Bs[ThreadRow * BlockSize + ThreadColumn] = InB[ThreadRow * MATRIX_SIZE + ThreadColumn];

		__syncthreads();

		InA += BlockSize;
		InB += BlockSize * MATRIX_SIZE;

		for (u32 DotIndex = 0; DotIndex < BlockSize; ++DotIndex) {
			f32 A = As[ThreadRow * BlockSize + DotIndex];
			f32 B = Bs[DotIndex * BlockSize + ThreadColumn];
			Result = fma(A, B, Result);
		}
		__syncthreads();
	}

	Out[ThreadRow * MATRIX_SIZE + ThreadColumn] = Result;
}
__global__ void MatrixMultiply4(f32 *A, f32 *B, f32 *Out) {

	const u32 TileSize = 32;
	const u32 NumResultsPerThread = 32;
	__shared__ f32 SharedA[TileSize][TileSize];
	__shared__ f32 SharedB[TileSize][TileSize];

	u32 Row = blockIdx.y * TileSize + threadIdx.y * NumResultsPerThread;
	u32 Column = blockIdx.x * TileSize + threadIdx.x;

	f32 Sum[NumResultsPerThread] = {0};

	for (u32 Tile = 0; Tile < MATRIX_SIZE / TileSize; ++Tile) {
		for (u32 i = 0; i < NumResultsPerThread; ++i) {
			u32 RowIndex = Row + i;
			if (RowIndex < MATRIX_SIZE) {
				SharedA[threadIdx.y * NumResultsPerThread + i][threadIdx.x] = A[RowIndex * MATRIX_SIZE + (Tile * TileSize + threadIdx.x)];
			}
		}

		SharedB[threadIdx.y][threadIdx.x] = B[(Tile * TileSize + threadIdx.y) * MATRIX_SIZE + Column];
		__syncthreads();

		for (u32 k = 0; k < TileSize; ++k) {
			for (u32 i = 0; i < NumResultsPerThread; ++i) {
				u32 RowIndex = Row + i;
				if (RowIndex < MATRIX_SIZE) {
					f32 Multiplicand = SharedA[i][k];
					f32 Multiplier = SharedB[k][threadIdx.x];
					Sum[i] += Multiplicand * Multiplier;
				}
			}
		}
		__syncthreads();
	}

	for (u32 i = 0; i < NumResultsPerThread; ++i) {
		u32 RowIndex = Row + i;
		if (RowIndex < MATRIX_SIZE) {
			Out[RowIndex * MATRIX_SIZE + Column] = Sum[i];
		}
	}
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

	dim3 blockDimension(32, 32);
	dim3 gridDimension((MATRIX_SIZE + 31) / 32, (MATRIX_SIZE + 31) / 32);
	MatrixMultiply5<<<gridDimension, blockDimension>>>(GPUArray1, GPUArray2, GPUArray3);
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
		}
	}
	if (!FoundError) {
		puts(ANSI_COLOR_GREEN "Test Passed!");
	}
#endif
	printf(ANSI_COLOR_RESET);
}
