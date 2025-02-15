#include "..\base.h"

#include <stdio.h>

#define VECTOR_SIZE 4
#define MATRIX_SIZE (VECTOR_SIZE * VECTOR_SIZE)
static u32 ColumnVector[VECTOR_SIZE] = {0};
static u32 RowVector[VECTOR_SIZE] = {0};
static u32 ResultArray[MATRIX_SIZE] = {0};

static void Init() {
	random_state RandomState = { 0xB40148552A2E3491ULL };
	for (u32 i = 0; i < VECTOR_SIZE; ++i) {
		ColumnVector[i] = RandomInt(&RandomState) % 32;
		RowVector[i] = RandomInt(&RandomState) % 32;
	}
}

#define USE_2D_GRID 0

#if USE_2D_GRID
__global__ void Map2D(u32 *Row, u32 *Column, u32 *OutMatrix) {
	u32 X = threadIdx.x;
	u32 Y = threadIdx.y;

	u32 A = Row[X];
	u32 B = Column[Y];
	OutMatrix[Y * VECTOR_SIZE + X] = A + B;
}
#else
__global__ void Map2D(u32 *Row, u32 *Column, u32 *OutMatrix) {
	u32 X = threadIdx.x / 4;
	u32 Y = threadIdx.x % 4;

	u32 A = Row[X];
	u32 B = Column[Y];
	OutMatrix[Y * VECTOR_SIZE + X] = A + B;
}
#endif

s32 main() {
	Init();

	u32 *GPUColumnVector = 0, *GPURowVector = 0, *GPUMatrix = 0;
	s32 VectorSizeInBytes = VECTOR_SIZE * sizeof(s32);
	s32 MatrixSizeInBytes = MATRIX_SIZE * sizeof(s32);
	cudaMalloc(&GPUColumnVector, VectorSizeInBytes);
	cudaMalloc(&GPURowVector, VectorSizeInBytes);
	cudaMalloc(&GPUMatrix, MatrixSizeInBytes);
	cudaMemcpy(GPUColumnVector, ColumnVector, VectorSizeInBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(GPURowVector, RowVector, VectorSizeInBytes, cudaMemcpyHostToDevice);
	cudaMemset(GPUMatrix, 0, MatrixSizeInBytes);

#if USE_2D_GRID
	dim3 threadDimension(VECTOR_SIZE, VECTOR_SIZE);
	Map2D<<<1, threadDimension>>>(GPURowVector, GPUColumnVector, GPUMatrix);
#else
	u32 ThreadCount = MATRIX_SIZE;
	Map2D<<<1, ThreadCount>>>(GPURowVector, GPUColumnVector, GPUMatrix);
#endif
	cudaMemcpy(ResultArray, GPUMatrix, MatrixSizeInBytes, cudaMemcpyDeviceToHost);

	for (u32 i = 0; i < MATRIX_SIZE; ++i) {
		u32 X = i / 4;
		u32 Y = i % 4;
		u32 A = RowVector[X];
		u32 B = ColumnVector[Y];
		u32 ExpectedResult = A + B;
		u32 ActualResult = ResultArray[Y * 4 + X];
		if (ExpectedResult == ActualResult) {
			printf(ANSI_COLOR_GREEN "%.2u + %.2u = %u\n" ANSI_COLOR_RESET, A, B, ActualResult);
		} else {
			printf(ANSI_COLOR_RED "%.2u + %.2u = %u\n" ANSI_COLOR_RESET, A, B, ActualResult);
		}
	}
}
