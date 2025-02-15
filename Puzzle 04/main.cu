#include "..\base.h"

#include <stdio.h>

#define ARRAY_WIDTH 8
#define ARRAY_SIZE (ARRAY_WIDTH * ARRAY_WIDTH)
static u32 InitialArray[ARRAY_WIDTH][ARRAY_WIDTH] = {0};
static u32 ResultArray[ARRAY_WIDTH][ARRAY_WIDTH] = {0};


void Init() {
	random_state RandomState = { 0xC96A64845C2BEA8DULL };

	for (u32 Y = 0; Y < ARRAY_SIZE; ++Y) {
		for (u32 X = 0; X < ARRAY_SIZE; ++X) {
			InitialArray[Y][X] = RandomInt(&RandomState) % 32;
		}
	}
}

__global__ void Map2D(u32 *A, uint2 ArrayDimensions) {
	u32 X = threadIdx.x;
	u32 Y = threadIdx.y;
	if (X < ArrayDimensions.x && Y < ArrayDimensions.y) {
		u32 Index = Y * ArrayDimensions.x + X;
		A[Index] += 10;
	}
}

s32 main() {
	Init();

	u32 *GPUMemory = 0;
	u32 SizeInBytes = ARRAY_SIZE * sizeof(s32);
	cudaMalloc(&GPUMemory, SizeInBytes);
	cudaMemcpy(GPUMemory, InitialArray, SizeInBytes, cudaMemcpyHostToDevice);

	dim3 threadDimension(ARRAY_WIDTH + 1, ARRAY_WIDTH + 1);
	Map2D<<<1, threadDimension>>>(GPUMemory, make_uint2(ARRAY_WIDTH, ARRAY_WIDTH));
	cudaMemcpy(ResultArray, GPUMemory, SizeInBytes, cudaMemcpyDeviceToHost);

	for (u32 Y = 0; Y < ARRAY_WIDTH; ++Y) {
		for (u32 X = 0; X < ARRAY_WIDTH; ++X) {
			u32 A = InitialArray[Y][X];
			u32 ExpectedResult = A + 10;
			u32 ActualResult = ResultArray[Y][X];
			if (ExpectedResult == ActualResult) {
				printf(ANSI_COLOR_GREEN "%.2u + 10 = %u\n", A, ActualResult);
			} else {
				printf(ANSI_COLOR_RED "%.2u + 10 = %u\n", A, ActualResult);
			}
		}
	}
	printf(ANSI_COLOR_RESET);
}
