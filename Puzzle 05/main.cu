#include "..\base.h"

#include <stdio.h>

static random_state RandomState = { 0xB40148552A2E3491 };
#define ARRAY_SIZE 16
static u32 InitialArrayHorizontal[ARRAY_SIZE] = {0};
static u32 InitialArrayVertical[ARRAY_SIZE] = {0};
static u32 ResultArray[ARRAY_SIZE] = {0};

void InitRandomIntegers() {
	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		InitialArrayHorizontal[i] = i;
	}
	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		InitialArrayVertical[i] = i;
	}
}

__global__ void Map2D(u32 *Horizontal, u32 *Vertical, u32 *Out, u32 Length) {
	u32 X = threadIdx.x;
	u32 Y = threadIdx.y;

	u32 A = Horizontal[X];
	u32 B = Vertical[Y];
	Out[Y * 4 + X] = A + B;
}

s32 main() {
	InitRandomIntegers();
	puts("===");

	u32 *GPUArray1 = 0, *GPUArray2 = 0, *GPUArray3 = 0;
	cudaMalloc(&GPUArray1, ARRAY_SIZE * sizeof(int));
	cudaMalloc(&GPUArray2, ARRAY_SIZE * sizeof(int));
	cudaMalloc(&GPUArray3, ARRAY_SIZE * sizeof(int));
	cudaMemcpy(GPUArray1, InitialArrayVertical, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(GPUArray2, InitialArrayHorizontal, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);

	dim3 threadDimension(4, 4);
	dim3 blockDimension(1, 1);
	Map2D<<<blockDimension, threadDimension>>>(GPUArray1, GPUArray2, GPUArray3, ARRAY_SIZE);
	cudaMemcpy(ResultArray, GPUArray3, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	puts("===");

	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		u32 X = i / 4;
		u32 Y = i % 4;
		u32 A = InitialArrayHorizontal[X];
		u32 B = InitialArrayVertical[Y];
		u32 ExpectedResult = A + B;
		u32 ActualResult = ResultArray[Y * 4 + X];
		if (ExpectedResult == ActualResult) {
			printf(ANSI_COLOR_GREEN "%.2u + %.2u = %u\n" ANSI_COLOR_RESET, A, B, ActualResult);
		} else {
			printf(ANSI_COLOR_RED "%.2u + %.2u = %u\n" ANSI_COLOR_RESET, A, B, ActualResult);
		}
	}
}
