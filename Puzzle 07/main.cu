#include "..\base.h"

#include <stdio.h>

#define POISON_VALUE 0x29
static random_state RandomState = { 0xB40148552A2E3491 };
#define ARRAY_SIZE 6 * 6
static u32 InitialArray[ARRAY_SIZE] = {0};
static u32 ResultArray[ARRAY_SIZE * 2] = {0};

void InitRandomIntegers() {
	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		InitialArray[i] = i;
	}
}

__global__ void Blocks(u32 *In, u32 *Out, u32 Length) {
	u32 X = blockIdx.x * blockDim.x + threadIdx.x;
	u32 Y = blockIdx.y * blockDim.y + threadIdx.y;
	u32 Index = Y * 6 + X;
	Out[Index] = In[Index] + 10;
}

s32 main() {
	InitRandomIntegers();
	puts("===");

	u32 *GPUArray1 = 0, *GPUArray2 = 0;
	cudaMalloc(&GPUArray1, ARRAY_SIZE * sizeof(s32));
	cudaMalloc(&GPUArray2, sizeof(ResultArray));
	cudaMemset(GPUArray2, POISON_VALUE, sizeof(ResultArray));
	cudaMemcpy(GPUArray1, InitialArray, ARRAY_SIZE * sizeof(s32), cudaMemcpyHostToDevice);

	dim3 threadDimension(3, 3);
	dim3 blockDimension(2, 2);
	Blocks<<<blockDimension, threadDimension>>>(GPUArray1, GPUArray2, ARRAY_SIZE);
	cudaMemcpy(ResultArray, GPUArray2, sizeof(ResultArray), cudaMemcpyDeviceToHost);

	puts("===");

	u32 i = 0;
	for (; i < ARRAY_SIZE; ++i) {
		u32 A = InitialArray[i];
		u32 ExpectedResult = A + 10;
		u32 ActualResult = ResultArray[i];
		if (ExpectedResult == ActualResult) {
			printf(ANSI_COLOR_GREEN "%.2u + 10 = %u\n", A, ActualResult);
		} else {
			printf(ANSI_COLOR_RED "%.2u + 10 = %u\n", A, ActualResult);
		}
	}
	for (; i < ARRAY_SIZE * 2; ++i) {
		u8 A = ResultArray[i] & 0x000000FF;
		u8 B = (ResultArray[i] & 0x0000FF00) >> 8;
		u8 C = (ResultArray[i] & 0x00FF0000) >> 16;
		u8 D = (ResultArray[i] & 0xFF000000) >> 24;
		if (A != POISON_VALUE || B != POISON_VALUE || C != POISON_VALUE || D != POISON_VALUE) {
			printf(ANSI_COLOR_RED "Wrote past the end of the array at index: %u", i);
			break;
		}
	}
	printf(ANSI_COLOR_RESET);
}
