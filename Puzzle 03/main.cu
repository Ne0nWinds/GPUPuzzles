#include "..\base.h"

#include <stdio.h>

static random_state RandomState = { 0xB40148552A2E3491 };
#define ARRAY_SIZE 4
static u32 InitialArray[ARRAY_SIZE] = {0};
static u32 ResultArray[ARRAY_SIZE] = {0};

void InitRandomIntegers() {
	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		InitialArray[i] = i;
	}
	(void)RandomState;
}

__global__ void Guard(u32 *In, u32 *Out, u32 Length) {
	u32 Index = threadIdx.x;
	if (Index < Length) {
		Out[Index] = In[Index] + 10;
	}
}

s32 main() {
	InitRandomIntegers();

	u32 *GPUArray1 = 0, *GPUArray2 = 0;
	cudaMalloc(&GPUArray1, ARRAY_SIZE * sizeof(int));
	cudaMalloc(&GPUArray2, ARRAY_SIZE * sizeof(int));
	cudaMemcpy(GPUArray1, InitialArray, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(GPUArray2, InitialArray, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);

	Guard<<<1, 8>>>(GPUArray1, GPUArray2, ARRAY_SIZE);
	cudaMemcpy(ResultArray, GPUArray2, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	puts("===");

	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		u32 A = InitialArray[i];
		u32 ExpectedResult = A + 10;
		u32 ActualResult = ResultArray[i];
		if (ExpectedResult == ActualResult) {
			printf(ANSI_COLOR_GREEN "%.2u + 10 = %u\n" ANSI_COLOR_RESET, A, ActualResult);
		} else {
			printf(ANSI_COLOR_RED "%.2u + 10 = %u\n" ANSI_COLOR_RESET, A, ActualResult);
		}
	}
}
