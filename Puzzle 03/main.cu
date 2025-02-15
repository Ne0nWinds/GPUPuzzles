#include "..\base.h"

#include <stdio.h>

#define ARRAY_SIZE 4
static u32 InitialArray[ARRAY_SIZE] = {0};
static u32 ResultArray[ARRAY_SIZE] = {0};

void Init() {
	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		InitialArray[i] = i;
	}
}

__global__ void Guard(u32 *GPUMemory, u32 Length) {
	u32 Index = threadIdx.x;
	if (Index < Length) {
		GPUMemory[Index] += 10;
	}
}

s32 main() {
	Init();

	u32 *GPUMemory = 0;
	u32 SizeInBytes = ARRAY_SIZE * sizeof(s32);
	cudaMalloc(&GPUMemory, SizeInBytes);
	cudaMemcpy(GPUMemory, InitialArray, SizeInBytes, cudaMemcpyHostToDevice);

	Guard<<<1, 8>>>(GPUMemory, ARRAY_SIZE);
	cudaMemcpy(ResultArray, GPUMemory, SizeInBytes, cudaMemcpyDeviceToHost);

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
