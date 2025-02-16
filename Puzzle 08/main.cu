
#include "..\base.h"

#include <stdio.h>

#define ARRAY_SIZE 1092
static u32 InitialArray[ARRAY_SIZE] = {0};
static u32 ResultArray[ARRAY_SIZE] = {0};

void Init() {
	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		InitialArray[i] = i;
	}
}

__global__ void Shared(u32 *A, u32 Length) {
	__shared__ u32 SharedArray[8];

	u32 GlobalIndex = blockIdx.x * 8 + threadIdx.x;
	u32 LocalIndex = threadIdx.x;

	if (GlobalIndex < Length) {
		SharedArray[LocalIndex] = A[GlobalIndex];
		u32 Result = SharedArray[LocalIndex] + 10;
		A[GlobalIndex] = Result;
	}
}

s32 main() {
	Init();

	u32 *GPUArray = 0;
	u32 SizeInBytes = sizeof(ResultArray);
	cudaMalloc(&GPUArray, SizeInBytes);
	cudaMemcpy(GPUArray, InitialArray, SizeInBytes, cudaMemcpyHostToDevice);

	Shared<<<(ARRAY_SIZE + 7) / 8, 8>>>(GPUArray, ARRAY_SIZE);
	cudaMemcpy(ResultArray, GPUArray, SizeInBytes, cudaMemcpyDeviceToHost);

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
	printf(ANSI_COLOR_RESET);
}
