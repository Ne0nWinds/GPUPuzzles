#include "..\base.h"

#include <stdio.h>

#define ARRAY_SIZE 64
static u32 ArrayA[ARRAY_SIZE] = {0};
static u32 ArrayB[ARRAY_SIZE] = {0};
static u32 ResultArray[ARRAY_SIZE] = {0};

void Init() {
	random_state RandomState = { 0x2528260DA722CC5ULL };

	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		ArrayA[i] = RandomInt(&RandomState) % 32;
		ArrayB[i] = RandomInt(&RandomState) % 32;
	}
}

__global__ void Add(u32 *A, u32 *B, u32 *C) {
	u32 Index = threadIdx.x;
	C[Index] = A[Index] + B[Index];
}

s32 main() {
	Init();

	u32 *GPUArrayA = 0, *GPUArrayB = 0, *GPUArrayC = 0;
	const u32 SizeInBytes = ARRAY_SIZE * sizeof(s32);
	cudaMalloc(&GPUArrayA, SizeInBytes);
	cudaMalloc(&GPUArrayB, SizeInBytes);
	cudaMalloc(&GPUArrayC, SizeInBytes);
	cudaMemcpy(GPUArrayA, ArrayA, SizeInBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(GPUArrayB, ArrayB, SizeInBytes, cudaMemcpyHostToDevice);
	cudaMemset(GPUArrayC, 0, SizeInBytes);

	Add<<<1, ARRAY_SIZE>>>(GPUArrayA, GPUArrayB, GPUArrayC);
	cudaMemcpy(ResultArray, GPUArrayC, SizeInBytes, cudaMemcpyDeviceToHost);

#if 0
	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		u32 A = ArrayA[i];
		u32 B = ArrayB[i];
		u32 Expected = A + B;
		u32 Actual = ResultArray[i];
		if (Expected == Actual) {
			printf(ANSI_COLOR_GREEN "%.2u + %.2u = %u\n", A, B, Actual);
		} else {
			printf(ANSI_COLOR_RED "%.2u + %.2u = %u\n", A, B, Actual);
		}
	}
#endif

	printf(ANSI_COLOR_RESET);
}
