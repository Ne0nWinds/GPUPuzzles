#include "..\base.h"

#include <stdio.h>

static random_state RandomState = { 0xB40148552A2E3491 };
#define ARRAY_SIZE 1024
static u32 InitialArray1[ARRAY_SIZE] = {0};
static u32 InitialArray2[ARRAY_SIZE] = {0};
static u32 ResultArray[ARRAY_SIZE] = {0};

void InitRandomIntegers() {
	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		InitialArray1[i] = RandomInt(&RandomState) & 0xFFFF;
	}
	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		InitialArray2[i] = RandomInt(&RandomState) & 0xFFFF;
	}
}

__global__ void Add(u32 *Array1, u32 *Array2, u32 Length) {
	u32 Index = threadIdx.x + blockIdx.x * blockDim.x;

	Array1[Index] = Array1[Index] + Array2[Index];
}

s32 main() {
	InitRandomIntegers();

	u32 *GPUArray1 = 0, *GPUArray2 = 0;
	cudaMalloc(&GPUArray1, ARRAY_SIZE * sizeof(int));
	cudaMalloc(&GPUArray2, ARRAY_SIZE * sizeof(int));
	cudaMemcpy(GPUArray1, InitialArray1, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(GPUArray2, InitialArray2, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);

	u32 ThreadSize = 32;
	u32 BlockSize = (ARRAY_SIZE + ThreadSize - 1) / ThreadSize;
	Add<<<BlockSize, ThreadSize>>>(GPUArray1, GPUArray2, ARRAY_SIZE);
	cudaMemcpy(ResultArray, GPUArray1, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	puts("===");

	for (u32 i = 0; i < ARRAY_SIZE; ++i) {
		u32 A = InitialArray1[i];
		u32 B = InitialArray2[i];
		u32 ExpectedResult = A + B;;
		u32 ActualResult = ResultArray[i];
		if (ExpectedResult == ActualResult) {
			printf(ANSI_COLOR_GREEN "%.5u + %.5u = %u\n" ANSI_COLOR_RESET, A, B, ActualResult);
		} else {
			printf(ANSI_COLOR_RED "%.5u + %.5u = %u\n" ANSI_COLOR_RESET, A, B, ActualResult);
		}
	}

}
