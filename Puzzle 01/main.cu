#include "..\base.h"

#include <stdio.h>

static random_state RandomState = { 0xB40148552A2E3491 };
static u32 RandomIntegersInitialState[16];
static u32 RandomIntegers[16];

void InitRandomIntegers() {
	for (u32 i = 0; i < ARRAY_LEN(RandomIntegers); ++i) {
		u32 Random = RandomInt(&RandomState) & 0xF;
		RandomIntegers[i] = Random;
		RandomIntegersInitialState[i] = Random;
	}
}

__global__ void Add10(u32 *GPUMemory, u32 Length) {
	u32 Index = threadIdx.x;
	GPUMemory[Index] += 10;
}

s32 main() {
	InitRandomIntegers();

	for (u32 i = 0; i < ARRAY_LEN(RandomIntegers); ++i) {
		printf("%u\n", RandomIntegers[i]);
	}
	puts("===");

	u32 *GPUMemory = 0;
	cudaMalloc(&GPUMemory, sizeof(RandomIntegers));
	cudaMemcpy(GPUMemory, RandomIntegers, sizeof(RandomIntegers), cudaMemcpyHostToDevice);

	Add10<<<1, 16>>>(GPUMemory, ARRAY_LEN(RandomIntegers));

	cudaMemcpy(RandomIntegers, GPUMemory, sizeof(RandomIntegers), cudaMemcpyDeviceToHost);

	puts("===");
	for (u32 i = 0; i < ARRAY_LEN(RandomIntegers); ++i) {
		u32 Initial = RandomIntegersInitialState[i];
		u32 Result = RandomIntegers[i];
		if (RandomIntegers[i] == RandomIntegersInitialState[i] + 10) {
			printf(ANSI_COLOR_GREEN "%u + 10 = %u\n" ANSI_COLOR_RESET, Initial, Result);
		} else {
			printf(ANSI_COLOR_RED "%u + 10 = %u\n" ANSI_COLOR_RESET, Initial, Result);
		}
	}

	return 0;
}
