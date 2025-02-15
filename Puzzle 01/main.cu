#include "..\base.h"

#include <stdio.h>

static u32 InitialArray[16];
static u32 ResultArray[16];

void Init() {
	for (u32 i = 0; i < ARRAY_LEN(InitialArray); ++i) {
		InitialArray[i] = i;
	}
}

__global__ void Add10(u32 *GPUMemory) {
	u32 Index = threadIdx.x;
	printf("%d\n", Index);
	GPUMemory[Index] += 10;
}

s32 main() {
	Init();

	u32 *GPUMemory = 0;
	u32 SizeInBytes = sizeof(InitialArray);
	cudaMalloc(&GPUMemory, SizeInBytes);
	cudaMemcpy(GPUMemory, InitialArray, SizeInBytes, cudaMemcpyHostToDevice);

	Add10<<<1, 16>>>(GPUMemory);

	cudaMemcpy(ResultArray, GPUMemory, SizeInBytes, cudaMemcpyDeviceToHost);

	for (u32 i = 0; i < ARRAY_LEN(InitialArray); ++i) {
		u32 Expected = InitialArray[i] + 10;
		u32 Actual = ResultArray[i];
		if (Expected == Actual) {
			printf(ANSI_COLOR_GREEN "%u + 10 = %u\n", InitialArray[i], Actual);
		} else {
			printf(ANSI_COLOR_RED "%u + 10 = %u\n", InitialArray[i], Actual);
		}
	}
	printf(ANSI_COLOR_RESET);

	return 0;
}
