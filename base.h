#include <stdint.h>

typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef float f32;
typedef double f64;

typedef struct {
	u64 Seed;
} random_state;

static inline u32 RandomInt(random_state *State) {
	u64 OldSeed = State->Seed;
	State->Seed *= 0x4D7D3C53ULL;
	State->Seed += 0x65C3A6D5ULL;
	return OldSeed >> 32;
}

#define ARRAY_LEN(arr) (sizeof(arr) / sizeof(arr[0]))

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
