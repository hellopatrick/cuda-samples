#include <cuda.h>
#include <cuda_runtime_api.h>

#include "dimensions.h"

/*
	Any live cell with fewer than two live neighbours dies, as if caused by under-population.
	Any live cell with more than three live neighbours dies, as if by overcrowding.
	Any live cell with two or three live neighbours lives on to the next generation.
	Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction
*/

#define CACHE_LEFT_OKAY (local_x > 0)
#define CACHE_RIGHT_OKAY (local_x < 15)
#define CACHE_ABOVE_OKAY (local_y > 0)
#define CACHE_BELOW_OKAY (local_y < 15)

#define LEFT_OKAY (x > 0)
#define RIGHT_OKAY (x < (DIM_X - 1))
#define ABOVE_OKAY (y > 0)
#define BELOW_OKAY (y < (DIM_Y - 1))