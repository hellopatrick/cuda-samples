#include <png.h>
#include <vector_types.h>

#define PNG_SUCCESS 1<<8
#define PNG_FAILURE -1

typedef struct {
	int width, height;
	int bit_depth, color_type;
	int filter_method, compression_type, interlace_type;
	size_t bytes_per_row;
} png_t;

int read_png(char *file_name, png_t **info, uchar4 **pixels);
int write_png(char* file_name, png_t *info, uchar4 *pixels);