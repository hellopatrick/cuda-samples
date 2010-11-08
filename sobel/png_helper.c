#include "png_helper.h"
#include <stdio.h>
#include <stdlib.h>

int read_png(char *file_name, png_t **info, uchar4 **pixels) {
	char header[8];	// 8 is the maximum size that can be checked
	
	// open file. if error, return error.
	FILE *fp = fopen(file_name, "rb");
	if (!fp) { return PNG_FAILURE; }
	
	// read the header.
	fread(header, 1, 8, fp);
	// verify that this is png. if not, return error.
	if (png_sig_cmp((png_byte *) header, 0, 8)) { return PNG_FAILURE; }
	
	// libpng struct for reading. if not created, return error
	png_struct *png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png_ptr) { return PNG_FAILURE;}
	
	// libpng struct for information. if not created, return error
	png_info *info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr) { return PNG_FAILURE; }
	
	// init libpng with file and tell it that we have already read first 8 bytes.
	png_init_io(png_ptr, fp);
	png_set_sig_bytes(png_ptr, 8);
	// get information.
	png_read_info(png_ptr, info_ptr);
	
	// allocate our copy of the png info.
	*info = malloc(sizeof(png_t));
	// libpng function for reading information from info_ptr
	png_get_IHDR(png_ptr, info_ptr, (png_uint_32 *)&((*info)->width), (png_uint_32 *)&((*info)->height), &((*info)->bit_depth), &((*info)->color_type), &((*info)->interlace_type), &((*info)->compression_type), &((*info)->filter_method));
	
	// we want 8-bit RGBA color.
	if ((*info)->bit_depth == 16) { png_set_strip_16(png_ptr); }	
	if ((*info)->bit_depth < 8) { png_set_packing(png_ptr); }	
	if ((*info)->color_type == PNG_COLOR_TYPE_GRAY || (*info)->color_type == PNG_COLOR_TYPE_GRAY_ALPHA) { png_set_gray_to_rgb(png_ptr); }
	
	if ((*info)->color_type == PNG_COLOR_TYPE_RGB || (*info)->color_type == PNG_COLOR_TYPE_GRAY) {
		png_set_add_alpha(png_ptr, 0xFF, PNG_FILLER_AFTER);
	}
	
	// updates info if we changed anything from above.
	png_read_update_info(png_ptr, info_ptr);
	png_get_IHDR(png_ptr, info_ptr, (png_uint_32 *)&((*info)->width), (png_uint_32 *)&((*info)->height), &((*info)->bit_depth), &((*info)->color_type), &((*info)->interlace_type), &((*info)->compression_type), &((*info)->filter_method));
	
	// allocating our memory for the pixels
	size_t bytes_per_row = png_get_rowbytes(png_ptr, info_ptr);
	(*info)->bytes_per_row = bytes_per_row;
	*pixels = malloc(bytes_per_row * (*info)->height);
	
	// libpng works by row pointers & byte pointers. we will instead have all the pixels in a block of memory.
	// we will 'trick' libpng by iterating through our pointers. reading one row at a time.
	int i;
	png_bytep row = (png_bytep) (*pixels);
	
	for(i = 0; i < (*info)->height; i++) {
		png_read_row(png_ptr, row, NULL);
		row += bytes_per_row;
	}
	
	// finish reading.
	png_read_end(png_ptr, info_ptr);
	png_destroy_read_struct(&png_ptr, &info_ptr, NULL);

	fclose(fp);
	return PNG_SUCCESS;
}

int write_png(char* file_name, png_t *png, uchar4 *pixels) {
	// open file that we will write to.
	FILE *fp = fopen(file_name, "wb");
	if (!fp) { return PNG_FAILURE; }

	// create write and info structs.
	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png_ptr) { return PNG_FAILURE; }
	
	png_infop info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr) { return PNG_FAILURE; }
	
	// initialize libpng
	png_init_io(png_ptr, fp);
	png_set_IHDR(png_ptr, info_ptr, png->width, png->height, 8, 6, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
	png_write_info(png_ptr, info_ptr);

	// just like with read, we will write the file one row at a time.
	size_t bytes_per_row = png_get_rowbytes(png_ptr, info_ptr);

	int i;
	png_bytep row = (png_bytep) (pixels);
	
	for(i = 0; i < png->height; i++) {
		png_write_row(png_ptr, row);
		row += bytes_per_row;
	}
	
	// finish write & close.
	png_write_end(png_ptr, info_ptr);
	png_destroy_write_struct(&png_ptr, &info_ptr);
	
	fclose(fp);
	return PNG_SUCCESS;
}