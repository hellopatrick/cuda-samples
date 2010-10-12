#include "png_helper.h"
#include <stdio.h>
#include <stdlib.h>

int read_png(char *file_name, png_t **info, uchar4 **pixels) {
	char header[8];	// 8 is the maximum size that can be checked

	FILE *fp = fopen(file_name, "rb");
	if (!fp) { return PNG_FAILURE; }
	
	fread(header, 1, 8, fp);
	if (png_sig_cmp((png_byte *) header, 0, 8)) { return PNG_FAILURE; }
	
	png_struct *png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	if (!png_ptr) { return PNG_FAILURE;}
	png_info *info_ptr = png_create_info_struct(png_ptr);
	
	if (!info_ptr) { return PNG_FAILURE; }

	png_init_io(png_ptr, fp);
	png_set_sig_bytes(png_ptr, 8);
	png_read_info(png_ptr, info_ptr);
	
	*info = malloc(sizeof(png_t));
	png_get_IHDR(png_ptr, info_ptr, (png_uint_32 *)&((*info)->width), (png_uint_32 *)&((*info)->height), &((*info)->bit_depth), &((*info)->color_type), &((*info)->interlace_type), &((*info)->compression_type), &((*info)->filter_method));
	
	if ((*info)->bit_depth == 16) { png_set_strip_16(png_ptr); }	
	if ((*info)->bit_depth < 8) { png_set_packing(png_ptr); }	
	if ((*info)->color_type == PNG_COLOR_TYPE_GRAY || (*info)->color_type == PNG_COLOR_TYPE_GRAY_ALPHA) { png_set_gray_to_rgb(png_ptr); }
	
	if ((*info)->color_type == PNG_COLOR_TYPE_RGB || (*info)->color_type == PNG_COLOR_TYPE_GRAY) {
		printf("Adding Alpha Channel...");
		png_set_add_alpha(png_ptr, 0xFF, PNG_FILLER_AFTER);
	}
	
	png_read_update_info(png_ptr, info_ptr);
	png_get_IHDR(png_ptr, info_ptr, (png_uint_32 *)&((*info)->width), (png_uint_32 *)&((*info)->height), &((*info)->bit_depth), &((*info)->color_type), &((*info)->interlace_type), &((*info)->compression_type), &((*info)->filter_method));
	
	size_t bytes_per_row = png_get_rowbytes(png_ptr, info_ptr);
	*pixels = malloc(bytes_per_row * (*info)->height);

	int i;
	png_bytep row = (png_bytep) (*pixels);
	
	for(i = 0; i < (*info)->height; i++) {
		png_read_row(png_ptr, row, NULL);
		row += bytes_per_row;
	}
	
	png_read_end(png_ptr, info_ptr);
	png_destroy_read_struct(&png_ptr, &info_ptr, NULL);

	fclose(fp);
	return PNG_SUCCESS;
}

int write_png(char* file_name, png_t *png, uchar4 *pixels) {
	FILE *fp = fopen(file_name, "wb");
	if (!fp) { return PNG_FAILURE; }

	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png_ptr) { return PNG_FAILURE; }

	png_infop info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr) { return PNG_FAILURE; }

	png_init_io(png_ptr, fp);
	png_set_IHDR(png_ptr, info_ptr, png->width, png->height, 8, 6, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
	png_write_info(png_ptr, info_ptr);

	size_t bytes_per_row = png_get_rowbytes(png_ptr, info_ptr);

	int i;
	png_bytep row = (png_bytep) (pixels);
	
	for(i = 0; i < png->height; i++) {
		png_write_row(png_ptr, row);
		row += bytes_per_row;
	}
	
	png_write_end(png_ptr, info_ptr);
	png_destroy_write_struct(&png_ptr, &info_ptr);
	
	fclose(fp);
	return PNG_SUCCESS;
}