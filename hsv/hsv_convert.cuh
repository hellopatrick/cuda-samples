__device__ uchar4 convert_one_pixel_to_rgb(float4 pixel) {
	float r, g, b;
	float h, s, v;
	
	h = pixel.x;
	s = pixel.y;
	v = pixel.z;
	
	float f = h/60.0f;
	float hi = floorf(f);
	f = f - hi;
	float p = v * (1 - s);
	float q = v * (1 - s * f);
	float t = v * (1 - s * (1 - f));
	
	if(hi == 0.0f || hi == 6.0f) {
		r = v;
		g = t;
		b = p;
	} else if(hi == 1.0f) {
		r = q;
		g = v;
		b = p;
	} else if(hi == 2.0f) {
		r = p;
		g = v;
		b = t;
	} else if(hi == 3.0f) {
		r = p;
		g = q;
		b = v;
	} else if(hi == 4.0f) {
		r = t;
		g = p;
		b = v;
	} else {
		r = v;
		g = p;
		b = q;
	}
	
	unsigned char red = (unsigned char) __float2uint_rn(255.0f * r);
	unsigned char green = (unsigned char) __float2uint_rn(255.0f * g);
	unsigned char blue = (unsigned char) __float2uint_rn(255.0f * b);
	unsigned char alpha = (unsigned char) __float2uint_rn(pixel.w);
	return (uchar4) {red, green, blue, alpha};
}

__device__ float4 convert_one_pixel_to_hsv(uchar4 pixel) {
	float r, g, b, a;
	float h, s, v;
	
	r = pixel.x / 255.0f;
	g = pixel.y / 255.0f;
	b = pixel.z / 255.0f;
	a = pixel.w;
	
	float max = fmax(r, fmax(g, b));
	float min = fmin(r, fmin(g, b));
	float diff = max - min;
	
	v = max;
	
	if(v == 0.0f) { // black
		h = s = 0.0f;
	} else {
		s = diff / v;
		if(diff < 0.001f) { // grey
			h = 0.0f;
		} else { // color
			if(max == r) {
				h = 60.0f * (g - b)/diff;
				if(h < 0.0f) { h += 360.0f; }
			} else if(max == g) {
				h = 60.0f * (2 + (b - r)/diff);
			} else {
				h = 60.0f * (4 + (r - g)/diff);
			}
		}		
	}
	
	return (float4) {h, s, v, a};
}