__kernel void vector_mult(
	__global float *result,
	__global float *dataa,
	__global float *datab,
	const unsigned int count)
{
	int i = get_global_id(0);

	if (i < count)
		result[i] = dataa[i] * datab[i];
}

__kernel void matrix_mult(
	__global float *result, __global float *mata, 
	__global float *matb, const unsigned int aRows, 
	const unsigned int abDim, const unsigned int bCols)
{
	// row of the output
	unsigned int r = get_global_id(0);
	// column of the output
	unsigned int c = get_global_id(1);
	unsigned int h, aind, bind;

	float curcell = 0;
	
	if (r < aRows && c < bCols) {
		for (h = 0; h < abDim; h++) {
			aind = r * aRows + h;
			bind = h * abDim + c;
			curcell += mata[aind] * matb[bind];
		}
		result[r * aRows + c] = curcell;
	}
}
