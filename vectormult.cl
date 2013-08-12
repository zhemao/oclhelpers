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
