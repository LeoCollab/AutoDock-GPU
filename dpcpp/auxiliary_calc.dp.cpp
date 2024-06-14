#define invpi2 1.0f/(PI_TIMES_2)

// Magic positive integer exponent power ... -AT
__dpct_inline__
float positive_power(
	float a,
	uint exp)
{
	float result=(exp & 1)?a:1.0f;
	while(exp>>=1)
	{
		a *= a;
		result=(exp & 1)?result*a:result;
	}
	return result;
}

SYCL_EXTERNAL
__dpct_inline__
float fmod_pi2(float x)
{
	return x-(int)(invpi2*x)*PI_TIMES_2;
}

#define fast_acos_a  9.78056e-05f
#define fast_acos_b -0.00104588f
#define fast_acos_c  0.00418716f
#define fast_acos_d -0.00314347f
#define fast_acos_e  2.74084f
#define fast_acos_f  0.370388f
#define fast_acos_o -(fast_acos_a+fast_acos_b+fast_acos_c+fast_acos_d)

SYCL_EXTERNAL
__dpct_inline__
float fast_acos(float cosine)
{
	float x = sycl::fabs(cosine);
	float x2=x*x;
	float x3=x2*x;
	float x4=x3*x;
	float ac =
			(((fast_acos_o * x4 + fast_acos_a) * x3 + fast_acos_b) * x2 + fast_acos_c) *
			x +
            fast_acos_d +
            fast_acos_e * SYCL_SQRT(2.0f - SYCL_SQRT(2.0f + 2.0f * x)) -
			fast_acos_f * SYCL_SQRT(2.0f - 2.0f * x);
	return sycl::copysign(ac, cosine) + (cosine < 0.0f) * PI_FLOAT;
}

SYCL_EXTERNAL
__dpct_inline__
sycl::float4 cross(
	sycl::float3 &u,
	sycl::float3 &v)
{
	sycl::float4 result;
	result.x() = u.y() * v.z() - v.y() * u.z();
	result.y() = v.x() * u.z() - u.x() * v.z();
	result.z() = u.x() * v.y() - v.x() * u.y();
	result.w() = 0.0f;
	return result;
}

__dpct_inline__
sycl::float4 cross(
	sycl::float4 &u,
	sycl::float4 &v)
{
	sycl::float4 result;
	result.x() = u.y() * v.z() - v.y() * u.z();
	result.y() = v.x() * u.z() - u.x() * v.z();
	result.z() = u.x() * v.y() - v.x() * u.y();
	result.w() = 0.0f;
	return result;
}

SYCL_EXTERNAL
__dpct_inline__
sycl::float4 quaternion_multiply(
	sycl::float4 a,
	sycl::float4 b)
{
	sycl::float4 result = {
		a.w() * b.x() + a.x() * b.w() + a.y() * b.z() - a.z() * b.y(),  // x
		a.w() * b.y() - a.x() * b.z() + a.y() * b.w() + a.z() * b.x(),  // y
		a.w() * b.z() + a.x() * b.y() - a.y() * b.x() + a.z() * b.w(),  // z
		a.w() * b.w() - a.x() * b.x() - a.y() * b.y() - a.z() * b.z()	// w
	};
	return result;
}

SYCL_EXTERNAL
__dpct_inline__
sycl::float4 quaternion_rotate(
	sycl::float4 v,
	sycl::float4 rot)
{
	sycl::float4 result;
	sycl::float4 z = cross(rot, v);
	z.x() *= 2.0f;
	z.y() *= 2.0f;
	z.z() *= 2.0f;
	sycl::float4 c = cross(rot, z);
	result.x() = v.x() + z.x() * rot.w() + c.x();
	result.y() = v.y() + z.y() * rot.w() + c.y();
	result.z() = v.z() + z.z() * rot.w() + c.z();
	result.w() = 0.0f;
	return result;
}

