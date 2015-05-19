/*
 *
 * This code is released under the "attribution CC BY" creative commons license.
 * In other words, you can use it in any way you see fit, including commercially,
 * but please retain an attribution for the original authors:
 * the High Performance Computing Group at the University of Bristol.
 * Contributors include Simon McIntosh-Smith, James Price, Tom Deakin and Mike O'Connor.
 *
 */

const sampler_t sampler =
  CLK_NORMALIZED_COORDS_FALSE |
  CLK_ADDRESS_CLAMP_TO_EDGE   |
  CLK_FILTER_NEAREST;

kernel void bilateral(read_only image2d_t input,
                      write_only image2d_t output,
                      const float sigmaDomain, const float sigmaRange)
{
  int x = get_global_id(0);
  int y = get_global_id(1);

  float coeff = 0.f;
  float4 sum = 0.f;
  float4 center = read_imagef(input, sampler, (int2)(x, y));

  for (int j = -2; j <= 2; j++)
  {
    for (int i = -2; i <= 2; i++)
    {
      float norm, weight;
      float4 pixel = read_imagef(input, sampler, (int2)(x+i, y+j));

      norm = native_sqrt((float)(i*i) + (float)(j*j)) * native_recip(sigmaDomain);
      weight = native_exp(-0.5f * (norm*norm));

      norm = fast_distance(pixel.xyz, center.xyz) * native_recip(sigmaRange);
      weight *= native_exp(-0.5f * (norm*norm));

      coeff += weight;
      sum += weight*pixel;
    }
  }

  sum /= coeff;
  sum.w = center.w;

  write_imagef(output, (int2)(x, y), sum);
}