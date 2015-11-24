/*
 *
 * This code is released under the "attribution CC BY" creative commons license.
 * In other words, you can use it in any way you see fit, including commercially,
 * but please retain an attribution for the original authors:
 * the High Performance Computing Group at the University of Bristol.
 * Contributors include Simon McIntosh-Smith, James Price, Tom Deakin and Mike O'Connor.
 *
 */

kernel void bilateral(global const uchar4 *input,
                      global       uchar4 *output,
                      const float sigmaDomain, const float sigmaRange,
                      const int radius)
{
  int y      = get_global_id(0);
  int x      = get_global_id(1);
  int height = get_global_size(0);
  int width  = get_global_size(1);

  float  coeff  = 0.f;
  float4 sum    = 0.f;
  float4 center = convert_float4(input[x + y*width])/255.f;

  for (int j = -radius; j <= radius; j++)
  {
    for (int i = -radius; i <= radius; i++)
    {
      int xi = clamp(x+i, 0, width-1);
      int yj = clamp(y+j, 0, height-1);

      float norm, weight;
      float4 pixel = convert_float4(input[xi + yj*width])/255.f;

      norm    = sqrt((float)(i*i) + (float)(j*j)) * (1.f/sigmaDomain);
      weight  = exp(-0.5f * (norm*norm));

      norm    = distance(pixel.xyz, center.xyz) * (1.f/sigmaRange);
      weight *= exp(-0.5f * (norm*norm));

      coeff += weight;
      sum   += weight*pixel;
    }
  }

  sum   /= coeff;
  sum.w  = center.w;

  output[x + y*width] = convert_uchar4(sum*255.f);
}