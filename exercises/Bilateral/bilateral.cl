kernel void bilateral(global const uchar4 *input,
                      global       uchar4 *output,
                      const float sigmaDomain, const float sigmaRange)
{
  int x      = get_global_id(0);
  int y      = get_global_id(1);
  int width  = get_global_size(0);
  int height = get_global_size(1);

  float coeff = 0.f;
  float4 sum = 0.f;
  float4 center = convert_float4(input[x + y*width])/255.f;

  for (int j = -2; j <= 2; j++)
  {
    for (int i = -2; i <= 2; i++)
    {
      int _x = clamp(x+i, 0, width-1);
      int _y = clamp(y+j, 0, height-1);

      float norm, weight;
      float4 pixel = convert_float4(input[_x + _y*width])/255.f;

      norm = sqrt((float)(i*i) + (float)(j*j)) * (1.f/sigmaDomain);
      weight = native_exp(-0.5f * (norm*norm));

      norm = fast_distance(pixel.xyz, center.xyz) * (1.f/sigmaRange);
      weight *= native_exp(-0.5f * (norm*norm));

      coeff += weight;
      sum += weight*pixel;
    }
  }

  sum /= coeff;
  sum.w = center.w;

  output[x + y*width] = convert_uchar4(sum*255.f);
}