float4 computeForce(float4 ipos, float4 jpos)
{
  float4 d       = jpos - ipos;
         d.w     = 0;
  float  distSq  = d.x*d.x + d.y*d.y + d.z*d.z + softening*softening;
  float  invdist = native_rsqrt(distSq);
  float  coeff   = jpos.w * (invdist*invdist*invdist);
  return coeff * d;
}

kernel void nbody(global const float4 * restrict positionsIn,
                  global       float4 * restrict positionsOut,
                  global       float4 * restrict velocities,
                  const        uint              numBodies)
{
  uint i       = get_global_id(0);
  uint lid     = get_local_id(0);
  float4 ipos  = positionsIn[i];

  local float4 scratch[WGSIZE];

  // Compute force
  float4 force = 0.f;
  for (uint j = 0; j < numBodies; j+=WGSIZE)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    scratch[lid] = positionsIn[j + lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint k = 0; k < WGSIZE;)
    {
      force += computeForce(ipos, scratch[k++]);
#if UNROLL_FACTOR >= 2
      force += computeForce(ipos, scratch[k++]);
#endif
#if UNROLL_FACTOR >= 4
      force += computeForce(ipos, scratch[k++]);
      force += computeForce(ipos, scratch[k++]);
#endif
#if UNROLL_FACTOR >= 8
      force += computeForce(ipos, scratch[k++]);
      force += computeForce(ipos, scratch[k++]);
      force += computeForce(ipos, scratch[k++]);
      force += computeForce(ipos, scratch[k++]);
#endif
    }
  }

  // Update velocity
  float4 velocity = velocities[i];
  velocity       += force * delta;
  velocities[i]   = velocity;

  // Update position
  positionsOut[i] = ipos + velocity * delta;
}

kernel void fillTexture(write_only image2d_t texture)
{
  uint x      = get_global_id(0);
  uint y      = get_global_id(1);
  float4 fill = (float4)(0.f,0.f,0.1f,1.f);
  write_imagef(texture, (int2)(x, y), fill);
}

kernel void drawPositions(global const float4 * restrict positions,
                          write_only image2d_t texture,
                          const uint width, const uint height)
{
  uint i        = get_global_id(0);
  float4 ipos   = positions[i];
  int2   coord  = (int2)(ipos.x+(width/2), ipos.y+(height/2));
  float4 white  = (float4)(1.f, 1.f, 1.f, 1.f);
  float4 yellow = (float4)(1.f, 1.f, 0.f, 1.f);

  if (coord.x >= 2 && coord.x < (width-2) &&
      coord.y >= 2 && coord.y < (height-2))
  {
    write_imagef(texture, coord,               white);
    write_imagef(texture, coord+(int2)( 0, 1), yellow);
    write_imagef(texture, coord+(int2)( 0,-1), yellow);
    write_imagef(texture, coord+(int2)( 1, 0), yellow);
    write_imagef(texture, coord+(int2)(-1, 0), yellow);
  }
}
