/*
 *
 * This code is released under the "attribution CC BY" creative commons license.
 * In other words, you can use it in any way you see fit, including commercially,
 * but please retain an attribution for the original authors:
 * the High Performance Computing Group at the University of Bristol.
 * Contributors include Simon McIntosh-Smith, James Price, Tom Deakin and Mike O'Connor.
 *
 */

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

#ifdef USE_LOCAL
  local float4 scratch[WGSIZE];
#endif

  // Compute force
  float4 force = 0.f;
  for (uint j = 0; j < numBodies; j+=WGSIZE)
  {
#ifdef USE_LOCAL
    barrier(CLK_LOCAL_MEM_FENCE);
    scratch[lid] = positionsIn[j + lid];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    for (uint k = 0; k < WGSIZE; k++)
    {
#ifdef USE_LOCAL
      force += computeForce(ipos, scratch[k]);
#else
      force += computeForce(ipos, positionsIn[j + k]);
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
