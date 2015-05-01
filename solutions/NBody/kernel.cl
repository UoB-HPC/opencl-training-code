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
