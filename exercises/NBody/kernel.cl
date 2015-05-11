float4 computeForce(float4 ipos, float4 jpos, float softening)
{
  float4 d       = jpos - ipos;
         d.w     = 0;
  float  distSq  = d.x*d.x + d.y*d.y + d.z*d.z + softening*softening;
  float  invdist = native_rsqrt(distSq);
  float  coeff   = jpos.w * (invdist*invdist*invdist);
  return coeff * d;
}

kernel void nbody(global float4 *positionsIn,
                  global float4 *positionsOut,
                  global float4 *velocities,
                  const  uint    numBodies,
                  const  float   delta,
                  const  float   softening)
{
  uint i       = get_global_id(0);
  uint lid     = get_local_id(0);
  float4 ipos  = positionsIn[i];

  uint wgsize = get_local_size(0);

  // Allocated the most local memory we might need
  // Would be better if the compiler knew the work-group size...
  local float4 scratch[1024];

  // Compute force
  float4 force = 0.f;
  for (uint j = 0; j < numBodies; j+=wgsize)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    scratch[lid] = positionsIn[j + lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint k = 0; k < wgsize; k++)
    {
      force += computeForce(ipos, scratch[k], softening);
    }
  }

  // Update velocity
  float4 velocity = velocities[i];
  velocity       += force * delta;
  velocities[i]   = velocity;

  // Update position
  positionsOut[i] = ipos + velocity * delta;
}
