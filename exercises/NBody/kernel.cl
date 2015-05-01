#define UNROLL_FACTOR 1

float4 computeForce(float4 ipos, float4 jpos, float softening)
{
  float4 d      = jpos - ipos;
         d.w    = 0;
  float  distSq = d.x*d.x + d.y*d.y + d.z*d.z + softening*softening;
  float  dist   = sqrt(distSq);
  float  coeff  = jpos.w / (dist*dist*dist);
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
  float4 ipos  = positionsIn[i];

  // Compute force
  float4 force = 0.f;
  for (uint j = 0; j < numBodies;)
  {
    force += computeForce(ipos, positionsIn[j++], softening);
#if UNROLL_FACTOR >= 2
    force += computeForce(ipos, positionsIn[j++], softening);
#endif
#if UNROLL_FACTOR >= 4
    force += computeForce(ipos, positionsIn[j++], softening);
    force += computeForce(ipos, positionsIn[j++], softening);
#endif
#if UNROLL_FACTOR >= 8
    force += computeForce(ipos, positionsIn[j++], softening);
    force += computeForce(ipos, positionsIn[j++], softening);
    force += computeForce(ipos, positionsIn[j++], softening);
    force += computeForce(ipos, positionsIn[j++], softening);
#endif
  }

  // Update velocity
  float4 velocity = velocities[i];
  velocity       += force * delta;
  velocities[i]   = velocity;

  // Update position
  positionsOut[i] = ipos + velocity * delta;
}
