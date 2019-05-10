/*
 *
 * This code is released under the "attribution CC BY" creative commons license.
 * In other words, you can use it in any way you see fit, including commercially,
 * but please retain an attribution for the original authors:
 * the High Performance Computing Group at the University of Bristol.
 * Contributors include Simon McIntosh-Smith, James Price, Tom Deakin and Mike O'Connor.
 *
 */

#version 120

attribute vec4  positions;
uniform   mat4  vpMatrix;
uniform   vec3  eyePosition;
uniform   float pointScale;
uniform   float sightRange;

void main(void)
{
  gl_Position  = vpMatrix * positions;

  float dist   = distance(positions.xyz, eyePosition);
  float size   = pointScale * (1.0 - (dist / sightRange));
  gl_PointSize = max(size, 0.0);
}
