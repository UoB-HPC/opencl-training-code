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
