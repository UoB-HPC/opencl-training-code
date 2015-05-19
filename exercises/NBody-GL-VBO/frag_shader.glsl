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

void main(void)
{
  float dist = distance(vec2(0.5, 0.5), gl_PointCoord);
  if (dist > 0.5)
    discard;

  float intensity = (1.0 - (dist * 2.0)) * 0.6;

  gl_FragColor = vec4(intensity, intensity, intensity*0.6, 1.0);
}
