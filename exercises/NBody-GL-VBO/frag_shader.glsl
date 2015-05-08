#version 120

void main(void)
{
  float dist = distance(vec2(0.5, 0.5), gl_PointCoord);
  if (dist > 0.5)
    discard;

  float intensity = (1.0 - (dist * 2.0)) * 0.6;

  gl_FragColor = vec4(intensity, intensity, intensity*0.6, 1.0);
}
