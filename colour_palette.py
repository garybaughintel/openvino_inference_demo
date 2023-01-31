import random
import colorsys
import numpy as np

def dist(c1, c2):
        dh = min(abs(c1[0] - c2[0]), 1 - abs(c1[0] - c2[0])) * 2
        ds = abs(c1[1] - c2[1])
        dv = abs(c1[2] - c2[2])
        return dh * dh + ds * ds + dv * dv

def min_distance(colors_set, color_candidate):
    distances = [dist(o, color_candidate) for o in colors_set]
    return np.min(distances)


def hsv2rgb(h, s, v):
    return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))

def get_colour_palette(n) :
  rng = random.Random(0xACE)  # nosec - disable B311:random check
  candidates_num = 100
  hsv_colors = [(1.0, 1.0, 1.0)]
  for _ in range(1, n):
      colors_candidates = [(rng.random(), rng.uniform(0.8, 1.0), rng.uniform(0.5, 1.0))
                            for _ in range(candidates_num)]
      min_distances = [min_distance(hsv_colors, c) for c in colors_candidates]
      arg_max = np.argmax(min_distances)
      hsv_colors.append(colors_candidates[arg_max])

  palette = [hsv2rgb(*hsv) for hsv in hsv_colors]
  return palette