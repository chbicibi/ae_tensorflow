import cv2, glob, os, re, sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mp_colors
import cfd_loader as cl
import cfd_visualizer as cv


def resize_image(input):
  """
  input (720, 1280, 4)
  """
  src = input[:, :, 3]
  dst = cv2.resize(src, (src.shape[1]//2, src.shape[0]//2), interpolation=cv2.INTER_LINEAR)
  return dst


def resize_data():
  """
  TEMP
  """
  init, bound = True, None
  size = 100
  total = 2000
  count = total // size

  def get_f(i):
    file = f"naca2412_1_c_360x640_{size*i:0>4}-{size*(i+1):0>4}of{total:0>4}"
    return cl.load_cfd_data(file)

  def gen_f():
    return (d for i in range(count) for d in get_f(i))
    # for i in range(count):
    #   for d in get_f(i):
    #     yield d

  def plot_f(plot_f, img, fig):
    nonlocal init, bound
    if init:
      bound = max(abs(img.min()), abs(img.max())) * 1.2
      im = plot_f(img, cmap=cmap, vmin=-bound, vmax=bound)
      # im.set_clim(-100, 100)
      fig.colorbar(im)
      init = False
    else:
      im = plot_f(img, cmap=cmap, vmin=-bound, vmax=bound)
    return im

  color_list = [(0, "red"), (0.4, "red"), (0.5, "black"), (0.6, "green"), (1, "green")]
  cmap = mp_colors.LinearSegmentedColormap.from_list('custom_cmap', color_list)

  # gen = (get_f(i) for i in range(10))
  cv.create_anim(gen_f(), plot_f, total, "naca2412_c_RBkG_7.mp4")
  # cv.create_anim(gen, plot_f, 2000) # => 画面出力

def test_cylinder():
  file = f"cylinder_uvpc_0000-0199of1000"
  data = cl.load_cfd_data(file)
  print(data.shape)

if __name__ == '__main__':
  test_cylinder()
