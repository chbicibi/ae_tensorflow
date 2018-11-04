import cv2, glob, os, re, sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as anim

plt.rcParams["animation.ffmpeg_path"] = "/path/to/ffmpeg.exe"
FFMpegWriter = anim.writers['ffmpeg']


def show_anim(fig, update, frames=1000, init_func=lambda:None, interval=8, file=None, fps=100):
  ani = anim.FuncAnimation(fig, update, frames=frames, init_func=init_func, interval=interval)
  if file:
    ani.save(file, writer=FFMpegWriter(fps=fps))
  else:
    plt.show()


def create_anim(imgs, print_f, frames, file=None):
  """
  アニメーション出力 (1段)
  """
  fig, ax = plt.subplots(1, 1)

  def with_p_index(gen, i):
    gen.__next__()
    print(f"export: {i+1} / {frames}", end="\r")
    # sys.stdout.write(f"\rexport: {i+1} / {frames}")
    # sys.stdout.flush()

  def init_func():
    ax.tick_params(labelbottom=False, bottom=False) # x軸の削除
    ax.tick_params(labelleft=False, left=False) # y軸の削除
    # plt.colorbar() # カラーバー

  def update():
    def gen_f():
      for img in imgs:
        ax.cla()
        # fig.clf()
        # ax = fig.add_subplot(111)
        print_f(ax.imshow, img, fig)
        # fig.colorbar(im)
        # plt.imshow(x[:, :, idx], cmap='gray', vmin=-1, vmax=1)
        yield
    return lambda i: with_p_index(gen_f(), i)

  show_anim(fig, update(), frames=frames, init_func=init_func, interval=8, file=file, fps=100)


def create_anim_2row(gs, idx, file):
  """
  アニメーション出力
  """
  fig, (ax1, ax2) = plt.subplots(2, 1)

  for ax in [ax1, ax2]:
    ax.tick_params(labelbottom=False, bottom=False) # x軸の削除
    ax.tick_params(labelleft=False, left=False) # y軸の削除

  def clear_ax():
    for ax in [ax1, ax2]:
      ax.cla()

  def update():
    def _():
      yield
      for g in gs:
        for x, y in zip(*g):
          clear_ax()
          ax1.imshow(x[:, :, idx], cmap='gray', vmin=-1, vmax=2)
          ax2.imshow(y[:, :, idx], cmap='gray', vmin=-1, vmax=2)
          yield
    gen = _()
    return lambda i: print(i) or gen.__next__()

  show_anim(fig, update(), interval=8, frames=1000, fps=100)


################################################################################

def test_check_output(data, output, feed_dict_f):
  x_g = (cl.load_naca0012("a0", 1000 + 100 * i, 100) for i in range(10))
  # y_g = (sess.run(output, feed_dict=feed_dict_f(x)) for x in x_g)
  z_g = ((x, sess.run(output, feed_dict=feed_dict_f(x))) for x in x_g)
  # show_anim_f(z_g, "a0_output_xy_u.mp4", 0)
  show_anim_f(z_g, "a0_output_xy_v.mp4", 1)


def show(img):
  plt.imshow(img)
  plt.show()

def extract_u(d):
  """
  入力: (H, W, C=2)
  """
  return d[:, :, 0]

def extract_omg(d):
  np.convolve([1,2,3],[0,1,0.5], 'same')

def test_plotimg():
  """
  直接呼ぶ
  """
  from . import cfd_loader as cl

  data = cl.load_cfd_data2("a4_1000-1999of2000")
  for i, d in enumerate(data):
    if i % 1 == 0:
      plt.imshow(data)
    plt.show()
    break

if __name__ == '__main__':
  test_plotimg()
