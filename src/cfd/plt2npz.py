import cv2, glob, os, re, sys
import numpy as np
from matplotlib import pyplot as plt

from myutils import chdir


def read_plt(path, extract):
  """
  読み込む物理量を書き換える
  現在は u, v
  """
  n_ch = 4

  print("reading: " + path, end="\r")
  with open(path, "r") as file:
    file.readline()
    line   = file.readline()
    nx, ny = (int(s) for s in re.findall(r"\d+", line))
    image  = np.empty((ny, nx, n_ch), np.float32)

    for j in range(ny):
      for i in range(nx):
        line = file.readline()
        image[j, i, :] = extract(*(float(s) for s in line.split()))

  return image

def save_cfd_data_all():
  """
  naca0012のデータを書き込み
  *直接呼ぶ
  """
  def selector(d):
    return os.path.isdir(d) and glob.glob(f"{d}/out_*.plt")

  def extract(x, y, u, v, p, f, c):
    return [u, v, p, c]

  cfd_dir = "/path/to/cfd"

  file_fmt = lambda i: "out_%04d.plt" % i
  size = 100
  step = 1

  print("save_cfd_data_all: start")

  with chdir(cfd_dir):
    data_dirs = (x for x in glob.glob("*") if selector(x))

    for data_dir in data_dirs:
      files = glob.glob(f"{data_dir}/out_*.plt")
      datasize = len(files) // 100 * 100
      print(f"dir: {data_dir} data: {datasize}")
      begins = (size * i for i in range(datasize // size))

      for begin in begins:
        with chdir(data_dir):
          shape = "{0}x{1}".format(*read_plt(file_fmt(0), extract).shape)
          datafile = f"{data_dir}_uvpc_{shape}_{begin:0>4}-{begin+size:0>4}of{datasize:0>4}.npz"
          if os.path.isfile(datafile):
            continue
          data = [read_plt(file_fmt(i), extract) for i in range(begin, begin + size, step)]
          np.savez_compressed(datafile, data)

  print("save_cfd_data_all: done")

if __name__ == '__main__':
  save_cfd_data_all()
