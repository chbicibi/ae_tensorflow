# 追加
# os.environ[ 'TF_CPP_MIN_LOG_LEVEL' ] = '2'
# sys.path.append('../..')

import cv2, glob, os, re, sys
import numpy as np
from matplotlib import pyplot as plt

from myutils import chdir, stopwatch

# ffmpeg = "/path/to/ffmpeg.exe"
# import glob, re, shutil

def clamp255(c):
  return min(255, max(0, int(255 * c)))

def convert_image_cv(array, grid):
  """
  テスト
  """
  ny = array.shape[0]
  nx = array.shape[1]
  image  = np.empty((ny, nx, 3), np.uint8)
  for j, col in enumerate(array):
    for i, pixel in enumerate(col):
      # u, v, f = pixel
      # o = u - col[i - 1, 0] + v - array[j - 1, i, 1] if i > 0 and j > 0 else 0
      # u, v, f, o = pixel
      f = grid[j, i]
      o = pixel
      b, g, r = (0, clamp255(-8 * o), clamp255(8 * o)) if f > 0 else (255, 255, 255)
      image[j, i, 0] = b
      image[j, i, 1] = g
      image[j, i, 2] = r
  return image

def convert_image_plt(array, grid):
  """
  テスト
  """
  ny = array.shape[0]
  nx = array.shape[1]
  image  = np.empty((ny, nx, 3), np.uint8)
  to_rgb = lambda u,v,p,o: (clamp255(8 * o), clamp255(-8 * o), 0) # 渦度
  # to_rgb = lambda u,v,p,o: (clamp255(0.005 * p), 0, clamp255(-0.005 * p)) # 圧力

  for j, col in enumerate(array):
    for i, pixel in enumerate(col):
      # u, v, f = pixel
      # o = u - col[i - 1, 0] + v - array[j - 1, i, 1] if i > 0 and j > 0 else 0
      # u, v, f, o = pixel
      r, g, b = to_rgb(*pixel) if grid[j, i] > 0 else (255, 255, 255)
      image[j, i, 0] = r
      image[j, i, 1] = g
      image[j, i, 2] = b
  return image

def read_plt(path):
  """
  読み込む物理量を書き換える
  現在は u, v, p, r
  """
  print("reading: " + path)
  with open(path, "r") as file:
    file.readline()
    line   = file.readline()
    nx, ny = (int(s) for s in re.findall(r"\d+", line))
    image  = np.empty((ny, nx, 4), np.float32)
    for j in range(ny):
      for i in range(nx):
        line = file.readline()
        x, y, u, v, p, f, o = (float(s) for s in line.split())
        # b, g, r = fn(x, y, u, v, p, f, o) if f > 0 else (255, 255, 255)
        image[j, i, 0] = u
        image[j, i, 1] = v
        image[j, i, 2] = p
        image[j, i, 3] = o
  # if outfile:
  #   cv2.imwrite(outfile, image)
  return image

def read_cfd_data(path, begin=0, size=None, step=1):
  """
  呼び出し: read_plt
  """
  with chdir(path):
    files = glob.glob("*.plt")[begin : begin+size if size else -1 : step]
    return np.array([read_plt(x) for x in files])

def save_cfd_data():
  """
  naca0012のデータを書き込み
  呼び出し: read_cfd_data
  """
  print("save_cfd_data: start")
  cfd_dir = "/path/to/CFD"
  datafile = 'naca0012_t102_d1_re104_step100.npz'

  with chdir(cfd_dir):
    dirs = (d for d in glob.glob("*") if os.path.isdir(d))
    data = {}
    for d in dirs:
      alpha = (lambda m: m and m.group())(re.search(r"(?<=_)a\d+", d))
      if not alpha:
        continue
      print(d, alpha)
      data[alpha] = read_cfd_data(d, step=100)
  print(data)

  np.savez_compressed(datafile, **data)
  print("save_cfd_data: done")

def load_cfd_data(key=None):
  """
  naca0012のデータを読み込み
  """
  data_loc = os.path.dirname(__file__)
  data_file = 'naca0012_t102_d1_re104_step100.npz'
  with chdir(data_loc):
    if os.path.isfile(data_file):
      data = np.load(data_file)
    else:
      print(f"load_cfd_data: file not found ({data_file})")
      return None
  print("load_cfd_data: done")
  return data[key] if key else data

def load_grid():
  """
  格子のデータを読み込み
  """
  key = "a4"
  grid_loc = f'/path/to/naca0012_t102_d1_re104_{key}'
  grid_file = "grid.csv"
  with chdir(grid_loc):
    data = np.loadtxt(grid_file, delimiter=",", dtype=np.uint8)
  return data

def get_cfd_data(k=lambda a,_:a):
  """
  テスト (後で消す)
  """
  print("get_cfd_data: loading...")
  datafile = 'data_uvfw.npz'
  if os.path.isfile(datafile):
    data = np.load(datafile)
    data1 = data['t500']
    data2 = data['t900']
  else:
    data1 = read_cfd_data("/path/to/cir8_t102_d1_re104", 500, 400)
    data2 = read_cfd_data("/path/to/cir8_t102_d1_re104", 900, 100)
    np.savez_compressed(datafile, t500=data1, t900=data2)
  print("get_cfd_data: done")

  # print(type(data)) # => <class 'numpy.ndarray'>
  # print(data.shape) # => (100 , 102400)
  # print(data.dtype) # => float32
  # cv2.imwrite("testimage.png", convert_image(data[0]))
  # print("get_data0")

  # data1 = data1.reshape(400, -1)
  # data2 = data2.reshape(100, -1)
  # データ選択 (渦度)
  array1 = data1[:, :, :, 3].reshape(400, -1)
  array2 = data2[:, :, :, 3].reshape(100, -1)
  array0 = np.vstack([array1, array2])
  grid = data1[0, :, :, 2]
  return k(array0, grid)#[:, 2]

def test_imshow():
  """
  CFD画像を表示 (リサイズ)
  """
  cfd_data = load_cfd_data()
  grid = load_grid()

  # head = cfd_data[0, :, :, :] # idx4 => [u, v, p, o]

  # 出力1
  # image = convert_image_cv(head, grid)
  # cv2.imwrite("test_image.png", image)

  # 出力2
  for i in range(10):
    print(i)
    head = cfd_data[i, :, :, :] # idx4 => [u, v, p, o]
    image = convert_image_plt(head, grid)
    plt.subplot(2, 5, 1 + i)
    plt.imshow(image)
  plt.show()

  # 出力3
  # image = convert_image_plt(head, grid)
  # image_s = cv2.resize(image, (64, 32), interpolation=cv2.INTER_LINEAR)
  # print(image_s.shape)
  # plt.imshow(image_s)
  # plt.show()

if __name__ == '__main__':
  # test_imshow()
  # save_cfd_data()
  print(load_cfd_data().shape)
  # print(1 and None)
