# 追加
# os.environ[ 'TF_CPP_MIN_LOG_LEVEL' ] = '2'
# sys.path.append('../..')

import cv2
import glob
import os
import re
import sys

import numpy as np
from matplotlib import pyplot as plt

from myutils import chdir, stopwatch


# ffmpeg = "C/path/to/ffmpeg.exe"
# import glob, re, shutil


# def clamp255(c):
#   return min(255, max(0, int(255 * c)))


def read_plt(path, extract):
    """
    読み込む物理量を書き換える
    現在は u, v
    """
    # def extract(x, y, u, v, p, f, o):
    #   return [u, v]

    print("reading: " + path)
    with open(path, "r") as file:
        file.readline()
        line   = file.readline()
        nx, ny = (int(s) for s in re.findall(r"\d+", line))
        image  = np.empty((ny, nx, 2), dtype=np.float32)

        for j in range(ny):
            for i in range(nx):
                line = file.readline()
                image[j, i, :] = extract(*(float(s) for s in line.split()))

    return image


def read_cfd_data(path, begin=0, size=None, step=1):
    """
    呼び出し: read_plt
    """
    with chdir(path):
        files = glob.glob("out_*.plt")[begin : begin+size if size else -1 : step]
        return np.fromiter(read_plt(x) for x in files)


def find_dir(file, mode="file"):
    dirs = [
        'data',
        os.path.dirname(__file__)
    ]

    exist_f = os.path.isfile if mode == "file" else os.path.isdir

    for dir in dirs:
        for d in glob.iglob(dir + "/**", recursive=True):
            if not os.path.isdir(d):
                continue
            with chdir(d):
                if exist_f(file):
                    return d

    return None
    # raise FileNotFoundError(f"ファイルが見つかりません({file})")


def load_cfd_data(name, ext=".npy"):
    # data_loc = os.path.dirname(__file__)
    data_file = name + ext
    data_loc = find_dir(data_file)
    print(f"load_cfd_data: file={data_file}", end="\r")

    if not data_loc:
        if ext == ".npy":
            return load_cfd_data(name, ext=".npz")
        else:
            raise FileNotFoundError(f"load_cfd_data: ファイルが見つかりません ({data_file})")

    with chdir(data_loc):
        if os.path.isfile(data_file):
            if ext == ".npy":
                data = np.load(data_file)
            else:
                with np.load(data_file) as npf:
                    # print(f"load_cfd_data: name={name} files={npf.files}", end="\r")
                    data = npf['arr_0']
        # elif ext == ".npy":
        #   return load_cfd_data(name, ext=".npz")
        # else:
        #   raise FileNotFoundError(f"load_cfd_data: file not found ({data_file})")

    print(" " * 80, end="\r")
    # print("load_cfd_data: done", end="\r")
    return data


def load_grid(name, dtype=np.uint8):
    data_file = name + ".csv"
    data_loc = find_dir(data_file)

    if not data_loc:
        raise FileNotFoundError(f"load_grid: ファイルが見つかりません ({data_file})")

    with chdir(data_loc):
        return np.loadtxt(data_file, delimiter=",", dtype=np.float32)


################################################################################
# 以下一時作業用
################################################################################

def save_cfd_data():
    """
    naca0012のデータを書き込み (v2)
    *直接呼ぶ
    """
    print("save_cfd_data: start")
    cfd_dir = "/path/to/CFD"
    data_dir = "naca0012_t102_d1_re104_a4"
    file_fmt = "out_%04d.plt"
    begin = 1000
    size = 1000
    step = 1

    datafile = f"{data_dir}_{begin: %04d}-{begin+size-1: %04d}of{2000: %04d}.npz"

    with chdir(f"{cfd_dir}/{data_dir}"):
        data = [read_plt(file_fmt % i) for i in range(begin, begin + size, step)]

    np.savez_compressed(datafile, f())

    print("save_cfd_data: done")


def save_cfd_data_all():
    """
    naca0012のデータを書き込み (v2)
    *直接呼ぶ
    """
    print("save_cfd_data_all: start")
    cfd_dir = "/path/to/CFD"
    data_dirs = [
        "naca0012_t102_d1_re104_a0",
        "naca0012_t102_d1_re104_a4",
        "naca0012_t102_d1_re104_a8",
        "naca0012_t102_d1_re104_a12"
    ]
    file_fmt = "out_%04d.plt"
    begins = [0, 1000]
    size = 1000
    step = 1
    for data_dir in data_dirs:
        for begin in begins:
            datafile = f"{data_dir}_{begin:0>4}-{begin+size-1:0>4}of{2000:0>4}.npz"
            if os.path.isfile(datafile):
                continue
            with chdir(f"{cfd_dir}/{data_dir}"):
                data = [read_plt(file_fmt % i) for i in range(begin, begin + size, step)]
            np.savez_compressed(datafile, data)
    print("save_cfd_data_all: done")


def load_grid_manual():
    """
    格子のデータを読み込み
    """
    key = "a4"
    grid_loc = f'/path/to/naca0012_t102_d1_re104_{key}'
    grid_file = "grid.csv"
    with chdir(grid_loc):
        data = np.loadtxt(grid_file, delimiter=",", dtype=np.uint8)
    return data


def load_naca0012(key, begin, size):
    """
    EX: key=a0, begin=1000
    """
    file = f"naca0012_t102_d1_re104_{key}_{begin:0>4}-{begin+size:0>4}of2000"
    return load_cfd_data(file)


def load_grid_naca0012(key):
    """
    EX: key=a0, begin=1000
    """
    grid_dir = os.path.dirname(__file__)
    grid_file = f"grid_naca0012_480x960_{key}.csv"
    with chdir(grid_dir):
        data = np.loadtxt(grid_file, delimiter=",", dtype=np.uint8)
    return data


def split_data():
    """
    TEMP
    """
    key = "a8"
    src = lambda b: load_naca0012(key, b, 1000)
    for j in range(2):
        data_all = src(1000 * j)
        for i in range(10):
            begin = 1000 * j + 100 * i
            size = 100
            dst = f"naca0012_t102_d1_re104_{key}_{begin:0>4}-{begin+size:0>4}of2000"
            data = data_all[100 * i : 100 * (i + 1)]
            np.savez_compressed(dst, data)


def convert_data():
    """
    データ変換 (手動)
    """
    total = 1000
    size_i = 200
    size_o = 100
    count_i = total // size_i
    count_o = size_i // size_o
    # basename = "naca2412_1"
    basename = "cylinder"
    shape = (640, 360)

    # grid_i = load_grid("naca2412_720x1280", dtype=np.float32)
    # shape = (grid_i.shape[1]//2, grid_i.shape[0]//2)
    # grid_o = cv2.resize(grid_i, shape, interpolation=cv2.INTER_LINEAR)

    def resize_image(src):
        dst0 = cv2.resize(src[:, :, 0], shape, interpolation=cv2.INTER_LINEAR)
        dst1 = cv2.resize(src[:, :, 1], shape, interpolation=cv2.INTER_LINEAR)
        dst = cv2.merge((dst0, dst1))
        return dst

    def load_f(i):
        # file = f"{basename}_uvpc_720x1280_{size_i*i:0>4}-{size_i*(i+1):0>4}of{total:0>4}"
        file = f"{basename}_uvpc_{size_i*i:0>4}-{size_i*(i+1):0>4}of{total:0>4}"
        return load_cfd_data(file)

    def save_f(i, data_o):
        file = f"{basename}_uv_360x640_{size_o*i:0>4}-{size_o*(i+1):0>4}of{total:0>4}.npy"
        if os.path.isfile(file):
            return
        np.save(file, data_o)
        # np.savez_compressed(file, data_o)

    for i in range(count_i):
        data_i = load_f(i)
        for j in range(count_o):
            data_o = [resize_image(x) for x in data_i[size_o*j : size_o*(j+1)]]
            save_f(count_o*i+j, data_o)


def check_deta_range():
    size = 100
    total = 1000
    count = total // size
    # basename = "naca2412_2"
    basename = "cylinder_uv_360x640"
    for i in range(count):
        data = load_cfd_data(f"{basename}_{size*i:0>4}-{size*(i+1):0>4}of{total:0>4}")
        min_u = data[:, :, :, 0].min()
        max_u = data[:, :, :, 0].max()
        min_v = data[:, :, :, 1].min()
        max_v = data[:, :, :, 1].max()
        print(i, f"u: [{min_u:.3g}, {max_u:.3g}] v: [{min_v:.3g}, {max_v:.3g}]")


def loading_speed_test():
    """
    スピードテスト
    """
    size = 100
    total = 2000
    count = total // size
    @stopwatch
    def a():
        for i in range(count):
            load_cfd_data(f"naca2412_1_c_360x640_{size*i:0>4}-{size*(i+1):0>4}of{total:0>4}", ".npy")
    @stopwatch
    def b():
        for i in range(count):
            load_cfd_data(f"naca2412_1_c_360x640_{size*i:0>4}-{size*(i+1):0>4}of{total:0>4}")
    a()
    b()

if __name__ == '__main__':
    # test_imshow()
    # save_cfd_data()
    # print(load_cfd_data("a4_1000-1999of2000").shape)
    # save_cfd_data()
    # print(load_cfd_data("a4_1000-1999of2000").shape)
    # print(1 and None)
    # split_data()
    convert_data()
    # check_deta_range()
    # loading_speed_test()
    # shape = load_cfd_data("naca0012_t102_d1_re104_a0_uvpc_480x960_0000-0100of2000").shape
    # print(shape)
