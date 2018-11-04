#! /usr/bin/env python3

'''
tensorflow練習用スクリプト
FFDデータ学習
'''

import argparse
import os
import shutil
import sys
from datetime import date
from functools import reduce
from contextlib import contextmanager

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as plc
import cv2

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

# from cfd import cfd_base
from cfd import cfd_loader as cl
from cfd import cfd_visualizer as cv
# from cfd import cfd_printer as cp

import myutils as ut

# Tensorflow警告レベル設定
os.environ[ 'TF_CPP_MIN_LOG_LEVEL' ] = '2'

BATCH_NORM_FLAG = False
DROPOUT_FLAG = False
WITH_GRID = True


OS = os.environ.get('OS')
OS_WIN = OS == 'Windows_NT'
FILENAME = os.path.splitext(os.path.basename(__file__))[0]

DEVICE = '/cpu:0'
PROGRESSBAR = True


################################################################################

def layer_batch_norm(x, n_out, phase_train):
    beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
    gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32)
    beta = tf.get_variable("beta", [n_out], initializer=beta_init)
    gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)
    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean = ema.average(batch_mean)
    ema_var = ema.average(batch_var)

    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    # mean, var = control_flow_ops.cond(phase_train, mean_var_with_update, lambda: (ema_mean, ema_var))
    mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema_mean, ema_var))

    x_r = tf.reshape(x, [-1, 1, 1, n_out])
    # normed = tf.nn.batch_norm_with_global_normalization(x_r, mean, var, beta, gamma, 1e-3, True)
    normed = tf.nn.batch_normalization(x_r, mean, var, beta, gamma, 1e-3)
    return tf.reshape(normed, [-1, n_out])


def conv_batch_norm(x, n_out, phase_train):
    beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
    gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32)
    beta = tf.get_variable("beta", [n_out], initializer=beta_init)
    gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean = ema.average(batch_mean)
    ema_var = ema.average(batch_var)

    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    # mean, var = control_flow_ops.cond(phase_train, mean_var_with_update, lambda: (ema_mean, ema_var))
    mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema_mean, ema_var))

    # normed = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, 1e-3, True)
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def layer(input, weight_shape, phase_train, activate=tf.nn.relu):
    bias_shape = weight_shape[1:2]

    weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)

    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)

    output = tf.nn.bias_add(tf.matmul(input, W), b)
    if activate:
        if BATCH_NORM_FLAG:
            return activate(layer_batch_norm(output, bias_shape[0], phase_train))
            # return activate(tf.layers.batch_normalization(output, training=phase_train))
        else:
            return activate(output)
    else:
        return output


def conv2d(input, weight_shape, phase_train, activate=tf.nn.relu):
    incoming = weight_shape[0] * weight_shape[1] * weight_shape[2]
    bias_shape = weight_shape[3:4]

    weight_init = tf.random_normal_initializer(stddev=(2.0/incoming)**0.5)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)

    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)

    output = tf.nn.bias_add(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME'), b)
    if activate:
        if BATCH_NORM_FLAG:
            return activate(conv_batch_norm(output, bias_shape[0], phase_train))
            # return activate(tf.layers.batch_normalization(output, training=phase_train))
        else:
            return activate(output)
    else:
        return output


def deconv2d(input, weight_shape, phase_train, k=1, activate=tf.nn.relu):
    incoming = weight_shape[0] * weight_shape[1] * weight_shape[3]
    bias_shape = weight_shape[2:3]

    weight_init = tf.random_normal_initializer(stddev=(2.0/incoming)**0.5)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)

    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)

    output_shape = tf.concat([tf.shape(input)[0:3] * [1, k, k], tf.shape(b)], 0)
    output = tf.nn.bias_add(tf.nn.conv2d_transpose(input, W, output_shape=output_shape, strides=[1, k, k, 1], padding='SAME'), b)
    if activate:
        if BATCH_NORM_FLAG:
            return activate(conv_batch_norm(output, bias_shape[0], phase_train))
            # return activate(tf.layers.batch_normalization(output, training=phase_train))
        else:
            return activate(output)
    else:
        return output


def max_pool(input, k=2):
    return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def upsample(input, k=2):
    shape = tf.shape(input) * [1, k, k, 1]
    return tf.reshape(tf.tile(tf.expand_dims(input, 3), [1, 1, k, k, 1]), shape)
    # return tf.image.resize_images(input, size=size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


################################################################################

def encoder(input, keep_prob, phase_train, model_shape):
    def create_conv(x, x_ch, filter_size, filter_ch, pool_size, act, name):
        with tf.variable_scope(name):
            weight_shape = [filter_size, filter_size, x_ch, filter_ch]
            layer_conv = conv2d(x, weight_shape, phase_train, activate=act)
            layer_pool = max_pool(layer_conv, k=pool_size)
            return layer_pool

    def func_conv(acc, item):
        x = acc
        x_ch = x.shape.as_list()[3]
        i, (f_s, f_ch, p_s, act) = item
        name = f"conv_{i}"
        return create_conv(x, x_ch, f_s, f_ch, p_s, act, name)

    def create_fc(x, x_ch, y_ch, act, name):
        with tf.variable_scope(name):
            weight_shape = [x_ch, y_ch]
            layer_fc = layer(x, weight_shape, phase_train, activate=act)
            if DROPOUT_FLAG:
                return tf.nn.dropout(layer_fc, keep_prob)
            else:
                return layer_fc

    def func_fc(acc, item):
        x = acc
        x_ch = x.shape.as_list()[1]
        i, (y_ch, act) = item
        name = f"fc_{i}"
        return create_fc(x, x_ch, y_ch, act, name)

    list_conv, list_fc = model_shape

    with tf.variable_scope("encoder"):
        conv_out = reduce(func_conv, enumerate(list_conv), input)
        conv_out_size = np.prod(conv_out.shape.as_list()[1:])
        conv_out_flat = tf.reshape(conv_out, [-1, conv_out_size])
        fc_out = reduce(func_fc, enumerate(list_fc), conv_out_flat)

    return fc_out


def decoder(input, keep_prob, phase_train, model_shape):
    def create_conv(x, x_ch, filter_size, filter_ch, pool_size, act, name):
        with tf.variable_scope(name):
            weight_shape = [filter_size, filter_size, filter_ch, x_ch]
            layer_pool = upsample(x, k=pool_size)
            layer_conv = deconv2d(layer_pool, weight_shape, phase_train, activate=act)
            return layer_conv

    def func_conv(acc, item):
        x = acc
        x_ch = x.shape.as_list()[3]
        i, (f_s, f_ch, p_s, act) = item
        name = f"conv_{i}"
        return create_conv(x, x_ch, f_s, f_ch, p_s, act, name)

    def create_fc(x, x_ch, y_ch, act, name):
        with tf.variable_scope(name):
            weight_shape = [x_ch, y_ch]
            layer_fc = layer(x, weight_shape, phase_train, activate=act)
            if DROPOUT_FLAG:
                return tf.nn.dropout(layer_fc, keep_prob)
            else:
                return layer_fc

    def func_fc(acc, item):
        x = acc
        x_ch = x.shape.as_list()[1]
        i, (y_ch, act) = item
        name = f"fc_{i}"
        return create_fc(x, x_ch, y_ch, act, name)

    list_fc, list_conv, conv_in_shape = model_shape

    with tf.variable_scope("decoder"):
        fc_out = reduce(func_fc, enumerate(list_fc), input)
        conv_in = tf.reshape(fc_out, [-1, *conv_in_shape])
        conv_out = reduce(func_conv, enumerate(list_conv), conv_in)

    return conv_out


def filter_summary_concat(list_conv, list_deconv):
    summarys = []
    list_shape = [("encoder", list_conv), ("decoder", list_deconv)]
    for name, shape in list_shape:
        with tf.variable_scope(name, reuse=True):
            for i in range(len(shape)):
                with tf.variable_scope(f"conv_{i}", reuse=True):
                    W = tf.get_variable("W")
                    b = tf.get_variable("b")
                    W_T = tf.transpose(W, perm=(2, 0, 3, 1)) # (H, W, Cin, Cout) -> (Cin, H, Cout, W)
                    img = tf.reshape(W_T, [1, W.shape[0] * W.shape[2], W.shape[1] * W.shape[3], 1])
                    summarys.append(tf.summary.image("filters", img, max_outputs=1))
                    summarys.append(tf.summary.histogram("hist_W", W))
                    summarys.append(tf.summary.histogram("hist_b", b))
    return tf.summary.merge(summarys)


def layer_summary(n_enc, n_dec):
    summarys = []
    list_shape = [("encoder", n_enc), ("decoder", n_dec)]
    for name, n_layer in list_shape:
        with tf.variable_scope(name, reuse=True):
            for i in range(n_layer):
                with tf.variable_scope(f"fc_{i}", reuse=True):
                    W = tf.get_variable("W")
                    b = tf.get_variable("b")
                    summarys.append(tf.summary.histogram("hist_W", W))
                    summarys.append(tf.summary.histogram("hist_b", b))
    return tf.summary.merge(summarys)


################################################################################

def loss(output, x):
    with tf.variable_scope("training"):
        # l2 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(output, x)), axis=[1, 2, 3]))
        l2 = tf.reduce_sum(tf.square(tf.subtract(output, x)), axis=[1, 2, 3]) / 2
        # l2 = tf.reduce_sum(tf.square(tf.subtract(output, x)), axis=1) / 2
        # l2 = tf.nn.l2_loss(tf.subtract(output, x))
        train_loss = tf.reduce_mean(l2)
        train_summary_op = tf.summary.scalar("train_cost", train_loss)
        return train_loss, train_summary_op


def training(cost, global_step):
    optimizer = tf.train.AdamOptimizer(
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        use_locking=False,
        name='Adam'
    )
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op


# def image_summary(summary_label, tensor):
#   """
#   保留
#   """
#   # tensor_zeros = tf.zeros([tensor.shape[0], 480, 960, 1])
#   # tensor_reshaped = tf.reshape(tensor, [-1, 480, 960, 2])
#   tensor_image = tf.slice(tensor, [0, 0, 0, 0], [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
#   return tf.summary.image(summary_label, tensor_image)


# def evaluate(output, x):
#   """
#   保留
#   """
#   with tf.variable_scope("validation"):
#     # in_im_op = image_summary("input_image", x)
#     # out_im_op = image_summary("output_image", output)
#     # in_im_op = tf.constant(0)
#     # out_im_op = tf.constant(0)
#     l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(output, x, name="val_diff")), 1))
#     val_loss = tf.reduce_mean(l2)
#     val_summary_op = tf.summary.scalar("val_cost", val_loss)
#     return val_loss, val_summary_op#, in_im_op, out_im_op


################################################################################
# 訓練部
################################################################################

def create_model_set():
    input_ch = 3 if WITH_GRID else 2
    x = tf.placeholder(tf.float32, shape=(None, 320, 640, input_ch), name="input")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    phase_train = tf.placeholder(tf.bool, name="phase_train")
    global_step = tf.Variable(0, name='global_step', trainable=False)
    feed_dict_f = lambda data, pt: {x: data, keep_prob: 1, phase_train: pt}

    # モデル形状: エンコーダ
    # in => (320, 640, input_ch)
    filter_size = 5
    filter_num = 32
    list_conv = [
        (filter_size, filter_num, 2, tf.nn.relu), # out => (160, 320, 10)
        (filter_size, filter_num, 2, tf.nn.relu), # out => ( 80, 160, 10)
        (filter_size, filter_num, 2, tf.nn.relu), # out => ( 40,  80, 10)
        (filter_size, filter_num, 2, tf.nn.relu), # out => ( 20,  40, 10)
        (filter_size, filter_num, 2, tf.nn.relu), # out => ( 10,  20, 10)
        (filter_size, filter_num, 2, tf.nn.relu), # out => (  5,  10, 10)
    ]
    # fc_in => 500
    list_fc0 = [
        (100, tf.nn.relu),
        (10, None)
    ]

    # モデル形状: デコーダ
    # in => 10
    list_fc1 = [
        # (50, tf.nn.relu),
        (100, tf.nn.relu),
        # (250, tf.nn.relu),
        (5 * 10 * filter_num, tf.nn.relu)
    ]
    # deconv_in => (5, 10, 10)
    list_deconv = [
        (filter_size, filter_num, 2, tf.nn.relu), # out => ( 10,  20, 10)
        (filter_size, filter_num, 2, tf.nn.relu), # out => ( 20,  40, 10)
        (filter_size, filter_num, 2, tf.nn.relu), # out => ( 40,  80, 10)
        (filter_size, filter_num, 2, tf.nn.relu), # out => ( 80, 160, 10)
        (filter_size, filter_num, 2, tf.nn.relu), # out => (160, 320, 10)
        (filter_size, input_ch, 2, None), # out => (320, 640, input_ch)
    ]
    deconv_in_shape = (5, 10, filter_num)

    shape_encode = (list_conv, list_fc0)
    shape_decode = (list_fc1, list_deconv, deconv_in_shape)

    with tf.device(DEVICE):
        code = encoder(x, keep_prob, phase_train, shape_encode)
        output = decoder(code, keep_prob, phase_train, shape_decode)

    cost_op, train_summary_op = loss(output, x)
    train_op = training(cost_op, global_step)
    # eval_op, val_summary_op = evaluate(output, x)

    with tf.variable_scope("validation"):
        val_loss = tf.placeholder(tf.float32, name="val_loss")
        val_summary_op = tf.summary.scalar("val_cost", val_loss)
        val_summary_f = lambda sess, loss: sess.run(val_summary_op, feed_dict={val_loss: loss})

    # filter_summary_op = filter_summary_concat(list_conv, list_deconv)
    filter_summary_op = tf.summary.merge([
        filter_summary_concat(list_conv, list_deconv),
        layer_summary(len(list_fc0), len(list_fc1)),
    ])

    # all_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=200)

    return {
        "code": code,
        "output": output,
        "feed_dict_f": feed_dict_f,
        "global_step": global_step,
        "cost_op": cost_op,
        "train_op": train_op,
        "train_summary_op": train_summary_op,
        # "eval_op": eval_op,
        # "val_summary_op": val_summary_op,
        "val_summary_f": val_summary_f,
        # "all_summary_op": all_summary_op,
        "filter_summary_op": filter_summary_op,
        "saver": saver
    }


def append_grid(data):
    grid = cl.load_grid("naca2412_720x1280")
    grid_r = cv2.resize(grid, (640, 360), interpolation=cv2.INTER_LINEAR)[20:340, :]
    grid_t = np.tile(grid_r[np.newaxis, :, :, np.newaxis], [data.shape[0], 1, 1, 1])
    return np.concatenate([data, grid_t], axis=3)


def get_data():
    size = 100

    def get_f(i, key, t):
        file = f"{key}_{size*i:0>4}-{size*(i+1):0>4}of{t:0>4}"
        data = cl.load_cfd_data(file)[:, 20:340, :, :] * 0.5
        if WITH_GRID:
            return append_grid(data)
        else:
            return data

    data_dict = {}
    # get_f_1 = lambda: np.concatenate([get_f(i, "cylinder_uv_360x640", 1000) for i in range(8)], axis=0)
    # get_f_2 = lambda: np.concatenate([get_f(i, "cylinder_uv_360x640", 1000) for i in range(8,10)], axis=0)
    get_f_1 = lambda: np.concatenate([get_f(i, "naca2412_2_uv_360x640", 1200) for i in range(10)], axis=0)
    get_f_2 = lambda: np.concatenate([get_f(i, "naca2412_2_uv_360x640", 1200) for i in range(10,12)], axis=0)

    def get_batch_it(batch_size=None, data="train", shuffle=False):
        if not data in ["train", "val"]:
            raise Exception(f"data が不正です: {data}")
        if data == "train":
            if not "train" in data_dict:
                data_dict["train"] = get_f_1()
                data_dict["train_1"] = data_dict["train"].copy()
            if shuffle:
                np.random.shuffle(data_dict["train_1"])
                batch = data_dict["train_1"]
            else:
                batch = data_dict["train"]
        else:
            if not "val" in data_dict:
                data_dict["val"] = get_f_2()
            batch = data_dict["val"]

        if batch_size:
            for i in range(batch.shape[0] // batch_size):
                print(f"batch({data}):", i, end=" "*10+"\r")
                yield batch[batch_size*i:batch_size*(i+1)]
        else:
            for i in range(batch.shape[0]):
                yield batch[i]

    return get_batch_it


def train_body(sess, model, batch_it, log_dir):
    feed_dict_f      = model["feed_dict_f"]
    global_step      = model["global_step"]
    cost_op          = model["cost_op"]
    train_op         = model["train_op"]
    train_summary_op = model["train_summary_op"]
    val_summary_f    = model["val_summary_f"]
    saver            = model["saver"]

    filter_summary_op = model["filter_summary_op"]

    training_epochs = 500
    batch_size = 20
    display_step = 1
    saving_step = 10

    def train_batch(acc, batch):
        total_cost, i = acc
        new_cost, _, train_summary = sess.run([
            cost_op,
            train_op,
            train_summary_op
        ], feed_dict=feed_dict_f(batch, True))
        summary_writer.add_summary(train_summary, sess.run(global_step))
        return total_cost + new_cost, i + 1

    def validate_batch(acc, batch):
        total_cost, i = acc
        new_cost = sess.run(cost_op, feed_dict=feed_dict_f(batch, False))
        return total_cost + new_cost, i + 1

    def save_f(epoch, last=False):
        last_s = " (last)" if last else ""
        saver.save(
            sess,
            f"{log_dir}/model-{epoch:0>4}{last_s}.ckpt",
            global_step=global_step
        )

    summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print(f"[{ut.strnow()}]", "Start")
    for epoch in range(training_epochs):
        # === 学習 ===
        total_cost, total_batch = reduce(train_batch, batch_it(batch_size, shuffle=True), (0, 0))
        avg_cost = total_cost / total_batch
        if epoch % display_step == 0:
            print(
                f"[{ut.strnow()}]",
                "Epoch:", f"{epoch:0>4d}",
                "Step:", sess.run(global_step),
                "cost=", f"{avg_cost:.5f}"
            )

        # === 検証 ===
        total_cost, total_batch = reduce(validate_batch, batch_it(batch_size, data="val"), (0, 0))
        avg_cost = total_cost / total_batch
        summary_writer.add_summary(val_summary_f(sess, avg_cost), sess.run(global_step))
        summary_writer.add_summary(*sess.run([filter_summary_op, global_step]))
        print("Validation Loss:", f"{avg_cost:.5f}")

        if epoch % saving_step == 0:
            save_f(epoch)

    # === 学習終了 ===
    if (training_epochs - 1) % saving_step > 0:
        save_f(training_epochs - 1, True)
    print(f"[{ut.strnow()}]", "Optimization Finished!")
    try:
        os.rename(log_dir, ut.uniq_path(log_dir, "dir"))
    except PermissionError:
        print("ディレクトリ名変更スキップ", log_dir)


################################################################################
# メイン
################################################################################

@contextmanager
def run_env():
    gpuConfig = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9,
                                  allow_growth=True),
        device_count={'GPU': 1}
    )
    with tf.device("/cpu:0"):
        with tf.Graph().as_default():
            with tf.variable_scope("main"):
                with tf.Session(config=gpuConfig) as sess:
                    yield sess

@contextmanager
def last_model(sess, saver, log_dir):
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"ログディレクトリがありません: {log_dir}")
    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt:
        last_model = ckpt.model_checkpoint_path
        saver.restore(sess, last_model)
        yield
    else:
        print("モデルデータがありません")
        return


def train_main(out=f'result/{FILENAME}', clear=False):
    # log_dir = "logs_" + date.today().strftime("%y%m%d")
    log_dir = out
    if clear and os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    with run_env() as sess:
        model = create_model_set()
        batch_it = get_data()
        train_body(sess, model, batch_it, log_dir)


################################################################################
# 分析部
################################################################################

def plot_images(rows, cols, minmax, img_f, file=None):
    # == Plot ===
    fig, axs = plt.subplots(rows, cols, figsize=(0.7 * cols - 0.1, 0.7 * rows - 0.1))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=0.1)

    color_list = [(0, "blue"), (0.5, "white"), (1, "red")]
    cmap = plc.LinearSegmentedColormap.from_list('custom_cmap', color_list)

    for i, ax_row in enumerate(axs):
        for j, ax in enumerate(ax_row):
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for d in ["top", "bottom", "left", "right"]:
                # ax.spines[d].set_visible(False)
                ax.spines[d].set_linewidth(0.5)
            # ax.annotate(f"mode{5*i+j+1}", xy=(0.5, -0.1), xycoords="axes fraction", fontsize=10, horizontalalignment="center", verticalalignment="top") # 文字を入れる
            ax.imshow(img_f(i, j), cmap="gray", vmin=minmax[0], vmax=minmax[1])
            # ax.imshow(grid_img)

    if file:
        plt.savefig(file, transparent=True, bbox_inches="tight", pad_inches=0.0)
    else:
        plt.show()


def plot_filter():
    key = "180730"
    # data = ""
    log_dir = f"logs_{key}"
    # npy_file = f"code_history_{key}_{data}.npy"

    minmax = (0, 0)

    with run_env() as sess:
        model = create_model_set()
        saver = model["saver"]
        with last_model(sess, saver, log_dir):
            for name0 in ["encoder", "decoder"]:
                for name1 in ["conv_0", "conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]:
                    image_file = f"filters_{key}_{name0}_{name1}.png"
                    for phase in range(2):
                        with tf.variable_scope(name0, reuse=True):
                            with tf.variable_scope(name1, reuse=True):
                                W_ts = tf.get_variable("W")
                                W = sess.run(W_ts) # (H, W, Cin, Cout)
                                # minmax = np.min(W), np.max(W)
                                # print(W.shape, "min:", minmax[0], "max:", minmax[1])
                                if phase == 0:
                                    minmax = min(minmax[0], np.min(W)), max(minmax[1], np.max(W))
                                else:
                                    plot_images(W.shape[2], W.shape[3], minmax, lambda i, j: W[:, :, i, j], file=image_file)


def analyze_code():
    global BATCH_NORM_FLAG
    BATCH_NORM_FLAG = False

    key = "180729_3"
    data = "train"
    log_dir = f"logs_{key}"
    npy_file = f"code_history_{key}_{data}.npy"

    if os.path.isfile(npy_file):
        c = np.load(npy_file)
    else:
        with run_env() as sess:
            model = create_model_set()
            batch_it = get_data()
            code = model["code"]
            feed_dict_f = model["feed_dict_f"]
            saver = model["saver"]

            with last_model(sess, saver, log_dir):
                c = np.concatenate([
                    sess.run(code, feed_dict=feed_dict_f(x, False))
                    for x in batch_it(10, data=data)
                ], axis=0)

        np.save(npy_file, c)
        np.savetxt(f"code_history_{key}_{data}.csv", c, delimiter=",")

    print(c.shape)
    plt.plot(c)
    plt.show()


def get_grid_image():
    grid = cl.load_grid("naca2412_720x1280")
    grid_r = 255 * (1 - cv2.resize(grid, (640, 360), interpolation=cv2.INTER_LINEAR)[20:340, :])
    grid_c = np.zeros_like(grid_r, dtype=np.uint8)
    return cv2.merge((grid_c, grid_c, grid_c, grid_r))


def analyze_out():
    grid_img = get_grid_image()

    # === Plot ===
    fig, axs_1 = plt.subplots(2, 2)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=0)
    axs = axs_1.flatten()
    def init_ax():
        for ax in axs:
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    def clear_ax():
        for ax in axs:
            ax.cla()
    color_list = [(0, "blue"), (0.5, "white"), (1, "red")]
    cmap = plc.LinearSegmentedColormap.from_list('custom_cmap', color_list)
    # === Plot end ===

    key = "180730_4"
    data = "train"
    fps = 2
    log_dir = f"logs_{key}"
    video_file = f"cae_{key}_{data}_fps{fps}.mp4"

    barch_size = 20
    total_batch = 1000

    with run_env() as sess:
        model = create_model_set()
        batch_it = get_data()
        output = model["output"]
        feed_dict_f = model["feed_dict_f"]
        saver = model["saver"]
        errs = []

        def update():
            def f():
                with last_model(sess, saver, log_dir):
                    for x in batch_it(barch_size, data=data):
                        y = sess.run(output, feed_dict=feed_dict_f(x, False))
                        for i in range(0, barch_size):
                            err = np.sum((x[i] - y[i]) ** 2) / 2
                            errs.append(err)
                            print("err:", err)
                            # yield err

                            if i % 5 > 0:
                                continue

                            clear_ax()
                            axs[0].imshow(x[i, :, :, 0], cmap=cmap, vmin=0.6, vmax=0.9)
                            axs[1].imshow(y[i, :, :, 0], cmap=cmap, vmin=0.6, vmax=0.9)
                            axs[2].imshow(x[i, :, :, 1], cmap=cmap, vmin=0.3, vmax=0.7)
                            axs[3].imshow(y[i, :, :, 1], cmap=cmap, vmin=0.3, vmax=0.7)
                            for ax in axs:
                                ax.imshow(grid_img)
                            axs[3].annotate(f"Error {err:.3f}", xy=(0.9, -0.1), xycoords="axes fraction", fontsize=10, horizontalalignment="right", verticalalignment="top")
                            yield
            gen = f()
            return lambda i: print(f"export: {i} / {total_batch}", end=" "*10+"\r") or gen.__next__()

        with ut.stopwatch("anim"):
            cv.show_anim(fig, update(), frames=total_batch//5, init_func=init_ax, file=video_file, fps=fps)

        print("mean error:", np.mean(errs))
        np.save(f"cae_errs_{key}_{data}.npy", errs)

        return

        with ut.stopwatch("err"):
            errs = [x for x in update()]
            np.save(f"errs_{key}_{data}", errs)


def plot_train_history():
    h_t = np.loadtxt("cae_cfd_logs_180730_4_validation_train_cost.csv", delimiter=",", dtype=np.float32, skiprows=1)
    h_v = np.loadtxt("cae_cfd_logs_180730_4_validation_val_cost.csv", delimiter=",", dtype=np.float32, skiprows=1)
    print(h_t.shape)
    plt.ylim(0, 100)
    plt.xlabel("steps")
    plt.ylabel("error")
    plt.plot(h_t[:, 1], h_t[:, 2], label="training error")
    plt.plot(h_v[:, 1], h_v[:, 2], label="validation error")
    plt.legend()
    plt.savefig("errors_train_hist.png", transparent=True, bbox_inches="tight", pad_inches=0.0)
    plt.show()


################################################################################

def __test__():
        pass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', default='',
                        choices=['', '0', '01', '1'],
                        help='Number of main procedure')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--clear', '-c', action='store_true',
                        help='Remove directory at the beginning')
    parser.add_argument('--no-progress', '-p', action='store_true',
                        help='Hide progress bar')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run as test mode')
    args = parser.parse_args()
    return args


def main():
    global DEVICE

    args = get_args()
    if args.gpu >= 0:
        DEVICE = f'/gpu:{args.gpu}'
    PROGRESSBAR = not args.no_progress

    if args.out:
        out = f'result/{args.out}'
    else:
        out = f'result/{FILENAME}'
    clear = args.clear

    if args.test:
        __test__()
        return

    if args.mode == '0':
        with ut.stopwatch('sample0'):
            train_main(out=out, clear=clear)


if __name__ == '__main__':
    with ut.stopwatch("main"):
        main()
