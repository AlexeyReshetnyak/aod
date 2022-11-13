#!/usr/bin/env python3

import numpy as np
import cv2
import ctypes as C
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("alpha", help="display")
parser.add_argument("epsilon", help="display")


class rgb_pixel(C.Structure):
    _fields_ = [("r", C.c_float),
                ("g", C.c_float),
                ("b", C.c_float)]


class som_net(C.Structure):
    _fields_ = [("mt_n_rows",   C.c_int),
                ("mt_n_cols",   C.c_int),
                ("mt_n_layers", C.c_int),
                ("sm_n_rows",   C.c_int),
                ("sm_n_cols",   C.c_int),
                ("offset",      C.c_int),
                ("kernel_size", C.c_int),
                ("epsilon",     C.c_float),
                ("alpha",       C.c_float),
                ("alpha1d",     C.c_float),
                ("fg",          C.POINTER(C.c_float)),
                ("bg",          C.POINTER(C.c_float)),
                ("init_img",    C.POINTER(C.c_float)),
                ("mt",          C.POINTER(rgb_pixel)),
                ("sm",          C.POINTER(C.c_float)),
                ("count",       C.POINTER(C.POINTER(C.c_float))),
                ("kernel",      C.POINTER(C.POINTER(C.c_float))),
                ("kernel1d",    C.POINTER(C.c_float))]


def get_net_state(lib, img, layer):
    lib.get_net_state(img.ctypes.data_as(C.POINTER(C.c_ubyte)),
                      C.c_int(layer))


def update(lib, img, net):
    lib.update(img.ctypes.data_as(C.POINTER(C.c_float)),
               C.byref(net))


def destroy_net(lib):
    lib.destroy_net()


def img2f(img):
    return np.float32(img/255.0)


def init_kernel1d(k_len, normalize=False):
    sigma = 1.0

    def g(x): return (1 / (sigma * np.sqrt(2*np.pi)) *
                      np.exp(-float(x)**2/(2*sigma**2)))
    kernel = [g(x) for x in range(k_len)]
    np_kernel = np.array(kernel, dtype=np.float32)
    if normalize:
        np_kernel = np_kernel/np.max(np_kernel)
    return np_kernel


def init_net(lib, first_img, fg, bg, sm, img_n_rows, img_n_cols, mt_n_layers,
             offset, kernel_size, epsilon, alpha, alpha1d):
    net = som_net()
    net.mt_n_rows = img_n_rows + 2*offset
    net.mt_n_cols = img_n_cols + 2*offset
    net.mt_n_layers = mt_n_layers
    net.sm_n_rows = img_n_rows
    net.sm_n_cols = img_n_cols
    net.offset = offset
    net.kernel_size = kernel_size
    net.epsilon = epsilon
    net.alpha = alpha
    net.alpha1d = alpha1d
    net.init_img = first_img.ctypes.data_as(C.POINTER(C.c_float))
    net.bg = bg.ctypes.data_as(C.POINTER(C.c_float))
    net.fg = fg.ctypes.data_as(C.POINTER(C.c_float))
    kern1d = init_kernel1d(mt_n_layers, normalize=False)
    net.kernel1d = kern1d.ctypes.data_as(C.POINTER(C.c_float))
    net.count = (C.POINTER(C.c_float) * img_n_rows)()
    for i in range(img_n_rows):
        net.count[i] = (C.c_float * img_n_cols)()
        C.memset(net.count[i], 0, img_n_cols * C.sizeof(C.c_float))
    net.sm = sm.ctypes.data_as(C.POINTER(C.c_float))
    lib.init_net(C.byref(net))  # TODO: check returned value or alloc all
    # TODO: memory on the python side
    return net


if __name__ == '__main__':
    libaod3d = C.cdll.LoadLibrary('./aod3d/libaod3d.so')
    args = parser.parse_args()
    cap = cv2.VideoCapture(0)
    _, first_img = cap.read()
    imf = img2f(first_img)
    bg_img = np.zeros_like(imf)
    n_rows = first_img.shape[0]
    n_cols = first_img.shape[1]
    fg_img = np.zeros((n_rows, n_cols), dtype=np.float32)
    sm_img = np.zeros((n_rows, n_cols), dtype=np.float32)
    n_layers = 5
    offset = 1
    kernel_size = 3
    epsilon = 0.2  # np.float32(args.epsilon)
    alpha = 0.2  # np.float32(args.alpha)
    alpha1d = 0.1
    net = init_net(libaod3d, imf, fg_img, bg_img, sm_img, n_rows, n_cols,
                   n_layers, offset, kernel_size, epsilon, alpha, alpha1d)
    net_state_img0 = np.zeros_like(imf)
    net_state_img1 = np.zeros_like(imf)
    while True:
        _, img = cap.read()
        imgf = img2f(img)
        update(libaod3d, imgf, net)
   #     FIXME: segfolt on get_net_state
   #     get_net_state(libaod3d, net_state_img0, 0)
   #     get_net_state(libaod3d, net_state_img1, 1)
        cv2.imshow('imgf', imgf)
        cv2.imshow('fgi', fg_img)
        cv2.imshow('bgi', bg_img)
        cv2.imshow('sm', sm_img)
        cv2.imshow('net_state_img0', net_state_img0)
        cv2.imshow('net_state_img1', net_state_img1)
        if cv2.waitKey(1) == 27:
            exit()

    destroy_net(libaod3d)
