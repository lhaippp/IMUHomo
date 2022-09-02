import os
import cv2
import argparse

import numpy as np

from gyro_aglinment import warp_img_make_gif, RS_rect_warp_img_make_gif

K_cc9_600_800 = np.array([[573.8534, -0.6974, 406.0101], [0, 575.0448, 309.0112], [0, 0, 1]])

# ZOOM X3
# K_cc9_600_800 = 1e3 * np.array([[1.7105, 0.0009, 0.4094],
#                           [0, 1.7099, 0.3189], [0, 0, 0.0010]])
patch = 14
ts = 0.033312

# patch = 6
# ts = 0.015138971


def diffrotation(gyrostamp, gyroidxa, gyroidxb, ta, tb, anglev, gyrogap):
    '''
    R: rotation matrice
    anglev: rate of rotation
    '''
    R = np.eye(3)
    for i in range(gyroidxa, gyroidxb + 1):
        # 计算积分时间dt = tb - ta
        if i == gyroidxa:
            dt = gyrostamp[i] - ta
        elif i == gyroidxb:
            dt = tb - gyrostamp[i - 1]
        else:
            dt = gyrogap[i - 1]

        if gyroidxa == gyroidxb:
            dt = tb - ta
        tempv = dt * (anglev[i - 1, :])  # gyro积分
        theta = np.linalg.norm(tempv)
        tempv = tempv / theta
        tempv = tempv.tolist()
        # gyro和camera坐标轴不同（gyro的x是相机的y）
        skewv = np.array([[0, -tempv[2], tempv[0]], [tempv[2], 0, -tempv[1]], [-tempv[0], tempv[1], 0]])
        nnT = np.array([[tempv[1] * tempv[1], tempv[1] * tempv[0], tempv[1] * tempv[2]],
                        [tempv[1] * tempv[0], tempv[0] * tempv[0], tempv[0] * tempv[2]],
                        [tempv[1] * tempv[2], tempv[0] * tempv[2], tempv[2] * tempv[2]]])
        tempr = np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * nnT + np.sin(theta) * skewv
        R = np.dot(R, tempr)
    return R


def imagesHomography(num_frames: int,
                     patch: int,
                     gyrostamp: np.ndarray,
                     gyrogap: np.ndarray,
                     anglev: np.ndarray,
                     framestamp: np.ndarray,
                     K: np.ndarray,
                     delta_T=0):
    Kinv = np.linalg.inv(K)
    hom = np.zeros((patch, 3, 3))
    for i in range(patch):
        ta = framestamp[num_frames - 1] + ts * i / patch + delta_T
        tb = framestamp[num_frames] + ts * i / patch + delta_T
        gyroidxa = np.where(gyrostamp > ta)[0][0]
        gyroidxb = np.where(gyrostamp > tb)[0][0]
        gyroidxa = gyroidxa if gyroidxa != 0 else 1
        gyroidxb = gyroidxb if gyroidxb != 0 else 1
        R = diffrotation(gyrostamp, gyroidxa, gyroidxb, ta, tb, anglev, gyrogap)
        hom[i] = K.dot(R).dot(Kinv)
    return hom


def compute_intra_homography(num_frames: int, patch: int, gyrostamp: np.ndarray, gyrogap: np.ndarray, anglev: np.ndarray,
                             framestamp: np.ndarray, K: np.ndarray):
    Kinv = np.linalg.inv(K)
    hom = np.zeros((patch, 3, 3))
    for i in range(patch):
        # 每一帧的时间戳
        ta = framestamp[num_frames - 1]
        # 每一帧中第i个patch的时间戳
        tb = framestamp[num_frames - 1] + ts * (i / patch)
        gyroidxa = np.where(gyrostamp > ta)[0][0]
        gyroidxb = np.where(gyrostamp > tb)[0][0]
        gyroidxa = gyroidxa if gyroidxa != 0 else 1
        gyroidxb = gyroidxb if gyroidxb != 0 else 1
        R = diffrotation(gyrostamp, gyroidxa, gyroidxb, ta, tb, anglev, gyrogap)
        hom[i] = K.dot(R).dot(Kinv)
        if i == 0:
            hom[i] = np.eye(3, 3)
        # normalize homography
        hom[i] /= hom[i][-1, -1]
        # change direction to patch i -> patch 0
        hom[i] = np.linalg.inv(hom[i])
    return hom


def make_trainset_source(framestramp_file, gyro_file, K, first_frame, last_frame, delta_T=0):
    framestamp = np.loadtxt(framestramp_file, dtype='float_', delimiter=' ')
    gyro = np.loadtxt(gyro_file, dtype='float_', delimiter=',')

    homs_intra = np.zeros((last_frame - first_frame + 1, patch, 3, 3))
    homs_inter = np.zeros((last_frame - first_frame + 1, patch, 3, 3))

    gyrostamp = gyro[:, -1]
    gyrogap = np.diff(gyrostamp)
    anglev = gyro[:, :3]

    for i in range(len(homs_intra)):
        try:
            hom_intra = compute_intra_homography(i + first_frame, patch, gyrostamp, gyrogap, anglev, framestamp, K)
            hom_inter = imagesHomography(i + first_frame, patch, gyrostamp, gyrogap, anglev, framestamp, K, delta_T)
            homs_intra[i] = hom_intra
            homs_inter[i] = hom_inter
        except AssertionError as e:
            print(e)
            continue
    return homs_intra, homs_inter


def make_gyro_source_cc9(project_path, filename, idx=[0, 0]):
    GYRO_TRAIN_PATH = os.path.join(project_path, "{filename}".format(filename=filename))
    gyro_frames = [idx]
    gyro_homs_inter = []
    gyro_homs_intra = []
    for rotation_frame in gyro_frames:
        print("compute homography between {} and {}".format(rotation_frame[0], rotation_frame[1]))

        gyro_hom_intra, gyro_hom_inter = make_trainset_source(
            GYRO_TRAIN_PATH + '/framestamp.txt',
            GYRO_TRAIN_PATH + '/gyro.txt',
            K_cc9_600_800,
            int(rotation_frame[0]),
            int(rotation_frame[1]),
        )

        # print(gyro_hom_inter.shape)
        gyro_homs_intra.append(gyro_hom_intra)
        gyro_homs_inter.append(gyro_hom_inter)

    gyro_homs_inter_concat = np.vstack(gyro_homs_inter)
    gyro_homs_intra_concat = np.vstack(gyro_homs_intra)
    print("gyro_homs_inter_concat shape", gyro_homs_inter_concat.shape)
    print("gyro_homs_intra_concat shape", gyro_homs_intra_concat.shape)

    np.save(os.path.join(project_path, "{filename}/gyro_homo_h33_source.npy".format(filename=filename)), gyro_homs_inter_concat)
    return gyro_homs_intra_concat, gyro_homs_inter_concat


def image_alignment_with_gyro(gyro_homs_intra, gyro_homos_inter, data_path, idx, gif_path, split="RE"):
    _frame_path = os.path.join(data_path, "reshape")
    frames_path_ois_off = [os.path.join(_frame_path, "{}_frame-{}.jpg".format(split, i)) for i in range(int(idx[0]), int(idx[1]) + 1)]

    match_frames_path_ois_off = [
        os.path.join(_frame_path, "{}_frame-{}.jpg".format(split, i + 1)) for i in range(int(idx[0]),
                                                                                         int(idx[1]) + 1)
    ]

    # RS warping
    # warp_img_make_gif(frames_path_ois_off, match_frames_path_ois_off, gyro_homos_inter, gif_path=gif_path)
    # RS rectify and global warping
    RS_rect_warp_img_make_gif(frames_path_ois_off, match_frames_path_ois_off, gyro_homs_intra, gyro_homos_inter, gif_path=gif_path)


def transformImage(img, hom, patch):
    height, width = img.shape[:2]
    num = height // patch
    out_img = np.zeros((height, width, 3))
    for row in range(hom.shape[0]):
        temp = cv2.warpPerspective(img, hom[row, :, :], (width, height))
        if row == patch - 1:
            out_img[row * num:] = temp[row * num:]
            continue
        out_img[row * num:row * num + num] = temp[row * num:row * num + num]
    return out_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_path')
    parser.add_argument('--filename')
    parser.add_argument('--split')
    parser.add_argument('--idx', nargs='+', required=True)
    args = parser.parse_args()

    data_path = os.path.join(args.project_path, args.filename)
    gif_path = os.path.join(data_path, "gifs")

    gyro_homs_intra, gyro_homs_inter = make_gyro_source_cc9(args.project_path, args.filename, args.idx)
    image_alignment_with_gyro(gyro_homs_intra, gyro_homs_inter, data_path, args.idx, gif_path, args.split)
