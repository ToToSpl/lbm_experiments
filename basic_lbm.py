import numpy as np
from PIL import Image
from tqdm import tqdm
import math
from numba import njit

SIMULATION_WIDTH = 192
SIMULATION_HEIGHT = 108
SIM_DT = 0.05
SIM_DT_TAU = 1 / 2
SIM_TIME = 10.0
SIM_STEPS = int(SIM_TIME / SIM_DT)
SPEED_CNT = 9

STREAM_CENTER = 10
STREAM_START = int(SIMULATION_HEIGHT/2) - STREAM_CENTER
STRAM_END = int(SIMULATION_HEIGHT/2) + STREAM_CENTER + 1
STREAM_VAL = 0.1

'''
----- SPEED DESCRIPTION -----
0 - no movement
1 - down right
2 - down
3 - down left
4 - left
5 - up left
6 - up
7 - up right
8 - right
'''

velocities = np.array([
    [0, 0],
    [-1, 1],
    [-1, 0],
    [-1, -1],
    [0, -1],
    [1, -1],
    [1, 0],
    [1, 1],
    [0, 1]
], dtype=np.float64)
weights = np.array([4/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9])
cs2 = 0.5  # (1/sqrt(2)) ^ 2
reflections = [0, 5, 6, 7, 8, 1, 2, 3, 4]


@njit
def bgk_calc_dumb(speeds):
    '''it is dumb, because implementation is dumb'''
    speeds1d = speeds.reshape(SPEED_CNT)
    # find density
    ro = np.sum(speeds1d)
    # find first moment
    if ro > 0.0:
        u = np.sum(np.multiply(
            velocities, speeds1d.reshape(SPEED_CNT, 1)), axis=0) / ro
    else:
        u = np.array([0.0, 0.0])
    # calc f_eq
    f_eq = np.zeros(SPEED_CNT)
    element3 = np.dot(u, u) / (2*cs2)
    for i in range(SPEED_CNT):
        element1 = np.dot(u, velocities[i]) / cs2
        element2 = element1**2 / 2
        f_eq[i] = weights[i] * ro * (1 + element1 + element2 + element3)

    omega = - (speeds1d - f_eq) * SIM_DT_TAU
    return omega.reshape(1, 1, SPEED_CNT)


@njit
def calculate_speed(speeds):
    ro = np.sum(speeds.reshape(SPEED_CNT))
    u = np.sum(np.multiply(velocities, speeds.reshape(SPEED_CNT, 1)), axis=0) / ro
    return np.linalg.norm(u)


@njit
def collision_step(space):
    dim = space.shape
    for i in range(0, dim[0]):
        for j in range(0, dim[1]):
            space[i, j, :] = space[i, j, :] + bgk_calc_dumb(space[i, j, :])
    return space


def save_data(s, filename):
    dim = s.shape
    img = np.zeros((dim[0], dim[1]))
    for i in range(0, dim[0]):
        for j in range(0, dim[1]):
            img[i, j] = calculate_speed(s[i, j, :])
    img /= np.max(img)
    # img = np.sqrt(img)
    img *= 255
    img = img.astype(np.uint8)
    Image.fromarray(img).save(filename)


def streaming_step(s):
    for i in range(1, SPEED_CNT):
        s[:, :, i] = np.roll(s[:, :, i], int(velocities[i][0]), axis=0)
        s[:, :, i] = np.roll(s[:, :, i], int(velocities[i][1]), axis=1)

    # s[:, 0, :] = 0
    # s[:, -1, :] = 0
    # s[0, :, :] = 0
    # s[-1, :, :] = 0
    # s[STREAM_START:STRAM_END, 0, 8] = STREAM_VAL
    return s


def lbm_basic():

    # set speed buffers
    space = np.ones((SIMULATION_HEIGHT, SIMULATION_WIDTH,
                     SPEED_CNT), dtype=np.float64)
    # set initial velocities
    # space[:, :, 0] = STREAM_VAL
    space[:, :, 8] = STREAM_VAL * \
        (0.9 + 0.1 * np.random.randn(SIMULATION_HEIGHT, SIMULATION_WIDTH))
    # space[:, :, 0] = STREAM_VAL * \
    # (0.5 + 0.5 * np.random.randn(SIMULATION_HEIGHT, SIMULATION_WIDTH))
    # space[STREAM_START:STRAM_END, 0, 8] = 100
    for i in tqdm(range(SIM_STEPS)):
        space = collision_step(space)
        space = streaming_step(space)
        save_data(space, "data/exp_"+str(i)+".png")


if __name__ == "__main__":
    lbm_basic()
