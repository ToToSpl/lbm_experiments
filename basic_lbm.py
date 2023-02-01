import numpy as np
from PIL import Image
from tqdm import tqdm
import math
from numba import njit

SIMULATION_WIDTH = 400
SIMULATION_HEIGHT = 100
SIM_DT_TAU = 1.0 / 0.6
SIM_STEPS = 4000
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
], dtype=np.float32)
weights = np.array([4/9, 1/36, 1/9, 1/36, 1/9, 1/36,
                   1/9, 1/36, 1/9], dtype=np.float32)
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
        u = np.array([0.0, 0.0], dtype=np.float32)
    # calc f_eq
    f_eq = np.zeros(SPEED_CNT, dtype=np.float32)
    element3 = np.dot(u, u)
    for i in range(SPEED_CNT):
        element1 = np.dot(u, velocities[i])
        element2 = element1**2
        f_eq[i] = weights[i] * ro * \
            (1 + 3.0 * element1 + 4.5 * element2 - 1.5 * element3)

    omega = - (speeds1d - f_eq) * SIM_DT_TAU
    return omega.reshape(1, 1, SPEED_CNT)


@njit
def calculate_speed(speeds):
    ro = np.sum(speeds.reshape(SPEED_CNT))
    u = np.sum(np.multiply(velocities, speeds.reshape(SPEED_CNT, 1)), axis=0) / ro
    return u[1], u[0]


@njit
def collision_step(space):
    dim = space.shape
    for i in range(0, dim[0]):
        for j in range(0, dim[1]):
            space[i, j, :] = space[i, j, :] + bgk_calc_dumb(space[i, j, :])
    return space


def gen_data(s, cylinder):
    dim = s.shape
    ux = np.zeros((dim[0], dim[1]), dtype=np.float32)
    uy = np.zeros((dim[0], dim[1]), dtype=np.float32)
    for i in range(0, dim[0]):
        for j in range(0, dim[1]):
            ux[i, j], uy[i, j] = calculate_speed(s[i, j, :])
    vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - \
        (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
    vorticity[cylinder] = np.nan
    vorticity = np.ma.array(vorticity, mask=cylinder)
    vorticity = 2550 * np.clip(vorticity, -0.1, 0.1)

    img = np.ones((dim[0], dim[1], 3))
    for i in range(0, dim[0]):
        for j in range(0, dim[1]):
            if vorticity[i, j] > 0.0:
                img[i, j, 2] = vorticity[i, j]
            else:
                img[i, j, 0] = abs(vorticity[i, j])
    # img = np.sqrt(img)
    img = img.astype(np.uint8)
    return img


def save_data(img, filename):
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


def obstacle_step(space, cylinder):
    bndryC = space[cylinder, :]
    bndryC = bndryC[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]
    space[cylinder, :] = bndryC
    return space


def save_lattice(name, space):
    np.save(name, space)


def lbm_basic():
    X, Y = np.meshgrid(range(SIMULATION_WIDTH), range(SIMULATION_HEIGHT))
    # set speed buffers
    space = np.ones((SIMULATION_HEIGHT, SIMULATION_WIDTH,
                     SPEED_CNT), dtype=np.float32)
    space += 0.01 * \
        np.random.randn(SIMULATION_HEIGHT, SIMULATION_WIDTH, SPEED_CNT)
    # set initial velocities
    space[:, :, 8] += 2 * (1+0.2*np.cos(2*np.pi*X/SIMULATION_WIDTH*4))
    rho = np.sum(space, 2)
    for i in range(SPEED_CNT):
        space[:, :, i] *= 100 / rho
    # add cylinder
    cylinder = (X - SIMULATION_WIDTH/4)**2 + \
        (Y - SIMULATION_HEIGHT/2)**2 < (SIMULATION_HEIGHT/4)**2

    for i in tqdm(range(SIM_STEPS)):
        space = collision_step(space)
        space = streaming_step(space)
        space = obstacle_step(space, cylinder)

        # if i % 10 == 0:
        #     img = gen_data(space, cylinder)
        #     save_data(img, "data/images/exp_"+str(i)+".png")


if __name__ == "__main__":
    lbm_basic()
