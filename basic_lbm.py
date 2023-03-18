import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm
from numba import njit

THREAD_NUM = 8
SIMULATION_WIDTH = 400
SIMULATION_HEIGHT = 100
SIM_DT = 1.0
SIM_TAU = 0.7
SIM_DT_TAU = SIM_DT / SIM_TAU
SIM_STEPS = 8000  # 1000
SPEED_CNT = 9
CS2 = np.power(1.0 / np.sqrt(3), 2)

STREAM_CENTER = 10
STREAM_START = int(SIMULATION_HEIGHT/2) - STREAM_CENTER
STRAM_END = int(SIMULATION_HEIGHT/2) + STREAM_CENTER + 1
STREAM_VAL = 0.1

SMOKE_SPAWN_AMOUNT = 10
SMOKE_SPAWN_1 = (SIMULATION_HEIGHT * (1 / 4), 0, 0)
SMOKE_SPAWN_2 = (SIMULATION_HEIGHT * (3 / 4), 0, 1)


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

# ones and 3 to right
# INLET_SPEED = np.array([0.0, -0.18181818], dtype=np.float32)
INLET_SPEED = np.array([0.0, -0.1], dtype=np.float32)


@ njit
def bgk_calc_dumb(speeds):
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
            (1 + element1 / CS2 + element2 / (2 * CS2**2) - element3 / (2 * CS2))

    omega = - (speeds1d - f_eq) * SIM_DT_TAU
    return omega.reshape(1, 1, SPEED_CNT)


@ njit
def calculate_speed(speeds):
    ro = np.sum(speeds.reshape(SPEED_CNT))
    if ro == 0.0:
        return np.zeros((2), dtype=np.float32)
    u = np.sum(np.multiply(velocities, speeds.reshape(SPEED_CNT, 1)), axis=0) / ro
    return u


@ njit
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
            u = calculate_speed(s[i, j, :])
            ux[i, j], uy[i, j] = u[1], u[0]
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


def save_u_mag(vecs, filename):
    mags = np.linalg.norm(vecs, axis=2)
    mags = 255.0 / (np.sqrt(CS2)) * mags
    mags = mags.astype(np.uint8)
    Image.fromarray(mags).save(filename)


def streaming_step(s):
    for i in range(1, SPEED_CNT):
        s[:, :, i] = np.roll(s[:, :, i], int(velocities[i][0]), axis=0)
        s[:, :, i] = np.roll(s[:, :, i], int(velocities[i][1]), axis=1)
    return s


def obstacle_step(space, cylinder):
    normal = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    mirror = [0, 5, 6, 7, 8, 1, 2, 3, 4]
    # cylinder
    bndryC = space[cylinder, :]
    bndryC = bndryC[:, mirror]
    space[cylinder, :] = bndryC
    # walls up, down (bounce back)
    space[0, :, normal] = space[0, :, mirror]
    space[-1, :, normal] = space[-1, :, mirror]
    # inlet walls left, right (anti bounce back)
    for i in range(0, SIMULATION_HEIGHT):
        rho_b1 = np.sum(space[i, 0, :])
        rho_b2 = np.sum(space[i, 1, :])
        rho_w = rho_b1 + 0.5 * (rho_b1 - rho_b2)
        for j in range(0, SPEED_CNT):
            space[i, 0, j] = space[i, 0, mirror[j]] - 2.0 * rho_w * weights[j] * \
                (np.dot(velocities[j], INLET_SPEED) / CS2)
        rho_b1 = np.sum(space[i, -1, :])
        rho_b2 = np.sum(space[i, -2, :])
        rho_w = rho_b1 + 0.5 * (rho_b1 - rho_b2)
        for j in range(0, SPEED_CNT):
            space[i, -1, j] = space[i, 0, mirror[j]] - 2.0 * rho_w * weights[j] * \
                (np.dot(velocities[j], INLET_SPEED) / CS2)
    return space


def save_lattice(name, space):
    np.save(name, space)


def generate_vector_field(space, cylinder):
    vec = np.zeros((space.shape[0], space.shape[1], 2))

    for i in range(velocities.shape[0]):
        vec[:, :, 0] += velocities[i, 0] * space[:, :, i]
        vec[:, :, 1] += velocities[i, 1] * space[:, :, i]
    ro = np.sum(space, axis=2)
    vec[:, :, 0] /= ro
    vec[:, :, 1] /= ro
    vec[cylinder, :] = 0.0
    return vec


def save_vector_field_plot(name, vec, res=4):
    res = 4
    vp = np.zeros((vec.shape[0] // res, vec.shape[1] // res, 2))
    for y in range(0, vec.shape[0], res):
        for x in range(0, vec.shape[1], res):
            vp[y//res, x//res, 0] = np.sum(vec[y:y+res, x:x+res, 0]) / res**2
            vp[y//res, x//res, 1] = np.sum(vec[y:y+res, x:x+res, 1]) / res**2
    vp /= 100  # vp.max()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_aspect(space.shape[0] / space.shape[1])
    circ = plt.Circle((SIMULATION_WIDTH/(4*res), SIMULATION_HEIGHT /
                       (2*res)), SIMULATION_HEIGHT/(4*res), color='g')
    ax.add_patch(circ)
    ax.quiver(vp[:, :, 1], vp[:, :, 0])
    fig.savefig(name, dpi=300)
    plt.close()


@ njit
def update_smoke(vecs, smoke, steps=1):
    new_smoke = []
    for i in range(len(smoke)):
        v = vecs[int(smoke[i][0]), int(smoke[i][1])]
        s = (smoke[i][0] + v[0] * steps, smoke[i]
             [1] + v[1] * steps, smoke[i][2])
        if (s[0] < 0 or s[0] > SIMULATION_HEIGHT or s[1] < 0 or s[1] > SIMULATION_WIDTH):
            continue

        new_smoke.append(s)

    return new_smoke


def draw_smoke(filename, smoke):
    img = np.zeros((SIMULATION_HEIGHT, SIMULATION_WIDTH, 3))

    for s in smoke:
        if s[2]:
            img[int(s[0]), int(s[1])] += [80, 0, 0]
        else:
            img[int(s[0]), int(s[1])] += [0, 0, 80]

    im = Image.fromarray(img.astype(np.uint8))
    draw = ImageDraw.Draw(im)
    draw.ellipse(
        (SIMULATION_WIDTH/4 - SIMULATION_HEIGHT/4,
         SIMULATION_HEIGHT/2 - SIMULATION_HEIGHT/4,
         SIMULATION_WIDTH/4 + SIMULATION_HEIGHT/4,
         SIMULATION_HEIGHT/2 + SIMULATION_HEIGHT/4),
        fill=(0, 255, 0), outline=(0, 0, 0))
    im.save(filename)


def lbm_basic():
    X, Y = np.meshgrid(range(SIMULATION_WIDTH), range(SIMULATION_HEIGHT))
    # set speed buffers
    space = np.ones((SIMULATION_HEIGHT, SIMULATION_WIDTH,
                     SPEED_CNT), dtype=np.float32)
    space += 0.01 * \
        np.random.randn(SIMULATION_HEIGHT, SIMULATION_WIDTH, SPEED_CNT)
    # set initial velocities
    # space[:, :, 8] += 0.5  # * (1+0.2*np.cos(2*np.pi*X/SIMULATION_WIDTH*4))
    # rho = np.sum(space, 2)
    # for i in range(SPEED_CNT):
    #     space[:, :, i] *= 100 / rho
    # add cylinder
    cylinder = (X - SIMULATION_WIDTH/4)**2 + \
        (Y - SIMULATION_HEIGHT/2)**2 < (SIMULATION_HEIGHT/4)**2

    smoke_spawn_particles_red = []
    smoke_spawn_particles_blue = []
    for y in range(0, SIMULATION_HEIGHT//2, 16):
        for i in range(4):
            if y + i >= SIMULATION_HEIGHT//2:
                continue
            smoke_spawn_particles_red.append(
                (float(SIMULATION_HEIGHT//2 + y+i), 2.0, 0))
            smoke_spawn_particles_blue.append(
                (float(SIMULATION_HEIGHT//2 - y-i), 2.0, 1))

    smoke = []
    smoke += smoke_spawn_particles_red
    smoke += smoke_spawn_particles_blue

    for i in tqdm(range(SIM_STEPS + 1)):
        space = collision_step(space)
        space = obstacle_step(space, cylinder)
        space = streaming_step(space)

        # vecs = generate_vector_field(space, cylinder)
        # smoke = update_smoke(vecs, smoke)

        if i % 10 == 0:
            smoke += smoke_spawn_particles_red
            smoke += smoke_spawn_particles_blue
            vecs = generate_vector_field(space, cylinder)
            smoke = update_smoke(vecs, smoke, steps=10)
            if i % 20 != 0:
                continue

            # print(np.average(np.sum(space[1:-1, 1:-1, :], axis=2)))
            draw_smoke("data/smoke/"+str(i)+".png", smoke)
            save_vector_field_plot("data/vectors/"+str(i)+".png", vecs)

            save_u_mag(vecs, "data/u_mag/"+str(i)+".png")

            img = gen_data(space, cylinder)
            save_data(img, "data/images/"+str(i)+".png")


if __name__ == "__main__":
    lbm_basic()
