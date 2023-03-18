from glob import glob
import numpy as np
import cv2
import imageio.v2 as imageio

PATH_VORTEX = "data/images/*.png"
PATH_SMOKE = "data/smoke/*.png"
PATH_VECTORS = "data/vectors/*.png"
PATH_U_MAG = "data/u_mag/*.png"


def generate_gif(images_path, output_path, duration=0.5, smoke=False):
    imgs = glob(images_path)
    imgs = sorted(imgs, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    with imageio.get_writer(output_path, mode='I', duration=duration) as writer:
        for img in imgs:
            image = imageio.imread(img)
            if smoke:
                image = process_smoke(image)
            writer.append_data(image)


def process_smoke(image):
    kernel = np.ones((5, 5), np.float32)/25
    image = cv2.resize(image, (image.shape[1]*4, image.shape[0]*4))
    image = cv2.filter2D(image, -1, kernel)
    return image


def main():
    generate_gif(PATH_VORTEX, "./data/vortex.gif", duration=1/30)
    generate_gif(PATH_SMOKE, "./data/smoke.gif", duration=1/30, smoke=True)
    generate_gif(PATH_VECTORS, "./data/vectors.gif", duration=1/30)
    generate_gif(PATH_U_MAG, "./data/u_mag.gif", duration=1/30)


if __name__ == "__main__":
    main()
