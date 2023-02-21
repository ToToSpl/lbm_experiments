import imageio
from glob import glob

PATH_VORTEX = "data/images/*.png"
PATH_SMOKE = "data/smoke/*.png"
PATH_VECTORS = "data/vectors/*.png"


def generate_gif(images_path, output_path, duration=0.5):
    imgs = glob(images_path)
    imgs = sorted(imgs, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    with imageio.get_writer(output_path, mode='I', duration=duration) as writer:
        for img in imgs:
            image = imageio.imread(img)
            writer.append_data(image)


def main():
    generate_gif(PATH_VORTEX, "./data/vortex.gif", duration=1/30)
    generate_gif(PATH_SMOKE, "./data/smoke.gif", duration=1/30)
    generate_gif(PATH_VECTORS, "./data/vectors.gif", duration=1/30)


if __name__ == "__main__":
    main()
