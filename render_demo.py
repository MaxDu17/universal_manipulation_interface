import imageio
import zarr
import numpy as np

DEMO = "DATA/L.zarr"

array = zarr.open(DEMO, mode = 'r')

imgs = array.data.img

out = imageio.get_writer("Demo_visual.mp4", codec='libx264')

for i in range(imgs.shape[0]):
    img = np.expand_dims(imgs[i], axis=0)
    out.append_data(img)

out.close()