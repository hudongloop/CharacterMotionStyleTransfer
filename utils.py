import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def show_image(images, nrow, ncolumn):
    fig = plt.figure()
    npimg = images.add(1.0).mul(127.5).numpy()[0,0,:,:,:]
    for i in xrange(nrow*ncolumn):
        fig.add_subplot(nrow, ncolumn, i + 1)
        img = npimg[:, :, i]
        plt.imshow(img)
    plt.show()


def save_image(images, nrow, ncolumn, filename):
    image_size = images.size(2)
    figure = np.zeros((nrow * image_size, ncolumn * image_size))
    npimg = images.add(1.0).mul(127.5).numpy()[0, 0, :, :, :]
    for i in xrange(nrow * ncolumn):
        img = npimg[:, :, i]
        h1=(i // ncolumn) * image_size
        h2=(i // ncolumn + 1)*image_size
        w1=(i % ncolumn) * image_size
        w2=(i % ncolumn + 1)*image_size
        figure[h1:h2, w1:w2] = img

    figure = figure.astype(np.uint8)
    im = Image.fromarray(figure)
    im.save(filename)


