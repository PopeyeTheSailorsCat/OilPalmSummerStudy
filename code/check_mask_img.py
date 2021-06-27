import cv2
from matplotlib import pyplot as plt

import albumentations as A


def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        # mask = cv2.cvtColor(mask, cv2.IMWRITE)
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask, vmin=0, vmax=3)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask, vmin=0, vmax=3)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
    plt.show()


image = cv2.imread('train_val_data/image/0012.jpg')
mask = cv2.imread('train_val_data/label/0012.png',cv2.IMREAD_GRAYSCALE)
new_image = cv2.imread('aug_data/src/1125.jpg')
new_mask = cv2.imread('aug_data/mask/1125.png', cv2.IMREAD_GRAYSCALE)
visualize( new_image, new_mask,original_image=image, original_mask=mask)
