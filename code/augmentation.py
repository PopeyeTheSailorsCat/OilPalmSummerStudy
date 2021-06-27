import random

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

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
    plt.show()


image = cv2.imread('train_val_data/image/0014.jpg')
mask = cv2.imread('train_val_data/label/0014.png', cv2.IMREAD_GRAYSCALE)
visualize(image, mask)
original_height = 600
original_width = 600

W = 256
H = 256

hard_aug = A.Compose([
    # A.RandomCrop(height=H, width=W, p=0.8),
    # A.RandomSizedCrop(min_max_height=(H, W), height=H, width=W, p=0.8),
    # A.OneOf([
    # A.RandomSizedCrop(min_max_height=(50, 101), height=original_height, width=original_width, p=0.5),
    A.PadIfNeeded(min_height=original_height, min_width=original_width, p=1),
    A.RandomSizedCrop(min_max_height=(H - 50, W + 50), height=H, width=W, p=1),
    # ], p=1),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.25),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.2, p=1),
        A.GaussNoise(),
    ], p=0.8),
    A.CLAHE(p=0.8),
    A.RandomBrightnessContrast(p=0.8),
    A.RandomGamma(p=0.8)])
random.seed()
augmented = hard_aug(image=image, mask=mask)

image_heavy = augmented['image']
mask_heavy = augmented['mask']

# visualize(image_heavy, mask_heavy, original_image=image, original_mask=mask)

# visualize(image_rot90, mask_rot90, original_image=image, original_mask=mask)

med_aug = A.Compose([
    # A.RandomCrop(height=H, width=W),
    A.PadIfNeeded(min_height=H, min_width=W, p=1),
    A.RandomSizedCrop(min_max_height=(H - 50, W + 50), height=H, width=W, p=1),
    # A.OneOf([

    # A.RandomSizedCrop(min_max_height=(50, 101), height=H, width=W, p=0.5),

    # ], p=1),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.25),
    A.OneOf([
        A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
        A.GaussNoise(),
    ], p=0.8),
])

augmented = med_aug(image=image, mask=mask)

image_medium = augmented['image']
mask_medium = augmented['mask']


def create_num(num):
    if num // 10 == 0:
        return "000" + str(num)
    elif num // 100 == 0:
        return "00" + str(num)
    else:
        return "0" + str(num)


# visualize(image_medium, mask_medium, original_image=image, original_mask=mask)
random.seed()
counter = 0
for x in range(1, 361):
    image = cv2.imread('train_val_data/image/' + create_num(x) + '.jpg')
    # print(image.shape)
    mask = cv2.imread('train_val_data/label/' + create_num(x) + '.png', cv2.IMREAD_GRAYSCALE)
    # visualize(image, mask)
    for y in range(50):
        m_augmented = med_aug(image=image, mask=mask)
        h_augmented = hard_aug(image=image, mask=mask)
        counter += 1
        cv2.imwrite('aug_data/src/' + str(counter) + '.jpg', m_augmented['image'])
        cv2.imwrite('aug_data/mask/' + str(counter) + '.png', m_augmented['mask'])

        counter += 1
        cv2.imwrite('aug_data/src/' + str(counter) + '.jpg', h_augmented['image'])
        cv2.imwrite('aug_data/mask/' + str(counter) + '.png', h_augmented['mask'])
        # if y == 10:
        # visualize(m_augmented['image'],m_augmented['mask'])
        # visualize(h_augmented['image'],h_augmented['mask'])
# image = cv2.imread('aug_data/src/10.jpg')
# mask = cv2.imread('aug_data/mask/10.png', cv2.IMREAD_GRAYSCALE)
# visualize(image, mask)
