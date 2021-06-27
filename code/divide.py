import os

PATH_TO_MASK = r'F:/GitFiles/oil_palm_data/aug_data/mask/'
PATH_TO_IMG = r'F:/GitFiles/oil_palm_data/aug_data/src/'
SAVE_IMG = r'F:/GitFiles/oil_palm_data/val_data/val_src/'
SAVE_MASK = r'F:/GitFiles/oil_palm_data/val_data/val_label/'
i_mask = lambda path, num: path + 'val_label' + str(num) + '.png'
true_mask = lambda path, num: path + str(num) + '.png'
i_img = lambda path, num: path + str(num) + '.jpg'
for i in range(1, 36001, 4):
    # os.rename(i_img(PATH_TO_IMG, i), i_img(SAVE_IMG, i))
    os.rename(i_mask(SAVE_MASK, i), true_mask(SAVE_MASK, i))
