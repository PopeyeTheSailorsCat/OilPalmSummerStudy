import os
import random

random.seed()
PATH_TO_MASK = r'F:/GitFiles/oil_palm_data/aug_data/mask/'
PATH_TO_IMG = r'F:/GitFiles/oil_palm_data/aug_data/src/'
i_mask = lambda num: PATH_TO_MASK + str(num) + '.png'
i_img = lambda num: PATH_TO_IMG + str(num) + '.jpg'
print(i_img(21))
print(i_mask(15))
temp_img = 'TEMP_NAME.jpg'
temp_mask = 'TEMP_NAME.png'
for y in range(2):
    for i in range(1,15000):
        change = random.randint(1, 36000)
        if i == change:
            continue
        os.rename(i_img(i), PATH_TO_IMG + temp_img)
        os.rename(i_mask(i), PATH_TO_MASK + temp_mask)

        os.rename(i_img(change), i_img(i))
        os.rename(i_mask(change), i_mask(i))

        os.rename(PATH_TO_IMG + temp_img, i_img(change))
        os.rename(PATH_TO_MASK + temp_mask, i_mask(change))
