import skimage.io as skio
import os
from skimage.color import rgb2gray
import scipy.misc
from sklearn.preprocessing import Binarizer

# filename = os.path.join(skio.data_dir, 'eye.png')
moon = skio.imread("viva.jpeg")
skio.imshow(moon)
moon.shape
gray = rgb2gray(moon)
gray.shape

binary = Binarizer().fit(gray)
transformer = Binarizer(.4).fit(gray)  # fit does nothing.
transformer

gray1=transformer.transform(gray)

skio.imshow(gray1)
gray1[200].sum()
scipy.misc.imsave('abc.jpeg',gray1)
