import cv2
import numpy as np

path = '/gqcnn/data/examples/clutter/phoxi/dex-net_4.0/depth_1.npy'
np_img = np.load(path)

# path = '/home/tidy/PycharmProjects/icra2024_baseline/data/test_img/depth2.png'
# np_img = cv2.imread(path)
print(np_img[100][100],np_img.shape)
# image = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
cv2.imwrite("OpenCV image", np_img)
cv2.waitKey(0)