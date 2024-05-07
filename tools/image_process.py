import cv2
from autolab_core import ColorImage, BinaryImage
import numpy as np
from tools.depth_process import *

# parameters, change them, test and optimize for your use case
BINARY_IM_MAX_VAL = np.iinfo(np.uint8).max
BINARY_IM_DEFAULT_THRESH = BINARY_IM_MAX_VAL / 2
LOW_GRAY = 70
UPPER_GRAY = 250
AREA_THRESH_DEFAULT = 800 # larger -> more pruned
DIST_THRESH_DEFAULT = 300 # smaller -> more pruned
FRAME = 'kinect' # in my case is realsense_overhead

def MOG2(background, image, varThreshold, detectShadows):
    backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=varThreshold, detectShadows=detectShadows)
    fgMask = backSub.apply(background)
    fgMask = backSub.apply(image)
    fgMask[np.where(fgMask != 0)] = 255
    return fgMask

def floodfill(fgMask, total_pix):
    fgMask_floodfill = fgMask.copy()
    fgMask_floodfill_inv = cv2.bitwise_not(fgMask_floodfill)
    h, w = fgMask_floodfill_inv.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(fgMask_floodfill_inv, mask, (0, 0), 0)
    if np.count_nonzero(fgMask_floodfill_inv > 0) / total_pix * 100.0 < 20:
        fgMask = fgMask | fgMask_floodfill_inv

    return fgMask


def binary_segmask(background, image, output_dir, filename, USE_ABS=False, USE_MOG2=False):
    """
    Create a binary image from the color image
    :param image: rgb image created with RGBD camera
    :param background: path to background image to use
    
    :param output_dir: path to output dir for processed images
    :param filename: name for saving data
    :return: binary_subtract_pruned
    """
    # print(background[360,640])
    # convert img to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    if USE_ABS:
        image = 255 - image[:, :, 0]
        # (thresh, im_bw) = cv2.threshold(image, cut, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thresh = 255 - 240
        im_bw = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite('%s/%s_binary.png' % (output_dir, filename), im_bw)
        cv2.imshow("segmask", im_bw)
    elif USE_MOG2:
        # get foreground
        fgMask = MOG2(background.copy(), image.copy(), varThreshold=32, detectShadows=True)
        total_pix = fgMask.shape[0] * fgMask.shape[1]

        seg_percentage = np.count_nonzero(fgMask > 0) / total_pix * 100.0
        # cv2.imshow(f"image process test1", fgMask)

        if not seg_percentage > 10:
            if seg_percentage < 1:
                fgMask = MOG2(background.copy(), image.copy(), varThreshold=2, detectShadows=True)
            elif seg_percentage < 2:
                fgMask = MOG2(background.copy(), image.copy(), varThreshold=4, detectShadows=True)
            elif seg_percentage < 5:
                fgMask = MOG2(background.copy(), image.copy(), varThreshold=8, detectShadows=True)
            else:
                fgMask = MOG2(background.copy(), image.copy(), varThreshold=16, detectShadows=True)
            # cv2.imshow(f"image process test1 with lower Threshold: {seg_percentage}", fgMask)

            # fill in empty space
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
            fgMask = floodfill(fgMask, total_pix)
            # cv2.imshow("image process test2", fgMask)

        # prune
        subtract_im_filtered = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2RGB)
        # print(subtract_im_filtered.shape)
        # print(image.shape, background.shape)
        subtract_im_filt = ColorImage(subtract_im_filtered, FRAME)
        # convert to BinaryImage
        binary_subtract = subtract_im_filt.to_binary(threshold=BINARY_IM_DEFAULT_THRESH)
        # Prune contours
        binary_subtract_pruned = binary_subtract.prune_contours(area_thresh=AREA_THRESH_DEFAULT,
                                                                dist_thresh=DIST_THRESH_DEFAULT)
        fgMask = floodfill(binary_subtract_pruned._image_data(), total_pix)
        # cv2.imshow("image process test3", fgMask)

        try:
            cv2.imwrite('%s/%s_binary.png' % (output_dir, filename), fgMask)
        except:
            print("Segmask not captured")
            return False
    else:
        min_val = 350
        max_val = 800
        background[background < min_val] = min_val
        background[background > max_val] = max_val
        image[image < min_val] = min_val
        image[image > max_val] = max_val
        # print('---------------------------------------------------------------------------------------------',
        #       np.min(background), np.max(background), np.min(image), np.max(image))

        # subtract foreground from background
        subtract = background - image
        subtract[subtract < min_val] = min_val
        # subtract[subtract > 255] = 255
        # print('---------------------------------------------------------------------------------------------',
        #       np.min(subtract), np.max(subtract))
        subtract = inpaint_depth(normalize_depth(subtract))
        subtract = cv2.cvtColor(subtract, cv2.COLOR_BGR2GRAY)

        # in range for grayscale values
        subtract_im_filtered = cv2.inRange(subtract, LOW_GRAY, UPPER_GRAY)
        # cv2.imshow("image process test", subtract_im_filtered)
        cv2.waitKey(1)
        # open as ColorImage
        subtract_im_filtered = cv2.cvtColor(subtract_im_filtered, cv2.COLOR_GRAY2RGB)
        # print(subtract_im_filtered.shape)
        # print(image.shape, background.shape)
        subtract_im_filt = ColorImage(subtract_im_filtered, FRAME)
        # convert to BinaryImage
        binary_subtract = subtract_im_filt.to_binary(threshold=BINARY_IM_DEFAULT_THRESH)
        # Prune contours
        binary_subtract_pruned = binary_subtract.prune_contours(area_thresh=AREA_THRESH_DEFAULT, dist_thresh=DIST_THRESH_DEFAULT)
        # save binary to npy and png format
        # np.save('%s/%s_binary.npy' % (output_dir, filename), binary_subtract_pruned._image_data())

        try:
            cv2.imwrite('%s/%s_binary.png' % (output_dir, filename), binary_subtract_pruned._image_data())
        except:
            print("Segmask not captured")
            return False

    return True

if __name__ == "__main__":
    image = cv2.imread("/home/tidy/PycharmProjects/icra2024_baseline/data/test_img/depth_img4.png") # background
    binary_segmask(image,
                   "/home/tidy/PycharmProjects/icra2024_baseline/data/test_img/depth_img3.png",  # object
                   "/home/tidy/PycharmProjects/icra2024_baseline/result/test_img", "segmask")