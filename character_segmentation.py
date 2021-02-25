
import argparse
import os
from random import random
from traceback import print_exc
import numpy as np
import cv2


def _get_boxes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, img_bin) = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    img_bin = 255 - img_bin
    v_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(img_bin, v_ker)
    contours, hierarchy = cv2.findContours(
        dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    table_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        bbox = [x, y, x + w, y + h]
        if w * h >= 0.5 * (img.shape[0] * img.shape[1]):
            continue
        if w > 100 and h > 100:  # Minimum area check
            table_boxes.append(bbox)
    return table_boxes


def _save_images(img, boxes, output_dir, prefix=""):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # expected_width = 128
    # expected_height = 128
    # print(len(boxes))
    for i, bbox in enumerate(boxes):
        # print(b)
        # print("i value is",i)
        # print("bbox value is", bbox)
        l, t, r, b = bbox
        # print(l, t, r, b)
        # cv2.imshow("sdfds",bbox)
        chip = img[t:b, l:r]
        chipgray = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)

        # cv2.imshow("chip", chip)
        # cv2.imshow("chipgray", chipgray)

        # cv2.waitKey(0)

        # print(chip.shape[0], chip.shape[1])
        # newimg=chip[0:100,100:200]

        # newimg=chip[t:b,l:r]
        # print(chip)
        # width, height = chip.shape[1], chip.shape[0]
    # print(width, height)
        # crop_width = expected_width if expected_width < chip.shape[1] else chip.shape[1]
        # crop_height = expected_height if expected_height < chip.shape[0] else chip.shape[0]
        # mid_x, mid_y = int(width/2), int(height/2)
        # cw2, ch2 = int(crop_width/2), int(crop_height/2)
        # cropped_image = chip[100:200, mid_x-cw2:mid_x+cw2]
        # a=cv2.imread()
        # print(chip.shape[0], chip.shape[1])

        # following line of code is for inverting the image while saving in the directory
        # newimage=chip[]
        chipblur = cv2.GaussianBlur(chipgray, (5, 5), 1)

        imgCanny = cv2.Canny(chipblur, 10, 250)

        # img_invert = cv2.bitwise_not(thresh)

        # cv2.imshow("threshimage",img_invert)
        # cv2.waitKey(0)
        # print(thresh)
        cnts, hierarchies = cv2.findContours(
            imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # idx = 0
        # cv2.imshow("cnts image is this ",cnts)
        # cv2.waitKey(0)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            # print(x, y, w, h)
            # print(c)
            if w > 10 and h > 10:
                # idx += 1
                new_img = chip[y-5:y+h+5, x-5:x+w+5]
                img_resize = cv2.resize(new_img, (128, 128))

                gray_image = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

                ret, thresh = cv2.threshold(
                    gray_image, 150, 255, cv2.THRESH_BINARY_INV)
                # img_resize = cv2.resize(new_img, (64, 64))

                # img_invert = cv2.bitwise_not(gray_image)
                kernel = np.ones((3, 3), np.uint8)
                img_dilated = cv2.dilate(thresh, kernel, iterations=1)

                cv2.imwrite(os.path.join(output_dir, prefix +
                                         "_" + str(i) + ".jpg"), img_dilated)

                # cropping images
                # cv.imshow("img", new_img)
                # cv.imwrite("cropped/"+str(idx) + '.jpg', new_img)
        # the following code is to be untouched yrr


def segment_characters(filename, output_dir, prefix=""):
    try:
        img = cv2.imread(filename)
        boxes = _get_boxes(img)
        # print(boxes[0])
        # first = cv2.imread(boxes[0])
        # print("first image is ", first)
        # cv2.imshow("first image", first)

        # print(len(boxes))
        # for i in boxes:
        #     width, height = img.shape[0], img.shape[1]
        #     print(width, height)

        os.makedirs(output_dir, exist_ok=True)
        _save_images(img, boxes, output_dir, prefix=prefix)
        return True
    except Exception as _:
        print_exc()


def _list_images(directory, shuffle=False):
    """
    Generic function to list images from directory or file
    :return:  list of images in a given directory or file
    """
    _ext = ['.jpg', '.jpeg', '.bmp', '.png',
            '.JPG', '.JPEG', '.ppm', '.pgm', '.webp']
    _images = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            if file_path.endswith(tuple(_ext)):
                _images.append(file_path)
    if shuffle:
        random.shuffle(_images)
    return _images


def _is_dir(path):
    return os.path.isdir(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Character segmentation")
    parser.add_argument(
        '--input', help="Input image file or directory", default=None)
    parser.add_argument('--output', help="Output directory", default="tmp/")
    parser.add_argument('--shuffle', help="Shuffle images", default=False)
    args = parser.parse_args()
    if args.input is None:
        print("Input image file or directory to process!")
        print("Example: python character_segmentation.py --input image.jpg --output tmp/")

        exit(0)

    if _is_dir(args.input):
        images = _list_images(args.input, args.shuffle)
    else:
        images = [args.input]
    for i, filename in enumerate(images):
        success = segment_characters(
            filename, args.output, prefix=filename[-7]+filename[-6]+filename[-5]+"_"+str(i + 1))
        if not success:
            print("Unable to process the file: {}".format(filename))
        if i > 0 and i % 10 == 0:
            print("Processed {} files: ".format(i))
    print("Processed total {} files".format(len(images)))
    print("Done.")
