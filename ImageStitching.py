import cv2
import numpy as np
import argparse


def trim(frame):
    # crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    # crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    # crop left
    elif not np.sum(frame[:, 0]):
        return trim(frame[:, 1:])
    # crop right
    elif not np.sum(frame[:, -1]):
        return trim(frame[:, :-2])
    return frame


def find_good_matches(descriptors, ratio):
    good = []
    for m, n in descriptors:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


ap = argparse.ArgumentParser()
ap.add_argument("-l", "--limage", required=True, help="Path to the left image to form a panorama photo")
ap.add_argument("-r", "--rimage", required=True, help="Path to the right image to form a panorama photo")
args = vars(ap.parse_args())

img1_path = args["limage"]
img2_path = args["rimage"]

img_left = cv2.imread(img1_path, 1)
img_left_bw = cv2.imread(img1_path, 0)
img_right = cv2.imread(img2_path, 1)
img_right_bw = cv2.imread(img2_path, 0)

sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors of the images
kp1, des1 = sift.detectAndCompute(img_right_bw, None)
kp2, des2 = sift.detectAndCompute(img_left_bw, None)

# find the overlapping points between the images with Brute Force Matcher
bfmatcher = cv2.BFMatcher()
matching_descriptors = bfmatcher.knnMatch(des1, des2, k=2)
# the k parameter = 2 gives out the 2 best matches for each descriptor and returns a list of lists
# where each sublist contains k objects'

good_matches = find_good_matches(matching_descriptors, 0.3)

'''To compute homography we need at least 4 matches'''
MIN_MATCH_COUNT = 4
if len(good_matches) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    height, width = img_right_bw.shape
    pts = np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    dst = cv2.warpPerspective(img_right, M, (img_left.shape[1] + img_right.shape[1], img_left.shape[0]))

    dst[0:img_left.shape[0], 0:img_left.shape[1]] = img_left
    cv2.imshow("final", trim(dst))
    cv2.imwrite("panorama.png", trim(dst))
    cv2.waitKey()
else:
    print("Not enough matches are found - %d/%d" % (len(good_matches), MIN_MATCH_COUNT))
