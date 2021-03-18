import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def deskew(orig_image, ench_image, M):
    print('Affine matrix:', M)
    im_out = cv2.warpPerspective(orig_image, np.linalg.inv(M), (ench_image.shape[1], ench_image.shape[0]))
    return im_out
    
def unwarp(orig, ench):
    '''
        Converts original image to shape and crop like enchanced. 
        Needed if you original images in dataset differs from resulted by crop and shape
    '''
    ench_image = cv2.imread(ench, 0)
    orig_image = cv2.imread(orig, 0)
    orig_image_rgb = cv2.imread(orig)
    # here we are using KAZE image descriptor to find similar points on both images.
    try:
        surf = cv2.KAZE_create()
        kp1, des1 = surf.detectAndCompute(ench_image, None)
        kp2, des2 = surf.detectAndCompute(orig_image, None)
    except cv2.error as e:
        raise e
    

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # if less then 10 points matched -> not the same images or higly distorted 
    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good
                              ]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good
                              ]).reshape(-1, 1, 2)

        kp1_matched=([ kp1[m.queryIdx] for m in good ])
        kp2_matched=([ kp2[m.trainIdx] for m in good ])   

        matches = cv2.drawMatches(ench_image,kp1,orig_image,kp2, good,None, flags=2)
        plt.figure(figsize=(20,10))
        plt.axis('off')
        plt.imshow(matches),plt.show()   
        # Finds a perspective transformation between two planes. 
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # see https://ch.mathworks.com/help/images/examples/find-image-rotation-and-scale-using-automated-feature-matching.html for details
        ss = M[0, 1]
        sc = M[0, 0]
        scaleRecovered = math.sqrt(ss * ss + sc * sc)
        thetaRecovered = math.atan2(ss, sc) * 180 / math.pi
        print("Calculated scale difference: %.2f\nCalculated rotation difference: %.2f" % (scaleRecovered, thetaRecovered))

        return deskew(orig_image_rgb, ench_image, M)
        
    else:
        print("Not  enough  matches are found   -   %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None
        return None


img = unwarp('test_1.jpg', 'test_1_edit.jpg')