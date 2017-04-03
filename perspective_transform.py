import pickle
import cv2
import numpy as np
from os.path import join, basename

# Program local libraries
from load_parameters import load_camera_mtx_dist_from_pickle as load_mtx_dist



# Where are the road test images?
road_test_images_dir = 'test_images' 

# Point to a straight road image here
road_straight_image_filename = 'straight_lines2.jpg'

# Where you want to save warped straight image for check?
road_straight_warped_image_dir = 'output_images/bird_eye_test'

# Where you want to save the transformation matrices (M,Minv)?
M_Minv_output_dir = 'output_images/camera_cal'

# Play with trapezoid ratio until you get the proper bird's eye lane lines projection
# bottom_width = percentage of image width
# top_width = percentage of image width
# height = percentage of image height
# car_hood = number of pixels to be cropped from bottom meant to get rid of car's hood
bottom_width=0.4
top_width=0.092
height=0.4
car_hood=45


# Sort coordinate points clock-wise, starting from top-left
# Inspired by the following discussion:
# http://stackoverflow.com/questions/1709283/how-can-i-sort-a-coordinate-list-for-a-rectangle-counterclockwise
def order_points(pts):
    # Normalises the input into the [0, 2pi] space, added 0.5*pi to initiate from top left
    # In this space, it will be naturally sorted "counter-clockwise", so we inverse order in the return
    mx = np.sum(pts.T[0]/len(pts))
    my = np.sum(pts.T[1]/len(pts))

    l = []
    for i in range(len(pts)):
        l.append(  (np.math.atan2(pts.T[0][i] - mx, pts.T[1][i] - my) + 2 * np.pi + 0.5 * np.pi) % (2*np.pi)  )
    sort_idx = np.argsort(l)
    
    return pts[sort_idx[::-1]]


def get_transform_matrices(pts, img_size):
    # Obtain a consistent order of the points and unpack them individually
    src = order_points(pts)
    
    #Give user some data to check
    print('Here are the ordered src pts:', src)
    
    # Destination points
    dst = np.float32([[src[3][0], 0],
                      [src[2][0], 0],
                      [src[2][0], img_size[1]],
                      [src[3][0], img_size[1]]])
    
    #Give user some data to check
    print('Here are the dst pts:', dst)
    
    # Compute the perspective transform matrix and the inverse of it
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv


# Re-using one of my functions used in the first detection project
# Modified to crop car hood
def trapezoid_vertices(image, bottom_width=0.85,top_width=0.07,height=0.40, car_hood=45):
    """
    Create trapezoid vertices for mask. 
    Inpus:
    image
    bottom_width = percentage of image width
    top_width = percentage of image width
    height = percentage of image height
    car_hood = number of pixels to be cropped from bottom meant to get rid of car's hood
    """   
    imshape = image.shape
    
    vertices = np.array([[\
        ((imshape[1] * (1 - bottom_width)) // 2, imshape[0]-car_hood),\
        ((imshape[1] * (1 - top_width)) // 2, imshape[0] - imshape[0] * height + car_hood),\
        (imshape[1] - (imshape[1] * (1 - top_width)) // 2, imshape[0] - imshape[0] * height + car_hood),\
        (imshape[1] - (imshape[1] * (1 - bottom_width)) // 2, imshape[0] - car_hood)]]\
        , dtype=np.int32)
    
    return vertices



def get_perspective_and_pickle_M_Minv():

    # Optimize source points by using straight road test image
    # Load image
    Readname = join(road_test_images_dir, road_straight_image_filename)
    img = cv2.imread(Readname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Give user some data to check
    print('Here is the straight image shape: ', img.shape)
    
    # Load camera coefficients
    mtx, dist = load_mtx_dist()

    # Undistort and get image size
    img = cv2.undistort(img, mtx, dist, None, mtx)
    img_size = (img.shape[1], img.shape[0])

    # Get the points by image ratios
    pts = trapezoid_vertices(img, bottom_width=bottom_width,top_width=top_width,height=height, car_hood=car_hood)
    # Modify it to expected format
    pts = pts.reshape(pts.shape[1:])
    pts = pts.astype(np.float32)

    # Give user some data to check
    print('Here are the initial src pts:', pts)

    # get the transform matrices
    M, Minv = get_transform_matrices(pts, img_size)

    # transform image and save it
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    write_name1 = join(road_straight_warped_image_dir, 'Warped_' + basename(Readname) )
    cv2.imwrite(write_name1,warped)

    # Save the transformation matrices for later use
    dist_pickle = {}
    dist_pickle["M"] = M
    dist_pickle["Minv"] = Minv
    write_name2 = join(M_Minv_output_dir,'perspective_trans_matrices.p')
    pickle.dump( dist_pickle, open( write_name2, "wb" ) )
    
    print('Done!')
    print("Warped image test: from [" + basename(Readname)  + "] to [" + basename(write_name1) + "]")
    print("Here is the warped image: [" + write_name1  + "]")
    print("M and Minv saved: [pickled file saved to: " + write_name2  + "]")
    
    
if __name__ == '__main__':
    get_perspective_and_pickle_M_Minv()
    
    
