import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2
import pandas as pd
import argparse
import yaml
import datetime

def decompose_projection_matrix(p):
    '''
    Shortcut to use cv2.decomposeProjectionMatrix(), which only returns k, r, t, and divides
    t by the scale, then returns it as a vector with shape (3,) (non-homogeneous)
    
    Arguments:
    p -- projection matrix to be decomposed
    
    Returns:
    k, r, t -- intrinsic matrix, rotation matrix, and 3D translation vector
    
    '''

    # Free implementation provided! You may use this function as is.
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = (t / t[3])[:3]
    
    return k, r, t


def compute_left_disparity_map(img_left, img_right, matcher='bm', rgb=False, verbose=False):
    '''
    Takes a left and right stereo pair of images and computes the disparity map for the left
    image. Pass rgb=True if the images are RGB.
    
    Arguments:
    img_left -- image from left camera
    img_right -- image from right camera
    
    Optional Arguments:
    matcher -- (str) can be 'bm' for StereoBM or 'sgbm' for StereoSGBM matching
    rgb -- (bool) set to True if passing RGB images as input
    verbose -- (bool) set to True to report matching type and time to compute
    
    Returns:
    disp_left -- disparity map for the left camera image
    
    '''
    # Free implementation provided! You may use this function as is.

    # Feel free to read OpenCV documentation and tweak these values. These work well
    sad_window = 6
    num_disparities = sad_window*16
    block_size = 11
    matcher_name = matcher
    
    if matcher_name == 'bm':
        matcher = cv2.StereoBM_create(numDisparities=num_disparities,
                                      blockSize=block_size
                                     )
        
    elif matcher_name == 'sgbm':
        matcher = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                        minDisparity=0,
                                        blockSize=block_size,
                                        P1 = 8 * 3 * sad_window ** 2,
                                        P2 = 32 * 3 * sad_window ** 2,
                                        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                                       )
    if rgb:
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    start = datetime.datetime.now()
    disp_left = matcher.compute(img_left, img_right).astype(np.float32)/16
    end = datetime.datetime.now()
    if verbose:
        print(f'Time to compute disparity map using Stereo{matcher_name.upper()}:', end-start)
    
    return disp_left


def calc_depth_map(disp_left, k_left, t_left, t_right, rectified=True):
    '''
    Calculate depth map using a disparity map, intrinsic camera matrix, and translation vectors
    from camera extrinsic matrices (to calculate baseline). Note that default behavior is for
    rectified projection matrix for right camera. If using a regular projection matrix, pass
    rectified=False to avoid issues.
    
    Arguments:
    disp_left -- disparity map of left camera
    k_left -- intrinsic matrix for left camera
    t_left -- translation vector for left camera
    t_right -- translation vector for right camera
    
    Optional Arguments:
    rectified -- (bool) set to False if t_right is not from rectified projection matrix
    
    Returns:
    depth_map -- calculated depth map for left camera
        
    '''

    focal_length=k_left[0,0]
    baseline=np.linalg.norm(t_left-t_right)
    disp_left[disp_left <= 0] = 0.1
    depth_map=(focal_length*baseline)/disp_left
    return depth_map

   

def stereo_2_depth(img_left, img_right, P0, P1, matcher='bm', rgb=False, verbose=False, 
                   rectified=True):
    '''
    Takes stereo pair of images and returns a depth map for the left camera. If your projection
    matrices are not rectified, set rectified=False.
    
    Arguments:
    img_left -- image of left camera
    img_right -- image of right camera
    P0 -- Projection matrix for the left camera
    P1 -- Projection matrix for the right camera
    
    Optional Arguments:
    matcher -- (str) can be 'bm' for StereoBM or 'sgbm' for StereoSGBM
    rgb -- (bool) set to True if images passed are RGB. Default is False
    verbose -- (bool) set to True to report computation time and method
    rectified -- (bool) set to False if P1 not rectified to P0. Default is True
    
    Returns:
    depth -- depth map for left camera
    
    '''
    disp_map = compute_left_disparity_map(img_left, img_right, matcher=matcher, rgb=rgb, verbose=verbose)
    K0, _, t0 = decompose_projection_matrix(P0)
    _,  _, t1 = decompose_projection_matrix(P1)
    depth = calc_depth_map(disp_map, K0, t0, t1, rectified=rectified)
    return depth



def extract_features(image, detector='sift', mask=None):
    """
    Find keypoints and descriptors for the image

    Arguments:
    image -- a grayscale image

    Returns:
    kp -- list of the extracted keypoints (features) in an image
    des -- list of the keypoint descriptors in an image
    """
    if detector == 'sift':
        feat_detectn_method=cv2.SIFT_create()
    else :
        print("invalid input for feature detection method")

    kp , des = feat_detectn_method.detectAndCompute(image , mask)

    return kp , des

def match_features(des1, des2, matching='BF', detector='sift', sort=True, k=2):
    """
    Match features from two images
    
    Note : you need not implement both Brute Force and FLANN matching. Choose one of them. Although, it is recommended to learn using both.

    Arguments:
    des1 -- list of the keypoint descriptors in the first image
    des2 -- list of the keypoint descriptors in the second image
    matching -- (str) can be 'BF' for Brute Force or 'FLANN'
    detector -- (str) can be 'sift or 'orb'. Default is 'sift'
    sort -- (bool) whether to sort matches by distance. Default is True
    k -- (int) number of neighbors to match to each feature.

    Returns:
    matches -- list of matched features from two images. Each match[i] is k or less matches for 
               the same query descriptor
    """
    if matching == 'BF':
        if detector == 'sift':
            norm = cv2.NORM_L2
        else:
            norm = cv2.NORM_HAMMING
        
        matcher = cv2.BFMatcher(norm, crossCheck=False)
        matches = matcher.knnMatch(des1, des2, k=k)


    return matches



def filter_matches_distance(matches, dist_threshold=0.60):
    """
    Filter matched features from two images by distance between the best matches

    Arguments:
    match -- list of matched features from two images
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0) 

    Returns:
    filtered_match -- list of good matches, satisfying the distance threshold
    """
    filtered_matches = []

    for m in matches:
        if len(m) == 2:
            a, b = m
            if a.distance < dist_threshold * b.distance:
                filtered_matches.append(a)

    return filtered_matches


def estimate_motion(match, kp1, kp2, k, depth1=None, max_depth=3000):
    """
    Estimate camera motion from a pair of subsequent image frames

    Arguments:
    match -- list of matched features from the pair of images
    kp1 -- list of the keypoints in the first image
    kp2 -- list of the keypoints in the second image
    k -- camera intrinsic calibration matrix 
    
    Optional arguments:
    depth1 -- Depth map of the first frame. Set to None to use Essential Matrix decomposition
    max_depth -- Threshold of depth to ignore matched features. 3000 is default

    Returns:
    rmat -- estimated 3x3 rotation matrix
    tvec -- estimated 3x1 translation vector
    image1_points -- matched feature pixel coordinates in the first image. 
                     image1_points[i] = [u, v] -> pixel coordinates of i-th match
    image2_points -- matched feature pixel coordinates in the second image. 
                     image2_points[i] = [u, v] -> pixel coordinates of i-th match
               
    """

    # CONVERSION OF 2D MATCHED KEYPOINTS INTO 3D COORDINATES
    cx=k[0,2]
    cy=k[1,2]
    fx=k[0,0]
    fy=k[1,1]


    object_points = []  
    image_points = []   

    image1_points = []
    image2_points = []

    for m in match:
        u1, v1 = kp1[m.queryIdx].pt
        u2, v2 = kp2[m.trainIdx].pt

        u1, v1 = int(round(u1)), int(round(v1))

        if u1 < 0 or v1 < 0 or u1 >= depth1.shape[1] or v1 >= depth1.shape[0]:
            continue

        z = depth1[v1, u1]

        if z <= 0 or z > max_depth:
            continue

        x = (u1 - cx) * z / fx
        y = (v1 - cy) * z / fy

        object_points.append([x, y, z])
        image_points.append([u2, v2])

        image1_points.append([u1, v1])
        image2_points.append([u2, v2])

    object_points = np.array(object_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)

    if len(object_points) < 6:
        print("Not enough points for PnP.")
        return None, None, image1_points, image2_points

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points, image_points, k, None)

    if not success:
        print("PnP failed.")
        return None, None, image1_points, image2_points

    rmat, _ = cv2.Rodrigues(rvec)
    tvec = tvec.reshape(3)

    return rmat, tvec, image1_points, image2_points

    
def get_true_pose(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.eye(4)
            pose[:3, :4] = np.array(values).reshape(3, 4)
            poses.append(pose)
    return poses


def get_camera_matrix(config):
    pass

def save_trajectory_to_csv(trajectory, file_name):
    df = pd.DataFrame(trajectory, columns=['index', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
    df.to_csv(file_name, index=False)


def two_frame_vo(config_path):
    """Main function for stereo visual odometry using only two frames."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Camera projection matrices (hard code it since it doesn't change)
    P0 = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00],
               [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00],
               [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]])

    P1 = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.861448000000e+02],
               [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00],
               [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]])


    images_path = config['images'] # Path to the images
    sequence = str(config['sequence']).zfill(2) # Sequence name
    frame_start = config['frame_start'] # This is the frame number to start with
    frame_end = frame_start + 1 # We are using only two frames

    # Load the images and poses for the two frames using the correct sequence and frame numbers
    path_l_start = os.path.join(images_path, "dataset/sequences", sequence, "image_0", f"{frame_start:06d}.png")
    path_r_start = os.path.join(images_path, "dataset/sequences", sequence, "image_1", f"{frame_start:06d}.png")
    path_l_end   = os.path.join(images_path, "dataset/sequences", sequence, "image_0", f"{frame_end:06d}.png")
    path_r_end   = os.path.join(images_path, "dataset/sequences", sequence, "image_1", f"{frame_end:06d}.png")

    image_l_start = cv2.imread(path_l_start, cv2.IMREAD_GRAYSCALE)
    image_r_start = cv2.imread(path_r_start, cv2.IMREAD_GRAYSCALE)
    image_l_end   = cv2.imread(path_l_end,   cv2.IMREAD_GRAYSCALE)
    image_r_end   = cv2.imread(path_r_end,   cv2.IMREAD_GRAYSCALE)

    trajectory = [[0,0,0,0,0,0,0,1]] # Initial pose
    pose =np.eye(4)


    # Decompose projection matrices 
    K0, _, t0 = decompose_projection_matrix(P0)
    _,  _, t1 = decompose_projection_matrix(P1)
   

    # Compute disparity map for start frames
    disp_map = compute_left_disparity_map(image_l_start, image_r_start, matcher='sgbm')

    # Compute depth map for start frames
    depth_map_start = calc_depth_map(disp_map, K0, t0, t1, rectified=True)
    
    # Extract features from start left frame
    image =image_l_start
    kp1 , des1 =extract_features(image, detector='sift', mask=None)

    # Track features from start left frame to end left frame
    # first extracting features from the end left frame 
    kp2 , des2 =extract_features(image_l_end, detector='sift', mask=None)
    # now tracking them
    rk_matches=match_features(des1, des2, matching='BF', detector='sift', sort=True, k=2)
    rk_good_matches=filter_matches_distance(rk_matches, dist_threshold=0.75)

    # Get 3D points of good matches
    # Estimate motion from 3D points
    rmat, tvec, image1_points, image2_points = estimate_motion(
    rk_good_matches, kp1, kp2, K0, depth_map_start)
    
    
    if rmat is not None and tvec is not None:
        from scipy.spatial.transform import Rotation as R

        T = np.eye(4)
        T[:3, :3] = rmat
        T[:3, 3] = tvec

        pose = pose @ np.linalg.inv(T)  # Update pose

        pos = pose[:3, 3]
        rot = R.from_matrix(pose[:3, :3]).as_quat()  # Quaternion [x, y, z, w]

        trajectory.append([1, pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]])
    else:
        print("pose cannot be updated due to failure of motion estimation ")


    # Get pose of end frame and append to trajectory

    for t in trajectory:
        print(t)


''' the visualization code only for my reference '''
# def plot_trajectories(estimated, ground_truth):
#     est_xyz = np.array([[pose[1], pose[3]] for pose in estimated])  # [x, z] from estimated

#     # Extract only tx and tz from 3x4 ground truth matrices
#     gt_xyz = np.array([[pose[0, 3], pose[2, 3]] for pose in ground_truth])  # [tx, tz] from gt

#     plt.figure(figsize=(10, 6))
#     plt.plot(est_xyz[:, 0], est_xyz[:, 1], label='Estimated', linewidth=2, color='blue')
#     plt.plot(gt_xyz[:, 0], gt_xyz[:, 1], label='Ground Truth', linewidth=2, color='orange')
#     plt.xlabel('X (meters)')
#     plt.ylabel('Z (meters)')
#     plt.title('Trajectory Comparison')
#     plt.legend()
#     plt.grid(True)
#     plt.axis('equal')
#     plt.show()



def full_svo(config_path):
    """Main function for stereo visual odometry."""
    from scipy.spatial.transform import Rotation as R
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    images_path = config['images']
    poses_path = config['poses']
    sequence = str(config['sequence']).zfill(2)
    frame_start = config['frame_start']
    frame_end = config['frame_end']
    csv_path = config.get('csv_path', './output.csv')

    P0 = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00],
               [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00],
               [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]])

    P1 = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.861448000000e+02],
               [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00],
               [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]])

    K0, _, t0 = decompose_projection_matrix(P0)
    _, _, t1 = decompose_projection_matrix(P1)

    pose = np.eye(4)
    trajectory = [[0, 0, 0, 0, 0, 0, 0, 1]]
    pose_file = os.path.join(poses_path, "dataset/poses", f"{sequence}.txt")
    gt_poses = get_true_pose(pose_file)

    for frame in range(frame_start, frame_end):
        path_l_start = os.path.join(images_path, "dataset/sequences", sequence, "image_0", f"{frame:06d}.png")
        path_r_start = os.path.join(images_path, "dataset/sequences", sequence, "image_1", f"{frame:06d}.png")
        path_l_end = os.path.join(images_path, "dataset/sequences", sequence, "image_0", f"{frame+1:06d}.png")
        path_r_end = os.path.join(images_path, "dataset/sequences", sequence, "image_1", f"{frame+1:06d}.png")

        image_l_start = cv2.imread(path_l_start, cv2.IMREAD_GRAYSCALE)
        image_r_start = cv2.imread(path_r_start, cv2.IMREAD_GRAYSCALE)
        image_l_end = cv2.imread(path_l_end, cv2.IMREAD_GRAYSCALE)
        image_r_end = cv2.imread(path_r_end, cv2.IMREAD_GRAYSCALE)

        disp_map = compute_left_disparity_map(image_l_start, image_r_start, matcher='sgbm')
        depth_map_start = calc_depth_map(disp_map, K0, t0, t1)

        kp1, des1 = extract_features(image_l_start)
        kp2, des2 = extract_features(image_l_end)

        if des1 is None or des2 is None:
            print(f"Skipping frame {frame} due to no descriptors.")
            continue

        matches = match_features(des1, des2, matching='BF', detector='sift', sort=True, k=2)
        good_matches = filter_matches_distance(matches)

        rmat, tvec, _, _ = estimate_motion(good_matches, kp1, kp2, K0, depth_map_start)

        if rmat is not None and tvec is not None:
            T = np.eye(4)
            T[:3, :3] = rmat
            T[:3, 3] = tvec

            pose = pose @ np.linalg.inv(T)
            pos = pose[:3, 3]
            rot = R.from_matrix(pose[:3, :3]).as_quat()
            trajectory.append([frame + 1, pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]])
        else:
            print(f"Skipping pose update for frame {frame}")

        print ("frame:",frame)
        print ("tvec:",tvec)

    save_trajectory_to_csv(trajectory, csv_path)
    plot_trajectories(trajectory, gt_poses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Odometry App")
    parser.add_argument('--config', type=str, help='Path to the config file', required=True)
    parser.add_argument('--simple', action='store_true', help='Use only two frames')
    args = parser.parse_args()
    if args.simple:
        two_frame_vo(args.config)
    else:
        full_svo(args.config)




