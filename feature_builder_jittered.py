import cv2
import numpy as np
from scipy.interpolate import griddata

def rotate_point(origin, point, angle):

    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)

    return int(qx), int(qy)


def generate_mesh(size, spacing):

    assert size % spacing == 0, "The 'spacing' value must evenly divide the 'size' of the mesh."
    
    mesh = [] 

    for ii in range(0, size+spacing, spacing):
        for jj in range(0, size+spacing, spacing):
            mesh.append([ii, jj])
    
    return mesh

# Mesh Jitter function
def jitter_img(img, amount, spacing=8, seed=57):

    np.random.seed(seed)

    sz = img.shape[0]

    # Create a sparse mesh grid (set by spacing arg)
    grid_x, grid_y = np.mgrid[0:sz:spacing, 0:sz:spacing]

    # Create a dense grid for the whole image
    dense_grid_x, dense_grid_y = np.mgrid[0:sz, 0:sz]

    x_jittered = grid_x + np.random.uniform(-amount, amount, grid_x.shape)
    y_jittered = grid_y + np.random.uniform(-amount, amount, grid_y.shape)

    shifted_grid_x = x_jittered#np.asarray(grid_x)
    shifted_grid_y = y_jittered#np.asarray(grid_y)

    # Interpolate the transformations from the sparse grid to the dense grid
    interp_x = griddata((grid_x.flatten(), grid_y.flatten()), shifted_grid_x.flatten(), (dense_grid_x, dense_grid_y), method='linear')
    interp_y = griddata((grid_x.flatten(), grid_y.flatten()), shifted_grid_y.flatten(), (dense_grid_x, dense_grid_y), method='linear')
  
    #mesh = [[int(a+b) for a,b in zip(pt1, pt2)] for pt1, pt2 in zip(mesh, jittered_pts)]
    jittered_img = cv2.remap(img, interp_x.astype(np.float32), interp_y.astype(np.float32), cv2.INTER_LINEAR)
    print(shifted_grid_y)
    return jittered_img


# Generates feature to animate
def generate_feature(win, feature, angle=0, thickness=1, jitter=0):

    # Check that feature is correctly specified
    assert feature in ("cross", "tee", "elbow", "end"), "Feature must be one of 'cross', 'tee', 'elbow', or 'end'."

    # Convert angle to radians
    angle = np.deg2rad(angle)
    
    # Create feature_output matrix, if end feature is selected, generate end-detector mask
    feature_output = np.zeros((win, win), np.uint8)
    if feature == "end":
        end_mask = np.zeros((win, win), np.uint8)

    # Define center of feature_output
    center = win // 2
    origin = (center, center)

    # Draw Top
    if feature in ("cross", "tee", "end", "elbow"):
        endpoint = rotate_point(origin, (center, 1), angle)
        cv2.line(feature_output, origin, endpoint, (1, 1, 1), thickness)
    # Draw Bottom
    if feature == "cross":
        endpoint = rotate_point(origin, (center, win-1), angle)
        cv2.line(feature_output, origin, endpoint, (1, 1, 1), thickness)
    # Draw Left 
    if feature in ("cross", "tee"):
        endpoint = rotate_point(origin, (1, center), angle)
        cv2.line(feature_output, origin, endpoint, (1, 1, 1), thickness)
    # Draw Right 
    if feature in ("cross", "tee", "elbow"):
        endpoint = rotate_point(origin, (win-1, center), angle)
        cv2.line(feature_output, origin, endpoint, (1, 1, 1), thickness)
    
    # Create Mesh
    mesh = generate_mesh(win, 16)
    if jitter != 0:
        feature_output = jitter_img(feature_output, jitter)
    
    return feature_output*255, mesh

# Window Size
win = 128
# Starting Position
deg = 0
# Toggle animation
animation = False

if animation:
    while deg in range(360):
        feature, mesh = generate_feature(win, "cross", deg, 1, jitter=5)
        for pt in range(len(mesh)):
            cv2.circle(feature, mesh[pt], 1, (255, 255, 255), -1)
        cv2.imshow("Feature", feature)
        cv2.waitKey(16)
        deg += 1
        if deg == 360:
            deg =  0
else:
    feature, mesh = generate_feature(win, "cross", deg, 2, jitter=0)

    cv2.imshow("Feature", feature)
    cv2.waitKey(0)
    cv2.destroyAllWindows()