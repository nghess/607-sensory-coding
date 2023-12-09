import cv2
import numpy as np
from PIL import Image
import tifffile

# Feature List
features = ["cross", "tee", "elbow", "radius", "diameter"]
spin = ["clockwise", "counterclockwise"]
lineweights = [1,4,16]
divisors = [2,4,6]

for shape in features:
    for lw in lineweights:
        for divisor in divisors:
            for dir in spin:
                # Load 1st-gen feature
                in_fn = f"{shape}_{lw}_{dir}.tiff"
                with tifffile.TiffFile(f"dataset/{in_fn}") as tiff:
                    tiff_stack = tiff.asarray().astype(np.uint8)*255

                    # Generate blank canvas
                    canvas = np.zeros((tiff_stack.shape[0], tiff_stack.shape[1], tiff_stack.shape[2]), np.uint8)

                    # Set up rescale
                    rescale = 1.0/divisor
                    dim_s = canvas.shape[1]//divisor
                    feature_scaled = np.zeros([canvas.shape[0], dim_s, dim_s], np.uint8)

                    # Rescale with PIL
                    for ii in range(canvas.shape[0]):
                        # Convert the slice to PIL Image
                        slice_image = Image.fromarray(tiff_stack[ii, :, :])
                        # Resize with anti-aliasing
                        resized_slice = slice_image.resize((dim_s, dim_s), Image.Resampling.LANCZOS)
                        # Convert back to array and insert in stack
                        feature_scaled[ii,:,:] = np.array(resized_slice)

                    # Random window location with upper left corner no closer to canvas edge than (dim_s, dim_s)
                    ceil = canvas.shape[1] - dim_s
                    x = int(np.random.uniform(0,ceil))
                    y = int(np.random.uniform(0,ceil))
                    canvas[:,x:x+dim_s,y:y+dim_s] += feature_scaled

                    out_fn = f"{shape}_{lw}_{dir}_x{x}_y{y}_d{divisor}.tiff"
                    tifffile.imwrite(f"dataset/scatter/{out_fn}", canvas, mode='w')

                    # Convert to bit depth 1 and save as TIFF stack 
                    # img_stack = [canvas[ii, :, :] for ii in range(canvas.shape[0])]
                    # frames = [Image.fromarray(img).convert('1') for img in img_stack]
                    # frames[0].save(f"dataset/scatter/{out_fn}", save_all=True, append_images=frames[1:])

                    # Append labels to labels file
                    with open('dataset/scatter/labels_scatter.csv', 'a') as file:
                        file.write(f"scatter/{out_fn},{dir},{shape}\n")

"""
Preview Output
"""
# cv2.imshow("Feature", tiff_stack[30,:,:])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Animated Preview
# deg = 0
# while deg in range(tiff_stack.shape[0]):
#     cv2.imshow("Feature", canvas[deg, :, :])
#     cv2.waitKey(1)
#     deg += 1
#     if deg == tiff_stack.shape[0]:
#         deg =  0
    