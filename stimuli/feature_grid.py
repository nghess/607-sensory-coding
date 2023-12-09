import numpy as np
from PIL import Image
import tifffile

# Generate blank canvas
canvas = np.zeros((180, 128, 128), np.uint8)

# Set up rescale
divisor = 6
rescale = 1.0/divisor
dim_s = canvas.shape[1]//divisor
feature_scaled = np.zeros([canvas.shape[0], dim_s, dim_s], np.uint8)

# Set up grid
limit = canvas.shape[1] - dim_s
step = 9

# Direction
rotation = ["clockwise", "counterclockwise"]

for dir in rotation: 
    # Load binary Tiff stack and convert to 0-255 np.array
    feature = tifffile.TiffFile(f"dataset/diameter_4_{dir}.tiff").asarray().astype(np.uint8)*255

    # Rescale feature with PIL
    for ii in range(canvas.shape[0]):
        # Convert the slice to PIL Image
        slice_image = Image.fromarray(feature[ii, :, :])
        # Resize with anti-aliasing
        resized_slice = slice_image.resize((dim_s, dim_s), Image.Resampling.LANCZOS)
        # Convert back to array and insert in stack
        feature_scaled[ii,:,:] = np.array(resized_slice)

    for x in range(0, limit, step):
        for y in range(0, limit, step):
            # Insert scaled feature into canvas
            canvas[:,x:x+dim_s,y:y+dim_s] += feature_scaled

            # Save
            out_fn = f"{dir}_x{x}_y{y}.tiff"
            tifffile.imwrite(f"dataset/grid/{out_fn}", canvas, mode='w')

            # Append labels to labels file
            with open('dataset/grid/labels_grid.csv', 'a') as file:
                file.write(f"dataset/grid/{out_fn},{dir},line\n")
            
            # Reset canvas
            canvas = np.zeros((180, 128, 128), np.uint8)

