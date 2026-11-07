import cv2
import numpy as np

# === 1. Load image ===
img = cv2.imread("results/floorplan_walls/walls.jpg")  # or your .png
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# === 2. Adaptive threshold (preserves faint, thin walls) ===
bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                           cv2.THRESH_BINARY_INV, 31, 5)

# === 3. Line-preserving morphological filtering ===
# Emphasize long vertical + horizontal lines (walls)
vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vert_kernel, iterations=1)
horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horiz_kernel, iterations=1)

# Combine vertical and horizontal walls
walls = cv2.bitwise_or(vertical, horizontal)

# === 4. Remove tiny specks or text ===
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(walls, connectivity=8)
mask = np.zeros_like(gray)
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    if area > 50 and (w > 5 or h > 5):
        mask[labels == i] = 255

# === 5. Optional cleanup / smoothing ===
mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
mask = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=1)

# === 6. Save final wall-only image ===
result = cv2.bitwise_not(mask)  # black walls on white
cv2.imwrite("results/floorplan_walls/walls_only.png", result)
print("✅ Saved clean wall-only image: results/floorplan_walls/walls_only.png")

# import cv2
# import numpy as np

# # === 1. Load image ===
# img = cv2.imread("results/floorplan_walls/walls.jpg")  # or .png
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # === 2. Adaptive threshold (preserves faint lines) ===
# bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                            cv2.THRESH_BINARY_INV, 35, 5)

# # === 3. Remove small specks / text ===
# # Use a *small* kernel so lines aren't removed
# kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
# clean = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel_small, iterations=1)

# # === 4. Keep long, thin structures (walls) ===
# # Label components
# num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)

# mask = np.zeros_like(gray)
# for i in range(1, num_labels):  # skip background
#     area = stats[i, cv2.CC_STAT_AREA]
#     w = stats[i, cv2.CC_STAT_WIDTH]
#     h = stats[i, cv2.CC_STAT_HEIGHT]
#     aspect = max(w, h) / (min(w, h) + 1e-5)
#     # walls: long and thin, not tiny blobs
#     if area > 50 and (w > 5 or h > 5) and aspect > 2:
#         mask[labels == i] = 255

# # === 5. (Optional) Smooth the lines slightly ===
# mask = cv2.dilate(mask, np.ones((2,2), np.uint8), iterations=1)
# mask = cv2.erode(mask, np.ones((2,2), np.uint8), iterations=1)

# # === 6. Save result ===
# result = cv2.bitwise_not(mask)
# cv2.imwrite("results/floorplan_walls/walls_filtered_safe.png", result)
# print("✅ Saved: results/floorplan_walls/walls_filtered_safe.png")
