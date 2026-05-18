import cv2
import matplotlib.pyplot as plt

img = cv2.imread("edges_in_mask.png", cv2.IMREAD_GRAYSCALE)

if len(img.shape) == 2:
    # Gray
    plt.imshow(img, cmap='gray')

elif img.shape[2] == 3:
    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)

elif img.shape[2] == 4:
    # BGRA -> RGBA
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    plt.imshow(img)

plt.axis('off')
plt.show()