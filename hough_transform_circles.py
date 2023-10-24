import numpy as np
from collections import defaultdict
import cv2
import os
import sys


def sobel_filter(image):
    # Sobel kernels
    g_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    g_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    g_x = cv2.filter2D(image, -1, g_x)
    g_y = cv2.filter2D(image, -1, g_y)

    g = np.hypot(g_x, g_y)
    g = np.array(g / g.max() * 255, np.uint8)

    theta = np.arctan2(g_y, g_x)

    return g, theta


def non_maximum_suppression(img, gradient):
    suppressed_image = np.zeros(img.shape)
    # Convert to degrees
    a = gradient * 180.0 / np.pi
    a[a < 0] += 180

    # Identify edge direction using the angle matrix
    for i in range(img.shape[0] - 1):
        for j in range(img.shape[1] - 1):
            u, v = 255, 255
            if (0 <= a[i, j] < 22.5) or (157.5 <= a[i, j] <= 180):  # 0°
                u, v = img[i, j - 1], img[i, j + 1]
            elif 22.5 <= a[i, j] < 67.5:  # 45°
                u, v = img[i + 1, j + 1], img[i - 1, j - 1]
            elif 67.5 <= a[i, j] < 112.5:  # 90°
                u, v = img[i - 1, j], img[i + 1, j]
            elif 112.5 <= a[i, j] < 157.5:  # 135°
                u, v = img[i - 1, j + 1], img[i + 1, j - 1]

            # Check if the pixels in the direction have a higher \
            # intensity than the current pixel
            if (img[i, j] >= u) and (img[i, j] >= v):
                suppressed_image[i, j] = img[i, j]
            else:
                suppressed_image[i, j] = 0

    return np.array(suppressed_image, np.uint8)


def double_threshold(image, th1=0.05, th2=0.1):
    weak_pixels = np.array(25)
    strong_pixels = np.array(255)

    high_threshold = image.max() * th2
    low_threshold = high_threshold * th1

    thresholded_image = np.zeros(image.shape)
    thresholded_image[np.where(image >= high_threshold)] = strong_pixels
    thresholded_image[np.where((image <= high_threshold) & (
        image >= low_threshold))] = weak_pixels
    thresholded_image[np.where(image < low_threshold)] = 0

    return np.array(thresholded_image, np.uint8), weak_pixels, strong_pixels


def hysteresis(thresholded_image, weak_pixels, strong_pixels):
    hysteresis_image = np.copy(thresholded_image)
    for i in range(1, hysteresis_image.shape[0] - 1):
        for j in range(1, hysteresis_image.shape[1] - 1):
            # Check 8 connected neighborhood pixels (around the pixel: \
            # top, right, bottom,
            # left, top-right, top-left, bottom-right, bottom-left)
            if hysteresis_image[i, j] == weak_pixels:
                if ((hysteresis_image[i + 1, j - 1] == strong_pixels) or
                        (hysteresis_image[i + 1, j] == strong_pixels) or
                        (hysteresis_image[
                            i + 1, j + 1] == strong_pixels) or
                        (hysteresis_image[i, j - 1] == strong_pixels) or
                        (hysteresis_image[i, j + 1] == strong_pixels) or
                        (hysteresis_image[
                            i - 1, j - 1] == strong_pixels) or
                        (hysteresis_image[i - 1, j] == strong_pixels) or
                        (hysteresis_image[
                            i - 1, j + 1] == strong_pixels)):
                    hysteresis_image[i, j] = 255
                else:
                    hysteresis_image[i, j] = 0

    return np.array(hysteresis_image, np.uint8)


def canny(gray, gaussian_kernel_size):
    gray = cv2.GaussianBlur(gray, (gaussian_kernel_size, gaussian_kernel_size),
                            0)
    blur, theta = sobel_filter(gray)
    suppressed = non_maximum_suppression(blur, theta)
    thresholded, weak_pixels, strong_pixels = double_threshold(suppressed)
    edges = hysteresis(thresholded, weak_pixels, strong_pixels)

    return edges


def hough_circles(edges, min_radius, max_radius):
    width = edges.shape[0]
    height = edges.shape[1]
    radius_range = range(min_radius, max_radius)
    empty = np.zeros_like(edges)
    accumulator = defaultdict(int)
    # Unknown radius - iterate over given radius range
    for radius in radius_range:
        # Iterate over pixels
        for i in range(0, width):
            for j in range(0, height):
                # Filter edge pixels
                if edges[i][j] != 0:
                    for theta in range(0, 360, 5):
                        # Polar coordinate for center
                        a = int(i - radius * np.cos(theta * np.pi / 180))
                        b = int(j - radius * np.sin(theta * np.pi / 180))
                        try:
                            # Voting
                            accumulator[(a, b, radius)] += 1
                            empty[a][b] += 1
                        except IndexError:
                            continue
    target = empty * 255

    return target, accumulator


def get_circles(accumulator, vote_threshold):
    circles = []
    for candidate in sorted(accumulator, key=lambda i: -i[2]):
        x, y, r = candidate
        if accumulator[candidate] >= vote_threshold and x > 0 and y > 0 and \
                all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for
                    xc, yc, rc in
                    circles):
            circles.append((x, y, r))
    return circles


def draw_circles(circles, image, color):
    for circle in circles:
        cv2.circle(image, (circle[1], circle[0]), circle[2], color, 2)


def draw_centers(circles, image, color):
    for circle in circles:
        cv2.circle(image, (circle[1], circle[0]), 1, color, 2)


def main():
    try:
        path = sys.argv[1]
        min_radius = int(sys.argv[2])
        max_radius = int(sys.argv[3])
        gaussian_kernel = int(sys.argv[4])
        vote_threshold = int(sys.argv[5])

        file_name = os.path.basename(path)
        test_image = cv2.imread(path)
        original_image = test_image.copy()

    except (AttributeError, IndexError, cv2.error):
        raise SystemExit(f"Usage: {sys.argv[0]} <image_path> <min_radius>"
                         f" <max_radius> <gaussian_kernel_size> "
                         f"<vote_threshold")

    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    test_image = canny(test_image, gaussian_kernel)
    test_image, array = hough_circles(test_image, min_radius, max_radius)

    circles_list = get_circles(array, vote_threshold)
    draw_circles(circles_list, original_image, (0, 255, 0))
    draw_centers(circles_list, original_image, (255, 0, 0))

    black = np.zeros_like(test_image, dtype=np.uint8)
    draw_centers(circles_list, black, (255, 255, 255))

    cv2.imwrite("output/accumulator_" + file_name, black)
    cv2.imwrite("output/circles_" + file_name, original_image)

    cv2.imshow("Accumulator Image", black)
    cv2.imshow("Hough Transfrom for Circles", original_image)
    cv2.waitKey()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
