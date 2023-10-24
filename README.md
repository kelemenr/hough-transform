# Hough Transform for Circles

This Python script performs Hough Transform with Canny edge detection for detecting circles in an image using OpenCV. The algorithm can be used to find circular objects in images.

## Algorithm Overview

1. **Sobel Filter:** The input image is filtered using the Sobel operator to detect edges.

2. **Non-Maximum Suppression:** The edges are thinned using non-maximum suppression, which helps identify the most prominent edges.

3. **Double Thresholding:** The edges are categorized into strong, weak, and non-edges by applying double thresholding.

4. **Hysteresis:** Edges are finalized by tracing along the strong edges and connecting weak edges if they are adjacent.

5. **Circle Detection:** The algorithm finds circles based on the accumulated votes and a specified vote threshold.

6. **Visualization:** The script includes functions to draw detected circles and their centers on the original image.

## Usage

To run the script, provide the following command-line arguments:
- `<image_path>`: Path to the input image.
- `<min_radius>`: Minimum radius of circles to detect.
- `<max_radius>`: Maximum radius of circles to detect.
- `<gaussian_kernel_size>`: Size of the Gaussian blur kernel for preprocessing.
- `<vote_threshold>`: Minimum vote count for a circle to be considered a valid detection.

## Example

Here is an example command to run the script:

```bash
python hough_transform_circles.py input/circles.png 14 16 7 25
```


## Output

The script produces the following results:

- **Accumulator Image:** A visualization of the Hough Transform accumulator, showing potential circle centers.

- **Hough Transform for Circles:** The original image with detected circles drawn in green and their centers in red.

<img src="https://github.com/kelemenr/hough-transform/assets/47530064/acf40952-9e84-4903-8475-86995bc5b4ff" width="300">
<img src="https://github.com/kelemenr/hough-transform/assets/47530064/1972a32f-0307-4946-a279-a82ea2a25ab2" width="300"><br><br>

<img src="https://github.com/kelemenr/hough-transform/assets/47530064/7c7845ca-f331-4f74-b3c0-e83c7ee151c7" width="300">
<img src="https://github.com/kelemenr/hough-transform/assets/47530064/fdda55bc-ebf8-4806-9257-0df04f981435" width="300"><br><br>

<img src="https://github.com/kelemenr/hough-transform/assets/47530064/20efcb4c-d926-460e-9a33-a407add4a596" width="300">
<img src="https://github.com/kelemenr/hough-transform/assets/47530064/e0c9d188-b7bc-4c1c-9bdd-711b5757f331" width="300"><br><br>




