import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

image = cv2.imread("images/road_lanes.jpg")

# Display the image using openCV
def display_Image_cv2(title,img):
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Display image using matplotlib to find the vertices
def display_Image_matplotlib(img):
    plt.figure()
    plt.imshow(img)
    plt.show()

def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    # Retrieve the number of color channels of the image.
    # channel_count = img.shape[2]
    # Create a match color with the same color channel counts.
    # match_mask_color = (255,) * channel_count
    match_mask_color = (255)
    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)
    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

[height,width,channels] = image.shape

#Find the edges in the cropped image
grey_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
grey_image_with_edges = cv2.Canny(grey_image,100,200)

#Define a region of interest for the image from Dash Cam
region_of_interest_vertices = [(130,height),(300,130),(350,130),(width-130,height)]

cropped_image = region_of_interest(grey_image_with_edges,np.array([region_of_interest_vertices], np.int32),)
# display_Image_cv2("cropped_image",cropped_image)
#display_Image_matplotlib(cropped_image)
# display_Image_cv2("Canny Image",cropped_image)


lines = cv2.HoughLinesP(
    cropped_image,
    rho=8,
    theta=np.pi / 30,
    threshold=40,
    lines=np.array([]),
    minLineLength=85,
    maxLineGap=50
)

def draw_lines(img, lines, color=[155, 233, 2], thickness=3):
    # If there are no lines to draw, exit.
    if lines is None:
        return
    # Make a copy of the original image.
    img = np.copy(img)
    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    # Return the modified image.
    return img


# left_line_x = []
# left_line_y = []
# right_line_x = []
# right_line_y = []
# for line in lines:
#     for x1, y1, x2, y2 in line:
#         slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.
#         if math.fabs(slope) < 0.5: # <-- Only consider extreme slope
#             continue
#         if slope <= 0: # <-- If the slope is negative, left group.
#             left_line_x.extend([x1, x2])
#             left_line_y.extend([y1, y2])
#         else: # <-- Otherwise, right group.
#             right_line_x.extend([x1, x2])
#             right_line_y.extend([y1, y2])
#
# min_y = image.shape[0] * (3 / 5) # <-- Just below the horizon
# max_y = image.shape[0] # <-- The bottom of the image
#
# poly_left = np.poly1d(np.polyfit(
#     left_line_y,
#     left_line_x,
#     deg=1
# ))
# left_x_start = int(poly_left(max_y))
# left_x_end = int(poly_left(min_y))
# poly_right = np.poly1d(np.polyfit(
#     right_line_y,
#     right_line_x,
#     deg=1
# ))
# right_x_start = int(poly_right(max_y))
# right_x_end = int(poly_right(min_y))
#

# final_image = draw_lines(
#     image,
#     [[
#         [int(left_x_start), int(max_y), int(left_x_end), int(min_y)],
#         [int(right_x_start), int(max_y), int(right_x_end), int(min_y)],
#     ]],
# )
final_image = draw_lines(image,lines)
display_Image_cv2("Lane Detected Image",final_image)
