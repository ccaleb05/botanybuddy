import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import time as tm
import os

SHOW_TIME_LAPSE = False

IMAGE_DELAY = 1
TIME_LAPSE_DELAY = 1
CONVERSION_FACTOR = 2000
seconds = 0
time = []
square_inches = []
NUM_IMAGES = 10;

CAM_PORT = 0
cam = cv2.VideoCapture(CAM_PORT)

while 1:
    seconds = seconds + 1
    result, vers = cam.read()

    hsvImage = cv2.cvtColor(vers, cv2.COLOR_RGB2HSV)

    low_green = np.array([40, 100, 50])
    high_green = np.array([80, 255, 255])

    green_mask = cv2.inRange(hsvImage, low_green, high_green)
    green = cv2.bitwise_and(vers, vers, mask=green_mask)

    GRAY_Image = cv2.cvtColor(green, cv2.COLOR_RGB2GRAY)
    (thresh, GRAY_Image) = cv2.threshold(GRAY_Image, 0, 255, cv2.THRESH_BINARY)

    im_floodfill = GRAY_Image.copy()
    h, w = GRAY_Image.shape[:2]
    BI_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, BI_mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    output = GRAY_Image | im_floodfill_inv

    count = cv2.countNonZero(output) / CONVERSION_FACTOR
    output = cv2.putText(output, str(count) + " square inches", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                         cv2.LINE_AA)
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

    square_inches.append(count)
    time.append(seconds)

    hfont = {'fontname': 'Consolas'}

    plt.plot(time, square_inches, color='green')
    plt.xlabel('Time (Seconds)',**hfont)
    plt.ylabel('Growth (Square Inches)',**hfont)
    plt.title('Plant Growth vs. Time',**hfont)
    plt.savefig('PlantGrowthGraph.png')

    plot = cv2.imread('PlantGrowthGraph.png')
    plot = cv2.resize(plot, (640, 480))
    display_window = np.concatenate((plot, output), axis=1)
    slideshow_window = np.concatenate((plot, vers), axis=1)
    path = r'C:\Users\cmike\PycharmProjects\enge1216project\images'
    cv2.imwrite(os.path.join(path, 'progress' + str(seconds) + '.png'), slideshow_window)
    cv2.imshow('output', display_window)
    key_press = cv2.waitKey(5)
    if key_press >= 0 or seconds == NUM_IMAGES:
        break
    #tm.sleep(IMAGE_DELAY)

cv2.destroyAllWindows()

if SHOW_TIME_LAPSE:
    def sort_folder(file_name):
        num = file_name[:-4]
        return int(num[8:])

    path = r'C:\Users\cmike\PycharmProjects\enge1216project\images'
    folder = os.listdir(path)
    folder.sort(reverse=False,key=sort_folder)
    print(folder)

    current_img = 0
    while 1:
        progress_image = cv2.imread(os.path.join(path, folder[current_img]))
        cv2.imshow('progress', progress_image)
        print("printing " + folder[current_img])
        tm.sleep(TIME_LAPSE_DELAY)
        current_img = current_img + 1
        key_press = cv2.waitKey(5)
        if key_press >= 0 or current_img == NUM_IMAGES:
            break

    cv2.destroyAllWindows()
