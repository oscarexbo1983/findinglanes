# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 18:50:24 2018

@author: el_bo
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML





def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#dettect edges    
def canny(img, low_threshold=50, high_threshold=150):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

#smoothing
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

#filterregion

def vertices(image):
    

    bottom_left  = [image.shape[1]*0.1, image.shape[0]*0.95]
    top_left     = [image.shape[1]*0.4, image.shape[0]*0.6]
    bottom_right = [image.shape[1]*0.9, image.shape[0]*0.95]
    top_right    = [image.shape[1]*0.6, image.shape[0]*0.6] 
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
 
    return vertices



def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#def filter_yw(image):
#    gray=grayscale(image)
#    converted_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#    lower_yellow = np.uint8([20, 100, 100])
#    upper_yellow = np.uint8([30, 255, 255])
#   mask_yellow = cv2.inRange(converted_hsv, lower_yellow, upper_yellow)
#    mask_white = cv2.inRange( converted_hsv, 200, 255)
#   masked_yw = cv2.bitwise_or(mask_white, mask_yellow)
#    return cv2.bitwise_and(gray, masked_yw)
    

def filter_yw(image):
    img_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    plt.imshow(img_hls)
      
    boundaries_wy = [
	([0 , 200, 0], [255, 255, 255]),
	([10, 10, 100], [40, 255, 255])]   
      
   
    mask_yw=[] 
    for (lower, upper) in boundaries_wy:
	# create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
 
	
        mask_yw.append(cv2.inRange(img_hls, lower, upper))
       
    mask = cv2.bitwise_or(mask_yw[0], mask_yw[1])
    masked= cv2.bitwise_and(image, image, mask = mask)
    return(masked)
    
    




def draw_lines(img, lines, color=[255, 0, 0], thickness=20):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """


    y_left= []
    y_right= []
    sumr=0
    suml=0
    sum_slope_l=0
    sum_int_l=0
    sum_slope_r=0
    sum_int_r=0
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1==x2:
                continue
            m = (y2-y1)/(x2-x1)
            a = y1 - m*x1
            length = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if m < 0: 
                sum_slope_l=sum_slope_l+m*length
                sum_int_l=sum_int_l+a*length                    
                suml=suml+length
                y_left.append(y1)
                y_left.append(y2)
            else:
                sum_slope_r=sum_slope_r+m*length
                sum_int_r=sum_int_r+a*length                    
                sumr=sumr+length
                y_right.append(y1)
                y_right.append(y2)
    
    
    slope1 = sum_slope_l/suml
    intercept1 = sum_int_l/suml
    slope2= sum_slope_r/sumr
    intercept2 = sum_int_r/sumr
    miny= min(min(y_left),min(y_right))
  
    y1 = img.shape[0] 
    y2 = miny
    
    
    x1l = int((y1 - intercept1)/slope1)
    x2l = int((y2 - intercept1)/slope1)
    y1 = int(y1)
    y2 = int(y2)
    x1r = int((y1 - intercept2)/slope2)
    x2r = int((y2 - intercept2)/slope2)
    
 
    


    r1= (x1r,y1)
    r2= (x2r,y2)
    cv2.line(img, r1, r2, color, thickness)

#    l1 = (max_left_x, miny)
 #   l2 = (min_left_x, img.shape[0])
    l1= (x1l,y1)
    l2= (x2l,y2)
    cv2.line(img, l1, l2, color, thickness)

   
    




def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
   
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def draw_lanes(image):
    
    
    f=filter_yw(image) 
    gray=grayscale(f)     
    smooth=gaussian_blur(gray, kernel_size=5)
    edges=canny(smooth, low_threshold=50, high_threshold=150)
    vert=vertices(edges)
    reg=region_of_interest(edges, vert)
    h=hough_lines(reg, 2, np.pi/180,  10, 20, 10)
    x=weighted_img(h,image)
    plt.imshow(h)
    return x
    
    
#draw_lanes(mpimg.imread('test_images/solidWhiteCurve.jpg'))
#draw_lanes(mpimg.imread('test_images/solidWhiteRight.jpg'))
draw_lanes(mpimg.imread('test_images/solidYellowLeft.jpg'))
#draw_lanes(mpimg.imread('test_images/solidYellowCurve2.jpg'))
#draw_lanes(mpimg.imread('test_images/whiteCarLaneSwitch.jpg'))
#draw_lanes(mpimg.imread('test_images/solidYellowCurve.jpg'))





#printing out some stats and plotting
#print('This image is:', type(image), 'with dimensions:', image.shape)
#plt.imshow(image) 





def video_reader(video_in, video_out):
    clip = VideoFileClip(video_in)
    clip = clip.fl_image(draw_lanes)
    clip.write_videofile(video_out, audio=False)
    clip.reader.close()
    clip.audio.reader.close_proc()
#process_video('solidWhiteRight.mp4', 'white.mp4') 

#clip = VideoFileClip(os.path.join('test_videos', 'solidWhiteRight.mp4'))
#video_reader('test_videos/solidWhiteRight.mp4','output_videos/whi5.mp4')
#video_reader('test_videos/solidYellowLeft.mp4','output_videos/yel5.mp4')
#video_reader('test_videos/challenge.mp4','output_videos/cha7.mp4')


