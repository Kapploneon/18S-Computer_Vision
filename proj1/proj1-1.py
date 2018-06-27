# CS6384 Computer Vision 2018S
# projet 1
# part1
# Hao WAN (hxw161730)


import cv2
import numpy as np
import sys

if(len(sys.argv) != 7) :
    print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
    print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
    sys.exit()

w1 = float(sys.argv[1])
h1 = float(sys.argv[2])
w2 = float(sys.argv[3])
h2 = float(sys.argv[4])
name_input = sys.argv[5]
name_output = sys.argv[6]

if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1) :
    print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
    sys.exit()

inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
if(inputImage is None) :
    print(sys.argv[0], ": Failed to read image from: ", name_input)
    sys.exit()

cv2.namedWindow("input image: " + name_input, cv2.WINDOW_NORMAL);
cv2.resizeWindow("input image: " + name_input, 300, 300);
cv2.moveWindow("input image: " + name_input, 20, 40);
cv2.imshow("input image: " + name_input, inputImage)

rows, cols, bands = inputImage.shape # bands == 3
W1 = round(w1*(cols-1))
H1 = round(h1*(rows-1))
W2 = round(w2*(cols-1))
H2 = round(h2*(rows-1))

# The transformation should be based on the
# historgram of the pixels in the W1,W2,H1,H2 range.
# The following code goes over these pixels

tmp = np.copy(inputImage)

for i in range(H1, H2) :
    for j in range(W1, W2) :
        b, g, r = inputImage[i, j]
        gray = round(0.3*r + 0.6*g + 0.1*b)
        tmp[i, j] = [gray, gray, gray]

cv2.namedWindow('tmp', cv2.WINDOW_NORMAL);
cv2.resizeWindow('tmp', 300, 300);
cv2.moveWindow('tmp', 350, 40);
cv2.imshow('tmp', tmp)

# end of example of going over window


# -------BGR to Luv and the inverse function-----

# constants for XYZ to Luv and the inverse
Xw = 0.95
Yw = 1.0
Zw = 1.09
uw = (4.0 * Xw) / (Xw + 15.0 * Yw + 3.0 * Zw)
vw = (9.0 * Yw) / (Xw + 15.0 * Yw + 3.0 * Zw)

def invGammaSingle(x):
    if(x < 0.03928):
        x = x / 12.92
    else:
        x = pow(((x + 0.055) / 1.055), 2.4)
    return x

def gammaSingle(x):
    if(x < 0.00304):
        x = x * 12.92
    else:
        x = 1.055 * pow(x, (1.0/2.4)) - 0.055
    return x

def checkValue(x):
    if(x>1.0):
        x=1.0
    if(x<0.0):
        x=0.0
    return x

def toLuv(x):
    # scale to [0,1]
    x = x / 255.0
    for i in range(0,rows):
        for j in range(0,cols):
            b, g, r = x[i,j]

            # inverse gamma correction
            b = invGammaSingle(b)
            g = invGammaSingle(g)
            r = invGammaSingle(r)

            # inverse matrix multiplication
            X = 0.412453*r + 0.35758*g + 0.180423*b
            Y = 0.212671*r + 0.71516*g + 0.072169*b
            Z = 0.019334*r + 0.119193*g + 0.950227*b

            # XYZ to Luv
            t = Y / Yw
            if(t > 0.008856):
                L = 116.0 * pow(t, 1.0 / 3.0) - 16.0
            else:
                L = 903.3 * t
            d = X + 15.0 * Y + 3.0 * Z
            if(d == 0.0):
                L = u = v = 0.0
            else:
                uprime = 4.0 * X / d
                vprime = 9.0 * Y / d
                u = 13.0 * L * (uprime - uw)
                v = 13.0 * L * (vprime - vw)

            x[i,j] = L, u, v
    return x

def toBgr(x):
    for i in range(0, rows):
        for j in range(0, cols):

            # Luv to XYZ
            L, u, v = x[i, j]
            if(L == 0.0):
                X = Y = Z = 0.0
            else:
                uprime = (u + 13 * uw * L) / (13 * L)
                vprime = (v + 13 * vw * L) / (13 * L)
                if(L > 7.9996):
                    Y = pow(((L + 16.0) / 116.0), 3.0) * Yw
                else:
                    Y = L * Yw / 903.3
                if(vprime == 0.0):
                    X = Z = 0.0
                else:
                    X = Y * 2.25 * uprime / vprime
                    Z = Y * (3.0 - 0.75 * uprime - 5.0 * vprime) / vprime

            # matrix multiplication
            r = 3.240479 * X + (- 1.53715) * Y + (-0.498535) * Z
            g = (- 0.969256) * X + 1.875991 * Y + 0.041556 * Z
            b = 0.055648 * X + (-0.204043) * Y + 1.057311 * Z
            r = checkValue(r)
            g = checkValue(g)
            b = checkValue(b)

            # gamma correction
            b = gammaSingle(b)
            g = gammaSingle(g)
            r = gammaSingle(r)
            b = checkValue(b)
            g = checkValue(g)
            r = checkValue(r)

            # scale and round to 0-255
            b = round(b * 255.0)
            g = round(g * 255.0)
            r = round(r * 255.0)

            x[i,j]=b, g, r
    return x.astype(int)
# -----END of BGR to Luv and the inverse function-----

# ---------linear scaling--------------------

# copy the data into a new array
inputCopy = np.copy(inputImage)
# do the BGR to Luv transformation
Luv_img = toLuv(inputCopy)

# get the value of L for all the pixels in the window
# H2-H1 rows and W2-W1 cols
L_window = np.zeros([H2-H1, W2-W1], dtype=np.uint8)
for i in range(H1, H2) :
    for j in range(W1, W2) :
        # round the value
        L_window[i-H1,j-W1] = round(Luv_img[i,j,0])

# get the max and min of L in the window
L_max = L_window.max()
L_min = L_window.min()

# check
print("L max", L_max)
print("L min", L_min)

# print("L values of the window", L_window)

# linear scaling for all pixels of the image
if(L_max==L_min): # if unique luminance e.g. pure color
    outputImage = inputImage
else:
    for i in range(0, rows) :
        for j in range(0, cols) :
            L, u, v = Luv_img[i,j]
            L = round(L) # every L to whole number
            if(L>L_max):
                L=100.0
            elif(L<L_min):
                L=0.0
            else:
                L=round((L-L_min)*100.0/(L_max-L_min))
            Luv_img[i,j]=round(L),u,v 
    # transform back from Luv to BGR
    outputImage = toBgr(Luv_img)

cv2.namedWindow('output:', cv2.WINDOW_NORMAL);
cv2.resizeWindow('output:', 300, 300);
cv2.moveWindow('output:', 680, 40);        
cv2.imshow("output:", np.array(outputImage, dtype = np.uint8 ))
cv2.imwrite(name_output, outputImage);

# check
# print("outputImage", outputImage)
# print(outputImage.shape)

# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
