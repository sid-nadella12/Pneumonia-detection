import tensorflow as tf
from tensorflow.keras import layers, optimizers, models, datasets, utils
from tensorflow.keras.preprocessing.image import load_img
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2, imageio


def predict(image):

	image = np.asarray(image)
	img = cv2.resize(image, (150,150))
	model_path = 'C:/Users/Dell/Desktop/CV Final/vgg16.h5'
	model = models.load_model(model_path)
	model.summary()

	data = np.expand_dims(img, axis=0)
	data = data * 1.0 / 255

	return model.predict(data)

def find_infection(input_image, patient):

    input_image = np.asarray(input_image)

    #if the input image is grayscale, convert into rgb
    if(len(input_image.shape)!=3):
    	input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
     
    #perform segmentation   
    segmented_image = kmeans_segmentation(input_image)
    
    #perform thresholding on segmented image
    thresholded_image = otsu_threshold(segmented_image)
    
    #partition the image into two to get each lung
    right_lung, left_lung = partition_lungs(thresholded_image)
    
    #Track the end pixels of the right lung
    right_tracked_image, right_tracked_max_length = track_right(right_lung)
    
    #Track the pixels of the left lung
    left_tracked_image, left_tracked_max_length = track_left(left_lung)
    
    
    
    if(right_tracked_max_length > left_tracked_max_length):
        slope = (float)(right_tracked_image[0][0] - left_tracked_image[0][0]+250)/(right_tracked_image[0][1] - left_tracked_image[0][1])
        if slope > -1.5:
            print("Infection in LEFT LUNG")
            plot_infection(input_image, left_tracked_image, left_tracked_max_length, 1, patient)
        else:
            print("Infection in RIGHT LUNG")
            plot_infection(input_image, right_tracked_image, right_tracked_max_length, 0, patient)
    else:
        print("Infection in LEFT LUNG")
        plot_infection(input_image, left_tracked_image, left_tracked_max_length, 1, patient)
    
    print("PLOTTING COMPLETE")
    print("Check in the results folder")
       
    
def kmeans_segmentation(input_image):
    #performs segmentation using K-means clustering algorithm.

    #Reshaping the input image
    Z = np.float32(input_image.reshape((-1,3)))

    #Specifying the criteria for K-means i.e., it can stop if either it reaches desired accuracy or it completes maximum iterations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret,label,center=cv2.kmeans(Z,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    # Converting back to int and returning the image
    center = np.uint8(center)
    img = center[label.flatten()]
    img = img.reshape((input_image.shape))
    
    return img


def otsu_threshold(img):
    #Performs Otsu's thresholding

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #image smoothing, thresholding and returning the thresholded image
    blurred_image = cv2.GaussianBlur(img,(5,5),0)
    ret,thresholded_image = cv2.threshold(blurred_image,127,255,cv2.THRESH_BINARY)
    
    return thresholded_image


def partition_lungs(src):
    #Divides the image into two.

    #Converting image to 500x500 => Each image will be of 500x250
    src = cv2.resize(src, (500,500) , interpolation = cv2.INTER_AREA)
    src1 = np.zeros((500,250))
    src2 = np.zeros((500,250))

    #getting an image with right lung
    for i in range(0,500):
        for j in range(0,250):
            src1[i][j] = src[i][j]

    #getting an image with left lung
    k=250
    for i in range(0,500):
        for j in range(0,250):
            src2[i][j] = src[i][k]
            k = k + 1
        k = 250

    return src1, src2
  
   
def track_right(src1):

    #retrieving the reference point in the first row
    for i in range(249,0,-1):
        if(src1[0][i] == 0):
            point_i = i
            break
        
    #tracking all the nearest black pixels to the end
    points_x = []
    points_y = []
    for i in range(0,500):
        for j in range(249,point_i ,-1):
            if(src1[i][j] == 0):
                points_x.append(i)
                points_y.append(j)
                if(j-1 < 0):
                    point_i = j - 1
                else:
                    point_i = 0
                break
     
    lis=[]

    for i,j in zip(points_x, points_y):
        number_of_black = 0
        number_of_white = 0
        
        if i+5>500:
            window_i = 500
        else:
            window_i = i+5
        
        if j-20<0:
            window_j = 0
        else:
            window_j = j-20
    
        for k in range(i,window_i):
            for l in range(j,window_j,-1):
    
                if src1[k][l] == 0:
                    number_of_black = number_of_black + 1
    
                else:
                    number_of_white = number_of_white + 1
    
        lis.append((i,j,number_of_black, number_of_white))
        
        
    #creating a list with the consecutive elements where white>black
    consec = []
    long_consec = [lis[0]]
    for i in range(1,len(lis)):
        if (lis[i][2] < lis[i][3]) and lis[i-1] in long_consec:
            long_consec.append(lis[i])
        else:
            consec.append(long_consec)
            long_consec = [lis[i]]
    
    
    #finding the maximum consecutive points where white>black
    maxList = max(consec, key = len) 
    maxLength = max(map(len, consec)) 
    
    
    return maxList,maxLength    
    
        
def track_left(src2):
    
    
    for i in range(0,250):
        if(src2[0][i] == 0):
            point_i = i
            break
        
    points_x_2 = []
    points_y_2 = []
    for i in range(0,500):
        for j in range(0,point_i):
            if(src2[i][j]==0):
                points_x_2.append(i)
                points_y_2.append(j)
                if(j+1 > 251):
                    point_i = j + 1
                else:
                    point_i = 250
    
                break
                           
    lis=[]

    for i,j in zip(points_x_2, points_y_2):
        number_of_black = 0
        number_of_white = 0
        
        if i+5>500:
            window_i = 500
        else:
            window_i = i+5
        
        if j+20>250:
            window_j = 250
        else:
            window_j = j+20
    
        for k in range(i,window_i):
            for l in range(j,window_j):
    
                if src2[k][l] == 255:
                    number_of_white = number_of_white + 1
    
                else:
                    number_of_black = number_of_black + 1
        
        lis.append((i,j,number_of_black, number_of_white))
        
    #creating a list with the consecutive elements where white>black
    consec = []
    long_consec = [lis[0]]
    for i in range(1,len(lis)):
        if (lis[i][2] < lis[i][3]) and lis[i-1] in long_consec:
            long_consec.append(lis[i])
        else:
            consec.append(long_consec)
            long_consec = [lis[i]]
    
    ##finding the maximum consecutive points where white>black
    maxList = max(consec, key = len)
    maxLength = max(map(len, consec))     
    return maxList, maxLength
        
def plot_infection(src,maxList, maxLength, lung, patient):
    
    bright = 0
    for i in range(1,maxLength):
        if((maxList[i][3]-maxList[i][2]) > (maxList[i-1][3] - maxList[i-1][2])):
            bright = i
    
    #retrieving the coordinates of the centre and drawing a circle.
    x = maxList[bright][0]
    y = maxList[bright][1]
    # print((x,y))
    
    
    src = cv2.resize(src,(500,500), interpolation = cv2.INTER_AREA)
    
    if lung == 0:
        image = cv2.circle(src, (y,x), 100, (255,0,0), 2)
    else:
        image = cv2.circle(src, (y+250,x), 100, (255,0,0), 2)
    
    imageio.imwrite('C:/Users/Dell/Desktop/CV Final/results/'+patient+'-reportsss.jpg',image)
#    im.imwrite("/Users/saisrinijasakinala/Desktop/final/results/mmm.jpg",image2)

if __name__ == "__main__":
    
	# image = load_img('C:/Users/Dell/Downloads/norm.jpg')
    input_path = input("Enter the path of input image:")

    file_name = input_path.split('/')
    file_name =file_name[len(file_name)-1]
    file_name = file_name.split('.')
    patient = file_name[0]
	#image = load_img('C:/Users/Dell/Desktop/Normal/112.jpg')
    image = load_img(input_path)
	# image = np.asarray(image)
    val = predict(image)
    print("Possibility of illness -",val)

    if val > 0.7:
        print("---" + patient + "---Pneumonia - POSITIVE")
        find_infection(image, patient)
    else:
        print("---" + patient + "---Pneumonia - NEGATIVE")