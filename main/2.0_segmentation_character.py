import cv2
import matplotlib.pyplot as plt
import os

delete_files_folder = os.listdir('3.0_Images_of_segmented_cahracters')
print(len(delete_files_folder))
os.chdir(r"3.0_Images_of_segmented_cahracters") 

#Delete files in diractory

for i in range(len(delete_files_folder)):
    file_for_delete='Cropped_image_'+str(i+1)+'.jpg'
    os.remove( file_for_delete)
 
print("The directory is cleaned.")

# varify the path using getcwd() 
cwd = os.getcwd() 
  
# print the current directory 
print("Current working directory is:", cwd) 

os.chdir(r"..") 

# varify the path using getcwd() 
cwd = os.getcwd() 
  
# print the current directory 
print("Current working directory is:", cwd) 

###################################################################

plate_image=cv2.imread("2_Croped_image/Cropped_image.jpg")
#plate_image = cv2.convertScaleAbs(plate_image[0], alpha=(255.0))
cv2.imshow('Original image',plate_image)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()


# convert to grayscale and blur the image
gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray scaled image',gray)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()

blur = cv2.GaussianBlur(gray,(7,7),0)
cv2.imshow('Gausian blur image',blur)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
    


#Applied inversed thresh_binary 
binary = cv2.threshold(blur, 180, 255,
                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cv2.imshow('Binary image',binary)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()

# Applied dilation 
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
cv2.imshow('Dilatation image',thre_mor)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()

#####################################################################################################################

# Create sort_contours() function to grab the contour of each digit from left to right
def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# creat a copy version "test_roi" of plat_image to draw bounding box
test_roi = plate_image.copy()

# Initialize a list which will be used to append charater image
crop_characters = []

# define standard width and height of character
digit_w, digit_h = 15, 15

for c in sort_contours(cont):
    (x, y, w, h) = cv2.boundingRect(c)
    ratio = h/w
    if 1<=ratio<=2.5: # Only select contour with defined ratio
        if h/plate_image.shape[0]>=0.4: # Select contour which has the height larger than 50% of the plate
            # Draw bounding box arroung digit number
            cv2.rectangle(test_roi, (x-5, y-5), (x+5+w, y+5+h), (0, 255,0), 3)

            cv2.imshow('Boundary image',test_roi)
            k = cv2.waitKey(0)
            if k == 27:         # wait for ESC key to exit
                cv2.destroyAllWindows()

            # Sperate number and gibe prediction
            curr_num = thre_mor[y-7:y+h+7,x-5:x+w+5]
            curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
            _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            crop_characters.append(curr_num)



print("Detect {} letters...".format(len(crop_characters)))

for i in range(len(crop_characters)):
    cv2.imshow('All segmentad letters',crop_characters[i])
    cv2.imwrite('3.0_Images_of_segmented_cahracters/Cropped_image_'+str(i+1)+'.jpg',crop_characters[i])
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()

print("Segmented cahracters saved in directory.")
cv2.imshow('All segmentad letters',test_roi)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()


'''
fig = plt.figure(figsize=(10,6))
plt.show()
cv2.imshow('fig image',fig)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
cv2.imshow('Segmented image',test_roi)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()


grid = gridspec.GridSpec(ncols=len(crop_characters),nrows=1,figure=fig)

for i in range(len(crop_characters)):
    fig.add_subplot(grid[i])
    plt.axis(False)
    plt.imshow(crop_characters[i],cmap="gray")
    plt.show()

    '''
