# load and evaluate a saved model
from numpy import loadtxt
from PIL import Image, ImageDraw, ImageFilter
from keras.models import load_model
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import cv2


delete_files_folder = os.listdir('3.2_Extended_segmented_images')
print(len(delete_files_folder))
os.chdir(r"3.2_Extended_segmented_images") 

#Delete files in diractory

for i in range(len(delete_files_folder)):
    file_for_delete='Extended_sgmented_character_'+str(i+1)+'.jpg'
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

###############################################################################


file_counter = os.listdir('3.0_Images_of_segmented_cahracters')
print(len(file_counter))


for i in range(len(file_counter)):

    black_background = Image.open("3.1_Black_background_image_28x28/Black_background.jfif")
    #black_background.show()

    final_image=black_background.copy()
    segmented_character = Image.open("3.0_Images_of_segmented_cahracters/Cropped_image_"+str(i+1)+".jpg")
    segmented_character.show()

    final_image.paste(segmented_character,(7,7))
    final_image.save("3.2_Extended_segmented_images/Extended_sgmented_character_"+str(i+1)+".jpg", quality=95)
    final_image.show()
