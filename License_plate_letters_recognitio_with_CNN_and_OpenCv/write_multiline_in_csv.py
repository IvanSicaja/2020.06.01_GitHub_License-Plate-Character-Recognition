import csv
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


multiline_string = pd.read_csv("4_Segmented_images_converted_in_CSV/image_1_converted_image_in_one_row.csv")              # assign string
multiline_string = multiline_string.replace('\n','\r\n') # convert newlines to newlines+carriage return

 with open('4_Segmented_images_converted_in_CSV/empty.csv', 'wb') as outfile:
      w = csv.writer(outfile)                            # assign csv writer method
      w.writerow(['sometext',multiline_string])          # append/write row to file