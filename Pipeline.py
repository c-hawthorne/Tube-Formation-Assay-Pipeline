##RUN PYTHON SCRIPT
#python C:\Users\User\Documents\Pipeline.py
#
#




##############################
#IMAGE DIRECTORIES
#Where is your folder of images - must include *.tif at end.

IMAGE_FOLDER = 'C:\\Users\\User\\Desktop\\Images\\*.tif'
#What is the name of this Project?
TITLE = 'Project_Title'
#Where would you like the created image folders placed?
PLACEMENT = 'C:\\Users\\User\\Documents\\'
##############################




















#############################
#######PIPELINE################
print("Start!")

###############################
####FUNCTIONS DEFINED#######
####Folder Function
import os
def folder(x):
    if os.path.exists(PLACEMENT + x):
        print(x + ' folder already exists.')
    else: 
        os.makedirs(PLACEMENT + x)
        print(x + ' folder created')

####Area Function
def area(x,y):
  AREA = label(x, background = 0)
  #print(AREA)
  plt.imsave(str(m)+y+'Area.tif', numpy.array(AREA))
  properties = regionprops(AREA)
  proparea = [p.area for p in properties]
  #centre mean of image
  proparea2 = [p.centroid for p in properties]
  #co-ordinates
  proparea3 = [p.coords for p in properties]
  areatype = img_as_ubyte(x)
  areatype = grey2rgb(areatype)
  areatype = adjust_log(areatype, gain = 50)
  for a in proparea2:
    #print(a[0])
    #print(a[1])
    rr,cc = circle(a[0],a[1],1) 
    areatype[rr,cc] = (0, 255, 255)
  proparea = DataFrame(proparea, columns = ['Area'])
  proparea2 = DataFrame(proparea2, columns = ['row - y','column - x'])
  merged = concat([proparea,proparea2], axis = 1)
  plt.imsave(str(m)+y + '_arearegions.tif', numpy.array(areatype))
  merged.to_csv(str(m)+y+'_proparea.csv') 
  print("Calculated area saved!")

#############################
########PIPELINE##############
###############################
#Import modules outside of loop, so then you aren't constantly importing. 
import exifread
import glob
import skimage
import imageio
import matplotlib.pyplot as plt
import numpy
##################################
from skimage.color import rgb2grey, grey2rgb
from skimage import io
####################################
from skimage.exposure import adjust_log
import matplotlib.pyplot as plt
####################################
from skimage.filters import gaussian
###################################
from skimage.filters import threshold_otsu, threshold_yen, threshold_local
from skimage.filters import try_all_threshold, threshold_mean
import matplotlib.cm
###########################################
from skimage.morphology import remove_small_holes
###################################
from skimage.morphology import opening
from skimage.morphology import closing
from skimage.morphology import disk
###################################
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle, line
###############################
from skimage.morphology import skeletonize
from skimage.util import invert
##################################
import skan
import pandas
from pandas import DataFrame, concat
from skan import csr
##################################
import cv2
#
from skimage.measure import label, regionprops
#################################
#http://scikit-image.org/docs/dev/user_guide/data_types.html
from skimage import img_as_float, img_as_ubyte,img_as_uint, img_as_int


##################################
#SELECTING IMAGE/FOLDER
#ref: https://docs.python.org/3/library/glob.html
#ref: https://docs.python.org/3/library/os.path.html
directory = glob.glob(IMAGE_FOLDER)

hashtag = '########################'
print(directory)
c=0
for i in directory:
  print(hashtag)
  directory_no = len(directory)
  c=c+1
  print(str(c) + ' out of ' + str(directory_no))
  print(i)
  n = os.path.splitext(os.path.basename(i))
  m = n[0]
  images = imageio.imread(i, 'TIFF')
  #dimensions
  dim = numpy.shape(images)
  strdim = str(dim)
  print("Dimensions of image are " + strdim)
  dim = numpy.shape(images)
  #print(dim)
  #print(dim[0])
  #print(dim[1])
  singledim = dim[0]
  #SCALE
  SCALE = (1/512)* singledim
  strSCALE = str(SCALE)
  print("Scale is set at " + strSCALE)
  #1 for 512 and 4 for 2048
  #Extraction of Metadata from Image
  #ref: https://pypi.org/project/ExifRead/#description
  read = open(i,'rb')
  tags = exifread.process_file(read, details=False)
  #tags is a dictionary
  #get the ImageDescription for unit
  #print(tags["Image ImageDescription"])
  #print(tags)
  hello = tags.get("Image ImageDescription")
  hello = str(hello)
  findthis = hello.find('micron')
  if findthis == -1:
    unit = 'no'
    print("Image measurement not in microns.")
    strpixelspermicron = 'pixels per micron not calculated'
  
  else:
    unit = 'microns'
    print("Image measurement in microns")
    resolution = tags.get('Image XResolution')
    #print(resolution)
    resolution = str(resolution)
    #print(resolution)
    #ref: https://docs.python.org/3/library/stdtypes.html
    resolution = resolution.replace('/',' ')
    #print(resolution)
    resolution = resolution.split()
    print(resolution)
    ratio1 = float(resolution[0])
    ratio2 = float(resolution[1])
    pixelspermicron = ratio1 / ratio2
    #print(pixelspermicron)
    strpixelspermicron = str(pixelspermicron)
    print("Resolution is " + strpixelspermicron)
    
  #Transform from RGB to Greyscale
  #ref: http://scikit-image.org/docs/dev/user_guide/transforming_image_data.html
  #ref: http://scikit-image.org/docs/dev/api/skimage.color.html#skimage.color.rgb2grey
  #ref: https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.imsave.html
  grey = rgb2grey(images)
  #folder creation
  folder(TITLE + '_1_Greyscale')
  #change directory
  os.chdir(PLACEMENT + TITLE + '_1_Greyscale')
  #save image
  plt.imsave(str(m) + "_Greyscale.tif", numpy.array(grey), cmap = plt.matplotlib.cm.gray)
  print("Greyscale image saved!")
  ##############################
  #CONTRAST
  #ref: http://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_log_gamma.html#sphx-glr-auto-examples-color-exposure-plot-log-gamma-py
  CONTRAST = adjust_log(grey, gain = 1.5)
  #save image
  plt.imsave(str(m)+'_Contrast.tif', numpy.array(CONTRAST),  cmap= plt.matplotlib.cm.gray)
  print("Contrast image saved!")
  ##############################
  #GAUSSIAN BLUR OBJECT
  #ref: https://scikit-image.org/docs/dev/api/skimage.filters.html?highlight=gaussian#skimage.filters.gaussian
  from skimage.filters import gaussian
  gaussian = gaussian(CONTRAST, sigma=2)
  #save image
  plt.imsave(str(m)+'_Gaussian.tif', numpy.array(gaussian),  cmap= plt.matplotlib.cm.gray)
  print("Gaussian image saved!")
  os.chdir(PLACEMENT)
  ###############################
  #THRESHOLD GREYSCALE OBJECT
  #ref: http://scikit-image.org/docs/stable/auto_examples/segmentation/plot_thresholding.html#sphx-glr-auto-examples-segmentation-plot-thresholding-py
  
  #threshold = threshold_otsu(gaussian) #otsu method 
  #print(threshold)
  #thresholdyen = threshold_yen(gaussian) #yen method
  #print(thresholdyen)
  #binary = gaussian <= threshold
  #binary2 = gaussian <= thresholdyen
  #binary3 = gaussian <= 0.015
  #thresholdadjusted = threshold*0.08
  #binary4 = gaussian <= thresholdadjusted
  
  #MEAN
  thresholdmean =  threshold_mean(gaussian)*0.60
  binary5 = gaussian <= thresholdmean


  #fig,ax = try_all_threshold(gaussian, figsize =(10,8), verbose=False)
  #plt.show

   
  #folder creation
  folder(TITLE + '_2_Threshold_Greyscale')
  #change directory
  os.chdir(PLACEMENT + TITLE + '_2_Threshold_Greyscale')
  #save image
  #plt.imsave(str(m)+'binary.tif', numpy.array(binary),  cmap= plt.matplotlib.cm.gray)
  #plt.imsave(str(m)+'binary2.tif', numpy.array(binary2),  cmap= plt.matplotlib.cm.gray)
  #plt.imsave(str(m)+'binary_0.015.tif', numpy.array(binary3),  cmap= plt.matplotlib.cm.gray)
  #plt.imsave(str(m)+'binary_adjustedotsu.tif', numpy.array(binary4),  cmap= plt.matplotlib.cm.gray)
  plt.imsave(str(m)+'_binary_mean.tif', numpy.array(binary5), cmap= plt.matplotlib.cm.gray)
  print("Threshold image saved!")
  
  #Image Data Histogram
  #ref: https://matplotlib.org/2.0.2/users/image_tutorial.html
  #ref: http://scikit-image.org/docs/dev/auto_examples/xx_applications/plot_thresholding.html#id2
  #ref: https://matplotlib.org/gallery/pyplots/pyplot_text.html#sphx-glr-gallery-pyplots-pyplot-text-py
  plt.hist(gaussian.ravel(), bins=256)
  plt.axvline(thresholdmean, color='r')
  plt.xlabel('Transformed pixel values')
  plt.ylabel('Number of pixels')
  plt.title('Histogram of image data')
  plt.tight_layout()
  plt.savefig(str(m)+'histogram.png')
  plt.close()
  
  ###############################
  #AREA OF THRESHOLD OBJECT
  areathreshold = invert(binary5)
  area(areathreshold,'_area_threshold')
  #change directory
  os.chdir(PLACEMENT)
 
  ###############################
  #CLEAN - REMOVE SMALL HOLES
  #ref: http://scikit-image.org/docs/stable/api/skimage.morphology.html?highlight=remove_small_holes#skimage.morphology.remove_small_holes
  clean = skimage.morphology.remove_small_holes(binary5)
  folder(TITLE + '_3_Cleanup')
  #change directory
  os.chdir(PLACEMENT + TITLE + '_3_Cleanup') 
  plt.imsave(str(m)+'_clean.tif', numpy.array(clean),  cmap= plt.matplotlib.cm.gray)
  ########################################
  #########OPENING AND CLOSING FOR CLEAN UP
  #ref: http://scikit-image.org/docs/stable/auto_examples/xx_applications/plot_morphology.html#sphx-glr-auto-examples-xx-applications-plot-morphology-py
  #
  if SCALE == 1:
    CLEANED = opening(clean, disk(1))
  else:
    value = (SCALE)
    CLEANED = opening(clean, disk(value))
      
  plt.imsave(str(m)+'_cleaning_opening.tif', numpy.array(CLEANED),  cmap= plt.matplotlib.cm.gray)
  
  if SCALE == 1:
     CLEANEDAGAIN = closing(CLEANED, disk(1))
  else:
    value = (SCALE)
    CLEANEDAGAIN = closing(CLEANED, disk(value))

  plt.imsave(str(m)+'_cleaning_closing.tif', numpy.array(CLEANEDAGAIN),  cmap= plt.matplotlib.cm.gray)
  
  print("Cleaned image saved!")
  ###############################
  #AREA OF OPEN OBJECT
  COPYCLEANED = CLEANED
  areacleaned = invert(COPYCLEANED)
  area(areacleaned,'_area_opening')
  ###############################
  #AREA OF CLOSING OBJECT
  COPYCLEANEDAGAIN = CLEANEDAGAIN
  areacleanedagain = invert(CLEANEDAGAIN)
  area(areacleanedagain,'_area_closing')
  
  ########################################
  ######Circular Hough Transform
  #http://scikit-image.org/docs/dev/auto_examples/edges/plot_circular_elliptical_hough_transform.html?highlight=circle 
  #edges = canny(CLEANEDAGAIN)
  #range
  if SCALE == 1:
    hough_radii = numpy.arange(20, 30, 2)
    #if the range places a circle outside the boundaries of the image, an error will occur 
  else:
    start = ((SCALE/2)*20)
    end = ((SCALE/1.6)*20)
    hough_radii = numpy.arange(start,end,2)
  print(hough_radii)  
  #array([40,42,44,46,48])
  hough_spaces = hough_circle(CLEANEDAGAIN, hough_radii)
  #hough_circle(image, radius, normalize=True, full_output=False)
 
  accum, cx, cy, rad = hough_circle_peaks(hough_spaces, hough_radii,total_num_peaks=100)
  #hough_circle_peaks(hspaces, radii, min_xdistance=1, min_ydistance=1, threshold=None, num_peaks=inf)
  #4 arrays returns for accum, cx, cy and rad.
  
  #ref: http://scikit-image.org/docs/dev/api/skimage.draw.html#skimage.draw.circle
  #skimage.draw.circle(r, c, radius, shape=None), r and c are the centre co-ordinate of circle
  #ref: zip - makes an iterator that aggregates elements from each of the iterables (https://docs.python.org/3.4/library/functions.html#zip)
  for center_y, center_x, radius in zip(cy, cx, rad):
    circy, circx = circle(center_y, center_x, radius)
   # print(circy)
   # print(circx)
    CLEANEDAGAIN[circy, circx] = (1) #colour
    #0 for black
    #1 for white

  #plt.imsave(str(m)+'edges.tif', numpy.array(edges),  cmap= plt.matplotlib.cm.gray)
  plt.imsave(str(m)+'_removal.tif', numpy.array(CLEANEDAGAIN),cmap= plt.matplotlib.cm.gray)
  print("Removal image saved!")
  os.chdir(PLACEMENT)
  
  #####################################
  #SKELETONISE CLEANED THRESHOLD OBJECT
  #ref: http://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html#sphx-glr-auto-examples-edges-plot-skeleton-py
  CLEANEDAGAIN = invert(CLEANEDAGAIN)
  skeleton = skeletonize(CLEANEDAGAIN)
  #skeleton2 = invert(skeleton)
  #Make a folder 
  folder(TITLE + '_4_Image_Analysis')
  #change directory
  os.chdir(PLACEMENT + TITLE + '_4_Image_Analysis') 
  ###
  plt.imsave(str(m)+'_skeleton.tif', numpy.array(skeleton), cmap= plt.matplotlib.cm.gray)
  #plt.imsave(str('skel_') + str(m)+'skeleton2.tif', numpy.array(skeleton2), cmap= plt.matplotlib.cm.gray)
  print("Skeleton image saved!")
  #plt.imsave(str(m)+'CLEANINVERT.tif', numpy.array(CLEANEDAGAIN), cmap= plt.matplotlib.cm.gray)
  ##############################
  ###Threshold binary skeletonise
  skeletonthreshold = invert(binary5)
  skeletonthreshold = skeletonize(skeletonthreshold)
  plt.imsave(str(m)+'_skeletonthreshold.tif', numpy.array(skeletonthreshold), cmap = plt.matplotlib.cm.gray)


  ####################################
  #SKELETON ANALYSIS
  #ref: https://jni.github.io/skan/getting_started.html
  #In pixels
  branch = csr.summarise(skeleton)
  #save calculated branches to csv
  branch.to_csv(str(m)+'_branches.csv') 
  print("Calculated branches saved!")
  ####################################
  #IMAGE OVERLAY
  #ref: https://docs.opencv.org/trunk/d0/d86/tutorial_py_image_arithmetics.html
  #ref: http://scikit-image.org/docs/dev/user_guide/numpy_images.html
  #ValueError: Image RGB array must be uint8 or floating point; found uint16
  colourskeleton = skeleton
  colourskeleton = img_as_ubyte(colourskeleton)#8bit
  colourskeleton = grey2rgb(colourskeleton)
  #print(colourskeleton.shape)
  #colourskeleton = adjust_log(colourskeleton, gain = 80)
  mask = colourskeleton[:,:,0] > 0
  #where it is greater than 0 and 0 is black
  colourskeleton[mask] = [0,255,0] 
  #replace pixels greater than 0 with green
  #Contrast used instead.
  colourgrey = CONTRAST 
  colourgrey = img_as_ubyte(colourgrey) #8bit
  colourgrey = grey2rgb(colourgrey)
  plt.imsave(str(m)+'_colour_skeleton.tif', numpy.array(colourskeleton))
  if SCALE == 1:
    cs = 0.2
    cg = 0.8
    ad = 1.5
  else:
    cs = 0.3
    cg = 0.7
    ad = 1
    
  overlay = cv2.addWeighted(colourskeleton,cs,colourgrey,cg,0)
  overlay = adjust_log(overlay, gain = ad)

  plt.imsave(str(m)+'_overlay_final.tif', numpy.array(overlay))
  
  
  ###########################################
  skeletonthreshold = img_as_ubyte(skeletonthreshold)
  skeletonthreshold = grey2rgb(skeletonthreshold)
  skeletonthreshold = adjust_log(skeletonthreshold, gain = 80)
  mask = skeletonthreshold[:,:,0] > 0
  #where it is greater than 0 and 0 is black
  skeletonthreshold[mask] = [0,255,0] 
  overlay2 = cv2.addWeighted(skeletonthreshold,cs,colourgrey,cg,0)
  overlay2 = adjust_log(overlay2, gain = ad)

  plt.imsave(str(m)+'_overlay_original.tif', numpy.array(overlay2))
  print("Overlay image saved!")
  ####################################
  #AREA OF IMAGE, CO-ORDINATES OF FEATURES
  #ref: http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
  #ref: http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_label.html
  #ref: https://www.youtube.com/watch?v=STnoJ3YCWus
  #ref: https://gis.stackexchange.com/questions/72458/export-list-of-values-into-csv-or-txt-file/72476
  #ref: http://scikit-image.org/docs/dev/auto_examples/edges/plot_shapes.html#sphx-glr-auto-examples-edges-plot-shapes-py
  #ref: https://stackoverflow.com/questions/12850345/how-to-combine-two-data-frames-in-python-pandas
  import copy
  #CLEANEDAGAIN has been inverted, black and white are reversed.
  
  
  AREA = label(CLEANEDAGAIN, background = 0)
  #print(AREA)
  plt.imsave(str(m)+'_Area.tif', numpy.array(AREA))
  properties = regionprops(AREA)
  proparea = [p.area for p in properties]
  #centre mean of image
  proparea2 = [p.centroid for p in properties]
  #co-ordinates - to draw on to an image!
  proparea3 = [p.coords for p in properties]
  CLEANEDAGAIN = img_as_ubyte(CLEANEDAGAIN)
  CLEANEDAGAIN = grey2rgb(CLEANEDAGAIN)
  CLEANEDAGAIN2 = adjust_log(CLEANEDAGAIN, gain = 50)
  CLEANED3 = copy.deepcopy(CLEANEDAGAIN2)
  for a in proparea2:
    #print(a[0])
    #print(a[1])
    rr,cc = circle(a[0],a[1],1) 
    CLEANEDAGAIN2[rr,cc] = (255,0,0)
    #skimage.draw.circle(r, c, radius, shape=None)
  proparea = DataFrame(proparea, columns = ['Area'])
  proparea2 = DataFrame(proparea2, columns = ['row - y','column - x'])
  merged = concat([proparea,proparea2], axis = 1)
  plt.imsave(str(m)+'_arearegions.tif', numpy.array(CLEANEDAGAIN2))
  merged.to_csv(str(m)+'_proparea.csv') 
  print("Calculated area saved!")
  
  #COLOUR BY AREA
  #ref: http://scikit-image.org/docs/dev/user_guide/numpy_images.html
  number1 = (singledim**2)*0.01 
  #print(number1)
  number2 = (singledim**2)*0.005
  #print(number2)
  for selection in properties:
      if selection.area > number1:
          for part in selection.coords:
            co1 = part[0]
            co2 = part[1]
            CLEANED3[co1,co2] = (0,255,0)
      elif selection.area > number2:
            for part in selection.coords:
                co1 = part[0]
                co2 = part[1]
                CLEANED3[co1,co2] = (255,191,0)
      else:
            for part in selection.coords:
                co1 = part[0]
                co2 = part[1]
                CLEANED3[co1,co2] = (255,0,0)
          
  plt.imsave(str(m)+'_Colour_Area.tif', numpy.array(CLEANED3))

  #OVERLAY_COLOURBYAREA

  CLEANED3

  overlayCBA = cv2.addWeighted(CLEANED3,cs,colourgrey,cg,0)
  plt.imsave(str(m)+'_areaoverlay.tif', numpy.array(overlayCBA))

  #Write output to file
  #ref: https://www.datacamp.com/community/tutorials/reading-writing-files-python
  #ref: https://docs.python.org/3/library/functions.html#open
  #ref: http://forum.imagej.net/t/converting-pixel-size-what-metadata-to-use/1442
  file=open(str(m)+"properties.txt", mode="a+", encoding="utf-8")
  file.write(str(i)+"\n")
  file.write(str(c)+"\n")
  file.write("Dimensions of image are "+ strdim +"\n")
  file.write("Scale is set at " + strSCALE +"\n")
  file.write("Resolution is " + strpixelspermicron+"\n")
  #file.write("Otsu method" + str(threshold)+"\n")
  #file.write("Yen method" + str(thresholdyen)+"\n")
  #file.write("Hard coded is 0.015\n")
  file.write("Threshold value is " + str(thresholdmean)+"\n")
  file.close()
  print("Property file saved!")
  
  



  
print("End!")




