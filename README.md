**Vehicle Detection Project**

## Image features
This project makes uses of a SVM classifier to detect vehicles in an image. Instead of directly providing the image pixel data as input to the classifer, an image processing pipeline extracts important features from the images and then feeds them into the classifier. Selecting the right features is key as superfluous features can slow down the pipeline without aiding detection. In contrast, training the classifier with very few features means the accuracy of the classifier is too low. The features considered for extraction were: 

1) Color histogram

A histogram analysis is performed on each of the three color components and the bin-counts are then flattened into an feature array. The choice of color format plays an important role in deciding which of the color components is useful for classification. The number of bins decide how much information is captured i.e., how big the feature array is. Here's the histogram for a test image. 

![alt text](https://github.com/dhnkrn/Vehicle-Detection/blob/master/output_images/color_histogram.png?raw=true)

2) Spatially binned pixel data

This essentially is a resizing operation that provides some invariance to image size for purpose of classification. 
![alt text](https://github.com/dhnkrn/Vehicle-Detection/blob/master/output_images/spatial_binning.png?raw=true)

3) Histogram Of Gradients (HOG)

This by far appears to be the most important feature to classify cars. HOG captures the shape information of objects very well. This pipeline makes HOG vectors from all three color components.
![alt text](https://github.com/dhnkrn/Vehicle-Detection/blob/master/output_images/hog_vizualization.png?raw=true)

#### Parameter selection and tuning

The approach to selecting the features was to train the SVM classifier for all combination of features and choose the one with the best testing accuracy. 

```
def get_configs():
	while(True):
        for color_space in ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']:
            for spatial_feat in [True, False]:
            	for hist_feat in [True, False]:
                	for hog_feat in [True, False]:
                    	if spatial_feat == False and hist_feat == False and hog_feat == False:
                                        	continue
                		for hog_channel in ['ALL',0,1,2]:
                    		for cell_per_block in [2,4]:
                        		for pix_per_cell in [8,6,10]:
                            		for orient in [9,8,10]:
                                		yield [color_space, spatial_feat, hist_feat, hog_feat,\
                                        	   hog_channel, cell_per_block, pix_per_cell, orient]
                                        
```
By gradually narrowing down the configurations, it was apparent that 
1) YUV, HSV and LUV performed the best
2) Color Histogram had very little effect
3) The accuracy was better with ALL hog channels.
4) The rest of the parameters did not matter much.

## Classification

The carefully selected features are concatenated into an array and normalized with a StandardScaler. A LinearSVC classifier forms the end of pipeline to detecting whether an image (or a window) is a car or not based on the feature array input. Training is performed with a data set containing 64x64 pixel, 3-channel car and non-car images. The data set has about 8792  cars and 8968  non-cars. Training was also one the primary considerations in selecting features.

Feature extraction takes 74.32s

Training takes 24.77s and results in an accuracy = 0.991

![alt text](https://github.com/dhnkrn/Vehicle-Detection/blob/master/output_images/feature_concatenation_normalization.png?raw=true)


## Sliding window search

The feature extraction and classification pipeline operates on 64x64x3 sized images. However, the images from the video are 1280x720x3 sized. Also, another problem is that the classifier is trained to detect cropped single car images, but the image from the video can contain several cars, of different sizes and at different locations within the image. To solve this problem, a sliding window that extracts a portion of the image and feeds it to the pipeline is implemented. The windows that results in a True detection for a car are then further processed.

![alt text](https://github.com/dhnkrn/Vehicle-Detection/blob/master/output_images/sliding_window.png?raw=true)

## Detecting cars

The classifier identifies sections(window) in an image that contain a car and outputs the co-ordinates of those windows. However, the detection has two major issues:
1) False positives where a non-car part of the image identified as a car
2) The classifier outputs multiple windows for a car.

![alt text](https://github.com/dhnkrn/Vehicle-Detection/blob/master/output_images/dups_and_falsePos.png?raw=true)

False positives are handled by creating a heatmap of the image that accumulates detections and checking if the accumulated heat is higher than a threshold. The idea is that the classifier outputs a weak detection in the false positive area with fewer windows.

Duplicate detections are handled by using the scipy.ndimage.measurements.label() function that groups connected pixels in the heatmap as belonging to one label.

![alt text](https://github.com/dhnkrn/Vehicle-Detection/blob/master/output_images/heatmap_detection.png?raw=true)


I also have running average filter that tries smooothen detections between frames. But this does not seem to be working great at this point.


#### Video

Here's a [link to my video result](https://github.com/dhnkrn/Vehicle-Detection/blob/master/project_video_out_clipped.mp4?raw=true)


## Discussion

This project is based on the code in Udacity lesson exercises. Selecting the features and optimzing the parameters for feature extraction required several rounds of iterating. Post-processing the detection to reject false detections and combine duplicates seemed to be the hardest part. Overall, the approach seem very sensitive to the choice of features, parameters and requires a fair amount of post-processing after detection.

