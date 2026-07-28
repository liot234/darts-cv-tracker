# Real-Time Dart Scoring & Tracking System with statistical analysis
A low-latency computer vision pipeline that tracks, maps, and scores physical dart impacts using dual-camera feeds.NO ML.

## The process
1.**Setting up the cameras for success** .This solution relies on a two camera setup, one camera at the top of the board pointing down parallel to the board  and the other on the left side pointing parallel to the board. The top camera is in charge of capturing the horizontal displacement of the dart
and the left camera the vertical.
<img width="2019" height="1492" alt="image" src="https://github.com/user-attachments/assets/4fe8e550-3ca2-4a04-a814-f7a4f1ee1a0c" />
Here is an image of what my bedroom camera setup looks like!

Things to note:

1. The cameras are moved sufficiently back so that they can capture the whole board 

2. The cameras are not of the same type, in an ideal world they would be since that would make the image processing MUCH simpler.

3. I have added two extra lighting sources, one is the lamp you can see one is a lamp under lighting the right side of the darts
and the background

4. I have added a plain bed sheet where the left camera is pointing to, this increases contrast and helps isolate the darts better.

<!-- end of the list -->
2.**Image processing**. Arguably the most important aspect of this project, since without being able to cleanly extract the dart you cannot
meaningfully score that dart.Here the cv2 library has been my best friend, it has many many inbuilt image processing methods.
   <img width="1093" height="476" alt="Screenshot 2026-06-04 132335" src="https://github.com/user-attachments/assets/8ac35072-eb8f-4f03-b574-7f1424ce35ea" />
Here is the basic result the image processing I performed within the imagediff script.On a high level it computes the difference in two images
one images has a dart in the board, and the other doesn't(in reality the image has darts but they are invisible to the image processor). I then go on to 
refine my algorithm from the image above in order to reduce noise and isolate the dart tip even better.

Improvements I implemented:

1. **Morphological operations**: I used dilation and erosion, which are two algorithms which each use a box called a kernel to sweep across the image array
 and basically either expand or delete the main "blobs" of an image.The below image illustrates how these algorithms work.
  <img width="685" height="226" alt="image" src="https://github.com/user-attachments/assets/82aad9a2-aa70-4710-be45-96d8f17f3705" />

   Using dilation and then erosion on my image differences helps build up the dart so it is clearly identifiable and reduce noise.

2. **Gausian blur**: Before applying the image difference I applied Gaussian blur to both input images, this helps reduce overall noise and produces a cleaner image.

3. **Reducing the processing frames**: Unlike the image of the isolated darts I showed, in my final algorithm I only processed the pixels from the tip of the dart to
about a 1/5 the way up the dart, this helped speed up image processing since the image matrices are smaller, and it also helped reduce the effect of noise.


<!-- end of the list -->
3.**Scoring the darts and statistical analysis**.
With the darts now isolated, I leveraged a very powerful cv2 feature called contours which creates a matrix object of the pixels of my isolated
dart. I then search through this matrix and find the coordinate of the tip from both cameras. We now have darts with x and y co-ordinates, we are almost there! There is one
big issue,Distortion, due to the differences in the cameras and the distortion in there lenses, the points do not get mapped as a linear grid, instead it acts more like a fish-eye
curved grid, where points which equal radius get mapped to an ellipse not a circle. The fix: Homography, this applies a linear transformation from the points on my default distorted grid onto a perfectly square gird where the center is the bullseye.
With my new square grid I was able to calculate angles and distances from the bullseye. This now makes scoring easy using the angle to identify the number on the board and the radius to see if they
hit a double or triple. 

Admittedly, my angle calculations have been much more accurate compared to the radius, this is because homography isn't the perfect solution and the cameras have some level of 
distortion which is very hard to correct for.

Moving onto the statistical analysis portion, for now I have just used numpy's inbuilt means and norm functions to calculate the centre of mass of a collection 
of darts throws and then calculate the mean deviation from that centre. This tells you how accurate you are, effectively how well you can closly hit your desired location.


# Real-Time Dart Scoring & Tracking System

A low-latency computer vision pipeline that tracks, maps, and scores physical dart impacts using dual-camera feeds. **(No Machine Learning involved)**.

## 1. Hardware & Environment Setup

This solution relies on a two cam setup to capture x,y position on the board:
* **Top Camera:** Positioned above the board, pointing downward to capture horizontal (X-axis) displacement.
* **Side Camera:** Positioned to the left of the board, pointing parallel across the face to capture vertical (Y-axis) displacement.

![Bedroom Camera Setup](setup_image_filename.jpg)
*(Replace `setup_image_filename.jpg` with the actual file name of your setup image)*

**Setting up the camera and lighting:**
* **Field of View:** Both cameras are mounted at a sufficient distance to capture the entire dartboard within their respective frames.
* **Lighting & Contrast:** Two auxiliary lighting sources (overhead and right-side under-lighting) were introduced to eliminate shadows. Additionally, a plain white backdrop (bed sheet) was placed behind the board to artificially increase contrast, which significantly aids the vision algorithm in isolating the darts.
* **Hardware Asymmetry:** The cameras used are of different models, which introduces varying levels of lens distortion that are corrected mathematically downstream.

## 2. The Computer Vision process

Cleanly extracting the dart tip is the most critical aspect of the pipeline. The project leverages OpenCV for high-speed image processing, relying on frame differencing (comparing a background reference frame to a frame containing a dart). 

![Basic Image Difference](difference_image_filename.jpg)
*(Replace `difference_image_filename.jpg` with the image of your basic isolation result)*

To reduce noise and isolate the sub-pixel coordinates of the dart tip, several optimizations were implemented:

* **Gaussian Blurring:** Applied to both the reference and current frames before taking the absolute difference. This smooths minor lighting fluctuations and produces a cleaner delta image.
* **Region of Interest (ROI) Cropping:** Rather than processing the entire frame, the algorithm crops the processing matrix to isolate only the bottom 1/5th of the dart (the tip). This drastically reduces the size of the image matrices, minimizing latency and eliminating peripheral room noise.
* **Morphological Operations:** The pipeline utilizes custom dilation and erosion algorithms (closing). A mathematical kernel sweeps across the image array to expand the primary "blobs" (the dart) and erode away disconnected noise, ensuring the dart tip remains a solid, unbroken contour.

![Morphological Operations Illustration](morphology_image_filename.jpg)
*(Replace `morphology_image_filename.jpg` with your illustration of dilation and erosion)*

## 3. Mathematical Scoring & Statistical Analysis

Once the dart is isolated, the pipeline utilizes OpenCV's contour mapping to create a matrix object of the pixels, allowing the system to pinpoint the (X, Y) coordinates of the tip from both camera angles.

### Correcting Lens Distortion
Because the two cameras have distinct, non-linear lens distortions (acting similarly to a fish-eye effect), the raw coordinates map to an ellipse rather than a true circle. To solve this, the pipeline applies a **Homography Matrix**. This applies a linear transformation to map the distorted coordinates onto a perfectly square grid where the exact center acts as the bullseye.

### Trigonometric Scoring
With the coordinates translated to a unified Cartesian plane, the system uses basic trigonometry to calculate:
* **Angle:** Determines the specific segment number (1-20) hit on the board.
* **Radius:** Determines the distance from the center (identifying singles, doubles, or triples). 
*(Note: Due to residual hardware distortion that is difficult to completely map, angle calculations currently yield higher accuracy than radial distance calculations).*

### Statistical Analytics
The system tracks historical throws and utilizes NumPy to calculate grouping consistency. By computing the center of mass (centroid) of a collection of throws, the pipeline calculates the mean radial deviation from that center. This outputs a quantifiable metric of player accuracy and grouping tightness over time.
### Current Limitations (Environment Specific)
Please note that this project is currently highly tailored to a specific physical setup. The Region of Interest (ROI) cropping, binary thresholding values for lighting, and the `DISTORTED_POINTS` used for the homography matrix are hardcoded. 

To use this on a new setup, you will need to manually measure and update these pixel coordinates in `main.py`. A dynamic calibration tool is planned for a future release.
   
