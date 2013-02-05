IMFinalB
========

The second final project for the Image processing subject

Description
===========

Main:detecting motions
----------------------

The algorithm expressed by the following code(main.py):

    mask = comp( mkShowByGen("a"),
                 mapGenWithState(g),
                 mkShowByGen("b"),
                 mapGen(dilate(getKernel(3))),
                 mapGen(mkThresholdFn(20)),
                 varianceTemporalFilter(50, framed=False), 
                 mapGen(mycvtConvert())
                 )

1)Firstly we just convert RGB image to the grayscale one. Secondly we used a temporal filter to detect movements.

2)In that system for detecting movament we used a filter that calculates for each pixel difference over time. So if we have sequence of images:

    [[1]] [[1]] [[6]] [[1]] [[1]]

the result for the central pixel will be the following:

    abs(x-E) = abs(6 - ((1+1+6+1+1)/5)) = abs(6 - (10/5)) = 4

Here we used size of a frame = 5.
As we can see if all the pixels were the same we would obtain 

    abs(x-(x*k/k)) = 0

for each x and k, where x is a repeated element and k is a size of a frame
We used size of a frame = 50 frames(it is two seconds)

3)The next step is to convert grayscale image to the binary one in order toobtain a mask -  a region to be classified as a human, as a car or as something else.
We set constant low threshold = 20 to see more movements.

4)In that step we aplied dilation to obtain bigger regions, because our temporal filter detect movements mostly on the contours and we want to obtain one big region, not smaller regions.

5)After we found the mask with motions we need to draw rectangles on the original image and this step is for that.

Function mkShowByGen("b") is for drawing stream of images to the screen. As we see we draw both mask and a resulting image.

We used that temporal filter instead of median temporal filter because it is more effecient(meaning processor resources). And it is more effecient because it is expressed through operations on the numpy datatypes:

    a = inTheMiddle(imgList)
    b = (sum(map(np.int32, imgList)) / len(imgList))
    c = np.uint8(abs(np.int32(a)-np.int32(b)))

Which are very fast. 
But we expect median temporal to be more exact in motion detection. 

Main:classification
-------------------

Main logic of the classification is in the file Utils.py(areaFilterClassifier).
At the moment we have 2 functions:isMachine, isHuman.

As they state we consider an object as a machine if motion track has area > 800 pixels and if with of the bounding box more than height of the bounding box

And we consider an object as a human if motion track has area > 250 pixels and if with of the bounding box less than height of the bounding box.

