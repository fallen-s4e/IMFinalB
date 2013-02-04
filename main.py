import cv2
import numpy as np
from numpy import array
from copy import copy, deepcopy
from math import pi, sqrt
from operator import itemgetter

from Utils import *

def run():
    n = 3
    """ making a mask for coins """
    
    """ for all the coins except red(brown?) one,
        both close and open operations was not able to do what I wanted """
    defaultMaskFn = comp(fillSmallHoles(), mkThresholdFn(), 
                         mycvtConvert())
    """ 2 is a red color. for getting red coins """
    redCoinsMask = comp(mkThresholdFn(), yetAnotherCoinsSplitter)
    """ combination of both """
    maskFn = comp(#closeMO(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n,n))),
                  fillSmallHoles(0, 1000000, inclFilled = True),
                  combineMasks(cv2.bitwise_or, defaultMaskFn, redCoinsMask))

    generalContoursFn = comp(myContours, mkThresholdFn(), splitFn(2))

    redStuffFilter = comp(invert, 
                          itemsWithBigHoles,
                          #showImg, closeMO(getKernel(5)), 
                          closeMO(getKernel(5)), 
                          mkThresholdFn(180), splitFn(2))

    filteredMaskFn = combineMasks(cv2.bitwise_and, maskFn, redStuffFilter)
    
    joined = onlyIdxs([4,5,8], allImages)
    icImages = onlyIdxs([2,3], allImages) # invisble coins images
    rsImages = onlyIdxs([7,8], allImages) # red stuff images
    failed =   onlyIdxs([3,6,7,8], allImages)    
    
    testImages(filteredMaskFn, allImages[8:9], allExceptedValues[8:9])
    print(deleteme[::-1])

# run()
if __name__ == '__main__':
    print "main started"
    f = comp( # mapGen(printArr), mapGen(printTypes),
              # mapGen(mkThresholdFn()),
              # mapGen(printArr), 
              # mapGen(printTypes),
              # mkShowByGen("b"), 
              # mkShowByGen("b"), 
              # mapGen(mkThresholdFn(200)),
              # genWrite("temp"),
              mkShowByGen("b"),
              mkFrameEnumerator(),
              mapGen(mkThresholdFn(50)),
              varianceTemporalFilter(12, framed=False), 
              # averageTemporalFilter(25, framed=False), 
              mkShowByGen("a"),
              mapGen(mycvtConvert())
              )
    video = cv2.VideoCapture(-1)
    list(take(100, f(genFromVideo(video))))
    cv2.destroyAllWindows()
    video.release()
    print "main ended"

