import cv2
import numpy as np
from numpy import array
from copy import copy, deepcopy
from math import pi, sqrt
from operator import itemgetter

from Utils import *

""" -------------------- combining gens """

# I couldn't find a way to combine two stream, because function on stream works on
# entire stream, so I decided to use haskell-like arrows


""" -------------------- entry point """

# run()
if __name__ == '__main__':
    print "main started"

    def g((state, image)):
        return (state, drawBoundBoxes(areaFilterClassifier)(state, image))

    mask = comp( # mapGen(printArr), mapGen(printTypes),
                 # mapGen(mkThresholdFn()),
                 # mapGen(printArr), 
                 # mapGen(printTypes),
                 mapGen(mkThresholdFn(200)),
                 #genWrite("temp"),
                 # mkShowByGen("b"),
                 # mkFrameEnumerator(),
                 mkShowByGen("a"),
                 mapGenWithState(g),
                 mkShowByGen("b"),
                 # mapGenWithState(lambda (a,b) : (printTypes( a0 ), printTypes( b ))),
                 #mapGen(closeMO(getKernel(3), 3)),
                 mapGen(dilate(getKernel(3))),

                 mapGen(mkThresholdFn(20)),
                 varianceTemporalFilter(50, framed=False), 
                 mapGen(mycvtConvert())
                 )

    f = mask
    video = cv2.VideoCapture(-1)
    video = cv2.VideoCapture("camera1.avi")
    for x in range(0,25*40):
        video.read()
    list(take(4000, f(genFromVideo(video))))
    cv2.destroyAllWindows()
    video.release()
    print "main ended"

