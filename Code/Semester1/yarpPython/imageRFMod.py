#!/usr/bin/env ipython

import readline
import yarp
import sys
import time
import numpy
import cv2

numpy.set_printoptions(threshold='nan')

class pythonRFMod(yarp.RFModule):

    def __init__(self):
        yarp.RFModule.__init__(self)
        self.respondPort = None
        self.inputPort = None
        self.outputStillPort = None
        self.outputImagePort = None
        self.inputImage = None
        self.outputImage = None
        self.outputStill = None
        self.uniCom = None
        self.imgArray = None
        self.tempImage = None

    def configure(self, rf):
        yarp.Network.init()

        self.respondPort = yarp.Port()
        self.inputPort = yarp.BufferedPortImageRgb()
        self.outputStillPort = yarp.BufferedPortImageRgb()
        self.outputImagePort = yarp.BufferedPortImageRgb()

        self.respondPort.open('/imageRPC')
        self.inputPort.open('/pythonRead')
        self.outputImagePort.open('/imageWrite')
        self.outputStillPort.open('/pictureWrite')

        self.attach(self.respondPort)

        yarp.Network.connect('/grabber', '/pythonRead')
        yarp.Network.connect('/pictureWrite', '/pictureView')
        yarp.Network.connect('/imageWrite', '/internalView')

        self.imgArray = numpy.zeros((240, 320, 3), dtype=numpy.uint8)
        self.imgBuffer = numpy.zeros((240, 320, 3), dtype=numpy.uint8)

        self.inputImage = yarp.ImageRgb()

        self.tempImage = yarp.ImageRgb()
        self.tempImage.resize(320, 240)
        self.tempImage.setExternal(self.imgArray, self.imgArray.shape[1], self.imgArray.shape[0])

        self.outputImage = yarp.ImageRgb()
        self.outputStill = yarp.ImageRgb()

        self.uniCom = 'None'

        print 'configured'
        return True

    def respond(self, command, reply):
        self.uniCom = command.toString()
        reply.clear()
        reply.addString('Recieved command')

        if self.uniCom == 'capture':

            self.outputStill = self.outputStillPort.prepare()
            self.outputStill.copy(self.tempImage)
            self.outputStillPort.write()

            self.uniCom = 'None'

        return True

    def getPeriod(self):
        return 0.03

    def updateModule(self):
        self.inputImage = self.inputPort.read()
        h = self.inputImage.height()
        w = self.inputImage.width()

        if (h != self.tempImage.height() or w != self.tempImage.width()):
            self.imgArray = numpy.zeros((h, w, 3), dtype=numpy.uint8)
            self.imgBuffer = numpy.zeros((h, w, 3), dtype=numpy.uint8)
            self.tempImage.resize(w, h)
            self.tempImage.setExternal(self.imgArray, self.imgArray.shape[1], self.imgArray.shape[0])

        self.tempImage.copy(self.inputImage)

        for i in range(0,h-1):
            for j in range(0,w-1):
                self.imgBuffer[i,j] = self.imgArray[h-1,w-1-j]
        self.imgArray = self.imgBuffer

        self.imgArray[:50, :50] = 0

        self.outputImage = self.outputImagePort.prepare()
        self.outputImage.copy(self.tempImage)
        self.outputImagePort.write()
        return True

    def interruptModule(self):
        return True

    def close(self):
        self.inputPort.close()
        self.outputPort.close()
        return True

if __name__ == '__main__':
    yarp.Network.init()
    mod = pythonRFMod()
    yrf = yarp.ResourceFinder()
    yrf.setVerbose(True)

    yrf.configure(sys.argv)
    mod.runModule(yrf)
