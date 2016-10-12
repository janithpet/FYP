#!/usr/bin/env ipython

import readline
import yarp
import sys
import time

class pythonRFMod(yarp.RFModule):

    def __init__(self):
        yarp.RFModule.__init__(self)
        self.respondPort = None
        self.inputPort = None
        self.outputPort = None
        self.inputBottle = None
        self.outputBottle = None
        self.uniCom = None
        self.x = None

    def configure(self, rf):
        yarp.Network.init()

        self.respondPort = yarp.Port()
        self.inputPort = yarp.BufferedPortBottle()
        self.outputPort = yarp.BufferedPortBottle()

        self.respondPort.open('/pythonRFMod')
        self.inputPort.open('/pythonRead')
        self.outputPort.open('/pythonWrite')

        self.attach(self.respondPort)

        yarp.Network.connect('/write', '/pythonRead')
        yarp.Network.connect('/pythonWrite', '/read')

        self.inputBottle = yarp.Bottle()
        self.outputBottle = yarp.Bottle()
        self.uniCom = 'None'
        print 'configured'
        return True

    def respond(self, command, reply):
        self.uniCom = command.toString()
        reply.clear()
        reply.addString('Recieved command')
        if self.uniCom == 'print':
            print self.uniCom
            self.outputBottle = self.outputPort.prepare()
            self.outputBottle.fromString(self.x)
            self.outputPort.write()
            self.uniCom = 'None'
        return True

    def getPeriod(self):
        return 0.1

    def updateModule(self):
        self.inputBottle = self.inputPort.read()
        if (self.inputBottle is not None):
            self.x = self.inputBottle.toString()
        else:
            time.sleep(0.05)

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
