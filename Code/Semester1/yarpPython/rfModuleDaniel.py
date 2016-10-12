#!/usr/bin/env ipython
#his line here at the very top ensures that you can run this by putting
# ./rfModuleBareExample.py in the terminal by doing chmod +x rfModuleBareExample.py once
# rather than python rfModuleBareExample.py everytime you want to run it
import readline
import yarp
import sys
import time


class NameOfModuleClass(yarp.RFModule):

    def __init__(self):
        yarp.RFModule.__init__(self)
        self.respondPort = None

    def configure(self, rf):
        yarp.Network.init()

        # self.respondPort is an rpc port (read up on this type of port)
        self.respondPort = yarp.Port()
        self.respondPort.open('/respond:i')
        # attaching this port to NameOfModuleClass means that whenever 
        # an input is received on this port, the NameOfModuleClass.respond function is triggered
        self.attach(self.respondPort)


        # initialise other input / output ports, 
        # open them and connect them in this function
        return True

    def close(self):
        # this function is run upon exiting the rfModule
        return True

    def respond(self, command, reply):
        # this function is run every time a command is received on 
        print command.toString()
        reply.clear()
        reply.addString('ack')
        return True

    def interruptModule(self):
        # this function is run when a termination code is sent to the terminal like ctrl+c
        return True

    def getPeriod(self):
        # returns the period at which updateModule is run
        # no need to touch this function
        return 0.1

    def updateModule(self):
        # this function is run on a while loop
        time.sleep(0.05)
        return True

if __name__ == '__main__':

    yarp.Network.init() 
    mod = NameOfModuleClass()
    yrf = yarp.ResourceFinder()
    yrf.setVerbose(True)
    #yrf.setDefaultContext("NameOfModuleClass") # uncomment when config files are required
    #yrf.setDefaultConfigFile("default.ini") # uncomment when config files are required
    yrf.configure(sys.argv)

    mod.runModule(yrf)
