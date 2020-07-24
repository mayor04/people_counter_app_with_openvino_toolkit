#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
import numpy as np
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_name = None
        self.output_name = None
        self.exec_network = None
        self.infer_request = None


    def load_model(self, model, device="CPU", cpu_extension=None):
        ### TODO: Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        self.plugin = IECore()
        
         ### TODO: Add any necessary extensions ###
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)
        
        self.network = IENetwork(model=model_xml, weights=model_bin) 
        self.exec_network = self.plugin.load_network(self.network, device)

        self.input_name = next(iter(self.network.inputs))
        self.output_name = next(iter(self.network.outputs))
        
        ### TODO: Check for supported layers ###
        supported_layers = self.plugin.query_network(network=self.network, device_name="CPU")

        unsupported_layers = []
        for l in self.network.layers.keys(): 
            if l not in supported_layers:
                unsupported_layers.append(l)
            
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check your extensions")
            exit(1)
            
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        return

#         oute = iter(self.network.outputs);
#         print(next(oute))
#         print(next(oute))
#         print(self.input_name)
#         print(self.network.inputs['image_info'].shape)

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.network.inputs[self.input_name].shape
    
    def get_rcnn_shape(self):
        return self.network.inputs['image_tensor'].shape
    
    def exec_net(self,image):
        ### TODO: Start an asynchronous request ###
        self.infer_request = self.exec_network.start_async(request_id=0, 
            inputs={self.input_name: image})
        
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return
    
    def exec_rcnn(self,image):
        shape = image.shape
        info = np.array([shape[2],shape[3],1])
        
        self.infer_request = self.exec_network.start_async(request_id=0, 
            inputs={'image_info':info,'image_tensor': image})
        
        return
    
    def sync(self,image):
        return self.exec_network.infer({input_blob: image})

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        status = self.infer_request.wait()
            
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return status

    def get_output(self):
        return self.infer_request.outputs[self.output_name]

# python main.py -m rcnn_v2/frozen_inference_graph.xml -i resources/Pedestrian_Detect_2_1_1.mp4 -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm 