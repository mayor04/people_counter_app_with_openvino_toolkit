"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
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
import time
import socket
import json
import cv2
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    network = Network()
    
    # Set Probability threshold for detections
    threshold = args.threshold

    ### TODO: Load the model through `infer_network` ###
    network.load_model(args.model,args.device,CPU_EXTENSION)
    shape = network.get_rcnn_shape()

    ### TODO: Handle the input stream ###
    if args.input == 'CAM':
        input_im = 0

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_im = args.input

    # Checks for video file
    else:
        input_im = args.input
        assert os.path.isfile(args.input), "file doesn't exist"
        
        
    cap = cv2.VideoCapture(input_im)
    cap.open(input_im)
    
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width,height))
    
    last_count = 0
    total = 0
    
    detect_range = 0
    average = 0
    duration_count = 0
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        
        ### TODO: Read from the video capture ###
        if not flag:
            break

        ### TODO: Pre-process the image as needed ###
        image = preprocess(frame,shape)
        
        start = time.time()
        ### TODO: Start asynchronous inference for specified request ###
        network.exec_rcnn(image)
        duration = None;
        
        ### TODO: Wait for the result ###
        if(network.wait()==0):
            
            end = time.time()
            ### TODO: Get the results of the inference request ###
            output = network.get_output()
            
            ### TODO: Extract any desired stats from the results ###
            frame,persons = draw_boxes(frame,output,threshold)

            ### TODO: Calculate and send relevant information on ###
            diff = persons-last_count
            duration_count+=1
                  
            if(diff == 0):
                detect_range=0
            else:
                detect_range+=1
            
            if(detect_range > 4):
                last_count = persons
                
                if(diff > 0):
                    total += diff
                    duration_count=0
                else:
                    #person left based on 30 frames per seconds
                    duration = duration_count/10
                    all_dur = average*(total-1)
                    average = (all_dur+duration)/total

            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            client.publish("person", json.dumps({"count": persons ,"total": total}))
                   
            ### Topic "person/duration": key of "duration" ### 
            if duration is not None:
                client.publish("person/duration", json.dumps({"duration": int(average)}),qos=0, retain=False)

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        
        # Break if escape key pressed
        key_pressed = cv2.waitKey(1)
        if key_pressed == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    ### Disconnect from MQTT
    client.disconnect()    
    ### TODO: Write an output image if `single_image_mode` ###    

def draw_boxes(frame, result,thres):
    h,w,c = frame.shape
    i = 0
    
#     print(result)
    for box in result[0][0]:
        conf = box[2]
        if conf >= thres and box[1] == 1:
            xmin = int(box[3] * w)
            ymin = int(box[4] * h)
            xmax = int(box[5] * w)
            ymax = int(box[6] * h)
            i+=1
            
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (130, 100, 255), 4)
            
    return frame,i

def inferOnPicture(args):
    network = Network()
    network.load_model(args.model,args.device,CPU_EXTENSION)
    
#     shape = network.get_rcnn_shape()
    shape = network.get_input_shape()
    original = cv2.imread(args.input)
    image1 = np.copy(original)
    
    image = preprocess(image1,shape)
    
    network.exec_rcnn(image)
#     network.exec_net(image)
    
    if(network.wait()==0):
        output = network.get_output()
        print(output.shape)
        
        new_image,i = draw_boxes(image1,output)
        print(new_image.shape)
        
        cv2.imwrite('images/new.jpg',new_image)

    
def preprocess(imag,shape):
    image = np.copy(imag)
    
    image = cv2.resize(image, (shape[3], shape[2]))
    image = image.transpose((2,0,1))
    image = image.reshape(1, *image.shape)
    
    return image
    
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)
#     inferOnPicture(args);



if __name__ == '__main__':
    main()
