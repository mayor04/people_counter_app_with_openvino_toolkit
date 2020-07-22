# People_counter_app_with_openvino_toolkit

The people counter application at the edge utilizes Intel® hardware and software tools for converting pre-trained models and running inference on the edge. Thus making overall process convenient and faster. The people counter app was use to detect in realtime the people in a video or camera source and the result which includes total number and average duration of people in a frame was sent across to a local web server using the Paho MQTT Python package.

### Process involved

* Downloading and Conversion of a suitable model capable of detecting persons in a frame
* Preprocessing the frames to the required format before Inference was perfomed
* Sending result locally through FFserver and MQTT server

![process involved](images/flow.png)

### Explaining Custom Layers in OpenVINO™

Custom layers are layers that are not in the list of known layers. The Model Optimizer first extracts information from the input model which includes the topology of the model layers along with parameters for each layer. The model is then optimized from the various known characteristics of the layers. Finally, it produces an output to the model IR files needed by the Inference Engine to run the model.

Both Ssd_inception_v2_coco and faster_rcnn_inception_v2_coco was tested on this project but faster_rcnn_inception_v2_coco performed better and was more accurate on the particular video given when compared to Ssd_inception_v2_coco
