# People_counter_app_with_openvino_toolkit

The people counter application at the edge utilizes Intel® hardware and software tools for converting pre-trained models and running inference on the edge. Thus making overall process convenient and faster. The people counter app was use to detect in realtime the people in a video or camera source and the result which includes total number and average duration of people in a frame was sent across to a local web server using the Paho MQTT Python package.

### Process involved

* Downloading and Conversion of a suitable model capable of detecting persons in a frame
* Preprocessing the frames to the required format before Inference was perfomed
* Sending result locally through FFserver and MQTT server

![process involved](images/flow.png)
