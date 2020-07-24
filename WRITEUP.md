# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The process behind converting custom layers involves the seting of personal custom function of a model layer where you are able to create your own personalize layers where the intel model optimizer will be able to recognize such personalized layer or functions

Custom layers are a necessary and important to have feature of the OpenVINO™ Toolkit, although you shouldn’t have to use it very often, if at all, due to all of the supported layers. However, it’s useful to know a little about its existence and how to use it if the need arises.
Some of the potential reasons for handling custom layers are
In industry problems it becomes very important to be able to convert custom layers as your teams might be developing something new or researching on something and your application to work smoothly you would need to know how you could have support for custom layers.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were Converted the model to intermediate representation using this model lacked accuracy as it didn't detect people correctly in the video. Made some alterations to the threshold for increasing its accuracy but the results were not fruitful.

The difference between model accuracy pre- and post-conversion was that the model accuarcy seems to work best during the post conversion than the pre conversion with a better and enhance inference time

The size of the model pre- and post-conversion was...
ssd_inception_v2_coco (plain TF)	size of model in megabytes	538
ssd_inception_v2_coco (OpenVINO)	size of model in megabytes	329
faster_rcnn_inception_v2_coco (plain TF)size of model in megabytes	562
faster_rcnn_inception_v2_coco (OpenVINO)size of model in megabytes	281

The inference time of the model pre- and post-conversion was...
ssd_inception_v2_coco (plain TF) inference time in microseconds	 222	
ssd_inception_v2_coco (OpenVINO)	inference time in microseconds		 155	
faster_rcnn_inception_v2_coco (plain TF)	inference time in microseconds	 1281	
faster_rcnn_inception_v2_coco (OpenVINO)	inference time in microseconds	  889	

## Assess Model Use Cases
the counter can be used to determine the number of person who enters a particular room from this use case a client will get advantage or value over it by permitted him or her to have control over a particular room or to restrict the number of people which may enter a particular room there by monitory the room efficiently
the model can be used in a queue management system we take the case of a super market to take of the people at each payement stand for efficient or optimized product payement within the market to avoid congestion 

Some of the potential use cases of the people counter app are...
This application could keep a check on the number of people in a particular area and could be helpful where there is restriction on the number of people present in a particular area. Further, with some updations
Each of these use cases would be useful because...

Each of these use cases will be useful because they will enhance, improve, optimize the management and efficiency  at  the workplace where actually the Artificial Intelligence model has been applied

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

the better the nodel accuracy, the better and more accurate the prediction and vice versa
The more or higher the lightening is the more accurate the predictions are since the image qualities will be intense
The higher the camera focal / image size the better the predictions at deployment since enough features will be provided to the model for predictions to be done at a higgh precision that when the camera foacl / image size is smaller

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [SSD MobileNet V2]
  - [http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz]
  - I converted the model to an Intermediate Representation with the following arguments python mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json
  - The model was insufficient for the app because the predictions were not well done where by the drawing boxes were blinking a lot giving false positive to the total number of people counted
  - I tried to improve the model for the app by changing the model threshold and also changed the model used
  
- Model 2: [SSD Inception V2]
  - [http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz]
  - I converted the model to an Intermediate Representation with the following arguments python mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json
  - The model was insufficient for the app because because the predictions were not well done even with its little more performance over SSD_MobileNet V2 COCO over the sma ethrshold it was still giving the same false postive at the level of the total number of people counted
  - I tried to improve the model for the app by chnaging the model I was using by another

- Model 3: [faster_rcnn_resnet101_coco_2018_01_28]
  - [http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz]
  - I converted the model to an Intermediate Representation with the following arguments python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
  - The model was insufficient for the app because its inference time was quite high giving rendering the CPU very since a lot of CPU resources were used
  - I tried to improve the model for the app by changing the app threshold
