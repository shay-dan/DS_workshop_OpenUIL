We tried to implemented a model capable of analyzing images of basic individual food items to predict their nutrition values.

The pipeline is as follows:

preprocess -> EDA -> augmentation ->
classification (optional - just for sanity check) ->
object detection -> volume estimation using bounding boxes ->
bounding boxes as inputs to SAM -> volume estimation using masks ->
nutritional value output

the notebooks should be ran in the following order:

Preprocess

EDA

Augmentation

Prediction - Classification - optional

Prediction - ObjectDetectionDetectron2

Prediction - ObjectDetectionYOLOv8

Model Comparison - optional

Weight estimation with bounding box

Weight estimation with SAM masks

We performed EDA looking for possible problems with the data, and different features to be used for classification, and tried to fix some of our problems using image augmentations.

Afterwards, we tested different models performance on different features, starting with simple models and increasing complexity.

We then went on to use object detection models to give us predicted bounding boxes, which we tried to use for volume estimation.

In general, the more complex model we used the better results we got, with the object detectors class and bounding box predictions being nearly perfect. results comparison can be seen on chapter 7, and in each model's summary notes.

However, the volume/weight estimations weren't as good, due to various reasons, such as boxes not giving accurate borderlines, objects with hard geometrical form, etc.

In trying to improve our results, we used the predicted bounding boxes as inputs for SAM, and used its output masks for volume estimation.

We can see that these predictions are far from perfect as well, but better than bounding boxes only.

While we are happy with our results for basic foods, our model is limited and cannot be applied to cooked dishes at the moment.

We belive that the model shows a nice idea that can be expanded further, with future improvements that could involve expanding the model's capabilities to include a wider variety of food types and refining its accuracy for a more comprehensive range of culinary scenarios.

Hardware:

The entire notebook was run on Google Colab, using Nvidia's T4 GPU (colab's default). this resulted in needing to save our work on disk, as colab offers only limited free resources. it also made training impossible without resizing our image data.

Sources of inspiration for the code

https://github.com/Liang-yc/ECUSTFD-resized-

https://github.com/pylabel-project/samples/blob/main/README.md

https://learnopencv.com/

https://github.com/bnsreenu/python_for_microscopists

https://github.com/henrhoi/image-classification/blob/master/feature_extraction_and_exploratory_data_analysis.ipynb

https://towardsdatascience.com/how-to-cluster-images-based-on-visual-similarity-cd6e7209fe34

https://kapernikov.com/tutorial-image-classification-with-scikit-learn/

https://neptune.ai/blog/data-exploration-for-image-segmentation-and-object-detection"

https://github.com/Paperspace/DataAugmentationForObjectDetection.git

https://github.com/facebookresearch/detectron2

https://www.ultralytics.com/yolo

https://github.com/ultralytics/ultralytics

https://github.com/facebookresearch/segment-anything

https://api-ninjas.com/api/nutrition

https://segment-anything.com/

https://www.cuemath.com/volume-formulas/

https://chat.openai.com/
