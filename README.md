# yoga_analyser

This program uses TensorFlow MoveNet to analyse yoga videos.

**References:**  
Yoga-82 dataset: https://sites.google.com/view/yoga-82/home  
MoveNet: https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html  
Examples: https://www.pexels.com/  
MLFlow: https://mlflow.org/  

**Usage:**  
First install the requirements:  
```pip install -r requirements.txt```  

Models and their artifacts are available on localhost:5000 after the following command:  
```mlflow server --backend-store-uri sqlite:///yoga_mlflow.db --default-artifact-root ./yoga_artifacts --host 0.0.0.0```  

```make_data.py``` makes csv files containing the data keypoints and the pose classes (the dataset should be downloaded for this)  
```train_models.py``` trains three different types of models for classification (Random Forest, K-Nearest Neighbor, Neural Network)  
```analyse_video.py``` can analyse a video (recognizes poses, measures time and checks correctness)  
