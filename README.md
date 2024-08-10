# Vehicle Damage Insurance Classification using ConvNeXt
Final project for the Technion's ECE Deep Learning course (046211)
<br>
Yarden Shavit
<br>
Dor Yogev
<br>
Ravid Goldenberg

  
# Introduction

This project aims to develop a deep learning model capable of identifying the type of damage from images of damaged vehicles. We utilize the ConvNeXt architecture and implement various techniques to achieve high accuracy in predictions.
![image](assets/dataset_samples.jpeg)

# ConvNeXt

ConvNeXt is a state-of-the-art convolutional neural network architecture introduced by Facebook AI Research in 2022. It was designed to bridge the gap between convolutional neural networks (CNNs) and vision transformers (ViTs), combining the best aspects of both approaches.
Key features of ConvNeXt include:

Modernized CNN design: Incorporates advancements from vision transformers while maintaining the efficiency of CNNs. <br>
Improved performance: Achieves comparable or better accuracy than vision transformers on various computer vision tasks.<br>
Scalability: Offers different model sizes (Tiny, Small, Base, Large) to suit various computational requirements.<br>
Transfer learning capabilities: Pre-trained on large datasets, allowing for effective fine-tuning on specific tasks like our vehicle damage classification.

We chose ConvNext for this project due to its strong performance in image classification tasks and its ability to capture both local and global features effectively.
![image](assets/convnext_structure.jpeg)


# Dataset:
We use the Vehicle Damage Insurance Verification dataset from Kaggle. This dataset provides a comprehensive collection of vehicle damage images for training and testing our model.

The dataset comprises 12,000 images, categorized into six distinct classes: 
Dent,
Scratch,
Crack,
Glass Shatter,
Lamp Broken,
Tire Flat.


# Results

### Hyperparameter Optimization

We used Optuna, an automatic hyperparameter optimization framework, to find the best hyperparameters for our model. Optuna employs efficient search algorithms to explore the hyperparameter space and find the optimal configuration.

Our Optuna trials yielded the following optimal hyperparameters:

![image](assets/optuna_result.jpeg)

### Final Model Performance

Using the optimal hyperparameters found by Optuna, our final model achieved the following results:

![image](assets/accuracy_graphs.jpeg)

**train accuracy: 99.771%**

**validation accuracy: 98.395%**

**test accuracy: 97.328%** 

**confusion matrix:**

![image](assets/confusion_matrix.jpeg)



# Usage
1. Dataset Preparation:
   - Use `dataset.py` to preprocess and organize your data.

2. Data Normalization:
   - Run `calculate_normalize_mean_std.py` to compute normalization parameters.

3. Model Training:
   - Execute `train.py` to train the ConvNext model on your dataset.

4. Evaluation and Results:
   - Use `results.py` to evaluate the model's performance and generate result metrics.

5. Hyperparameter Tuning:
   - If needed, run `tune_hyper_parameters.py` to optimize model hyperparameters.

# Future Work

- Experiment with additional architectures
- Implement more advanced data augmentation techniques
- Optimize model for deployment in real-world scenarios
  
# References:

dataset:

[https://www.kaggle.com/datasets/mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out?resource=download](https://www.kaggle.com/datasets/sudhanshu2198/ripik-hackfest/data)

ConvNeXt:

https://github.com/facebookresearch/ConvNeXt

