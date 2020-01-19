# Simpsons Characters recognition using transfer learning (VGG19)

The Simpsons Characters dataset is obtained from https://www.kaggle.com/alexattia/the-simpsons-characters-dataset.
I only used 23 characters from the dataset. There are 13495 images for the 23 characters.

I use transfer learning in the  baseline model for this experiment. I make use of the model (VGG19) that is already trained on ImageNet and use the convolutional layers as feature extractor and trained a classifier specifically for the Simpsons Characters classification task. The training was done using Nvidia GTX 1660 TI GPU. The accuracy of this model is around 85%.

## The Simpsons Characters and Classification Labels
![Classification Labels](/images/classification_labels.JPG)

## The Classification Model
![Classification Model](/images/classification_model.JPG)

## The Classification Report
![Classification Report](/images/classification_report.JPG)

## The Training Progress
![Training Progress](/images/training_progress.JPG)

## Prediction using test image
![Prediction](/images/prediction.JPG)
