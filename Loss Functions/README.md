# Motivations of Loss Functions :

1. MSE Loss : Since this is a very commonly used loss function, this was used to begin with. 
2. BCEDice Loss : In order to predict whether the corresponding pixel is a lane or not, i.e., binary classifications using Cross Entropy seemed like a good second choice. We used this along with Focal Loss too, in order to give slight tweak to background channel.
3. Focal Loss : Realising how unbaised the data set was with a ratio of approximately 2% of lane pixels to 98% of background pixels, BCE was not able to give a satisfactory result. Therefore, in order to give equal importance to both kinds of pixels, Focal Loss was chosen. Although it gave very good results in semantic segmentation, it failed to do so in instance segmentation.
4. IoU Loss : Since the main goal now was to perform good instance segmenatation, performing Intersection over Union to find out each instance seperately yielded very good results for instance segmentation. Hence, IoU Loss Function was used. 

# References :
https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
