# Project 5 - traffic

---

## Theorycrafting 

---

The constraint to not modify anything else than the functions mentioned in the assignment posed a special challenge by fixing the number of training epochs at a value of ten, thus making early stopping during training an invalid approach in order to avoid overfitting. 

### Prototyping

---

First prototypes model structure:
- Convolutional Layer followed by batch normalization
- Pooling Layer
- Flattening
- Fully connected hidden Layer followed by batch normalization
- Dense output Layer with softmax activation for classification

Initial training sessions delivered models with an accuracy within range of 94% to 96% already without any experimentation. 

## Experimentation

---

Tried implementing ohne dropout layer with varying rates up to 0.2, where it seemed as if the higher the rate the more inconsistent the training became without significant impact on the average accuracy of the trained model. 

Adding in more pooling layers resultet in overall decreased accuracy of the model. 

Implementation of more convolutional layers instead increased the models performance and decreased trainingtimes leading to overfitting the model.

To increase the training times i doubled the neurons in the hidden layer to 256 and added in an additional hidden layer with 128 neurons before the output layer. 

Reapeating the initial experimentation with varying dropout rates lead me to set them in realtion to the size of the dense hidden layers within a range of less then 5%. 

When the difference between the output vector of the flattening layer and the size of the dense layer taking it as input got smaller accuracy began to increase and be above 98% most of the times.

Finally i tried to add in another hidden layer of the size 512, matching the output dimension of the flattening layer seemingly without impacting the accuracy of the model, just increasing the training times by doubleing the amount of trainable parameters. 
