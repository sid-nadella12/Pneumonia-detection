# Pneumonia-detection

Xray- the dataset containing training, validating and test data sets.  
sample_inputs- sample x-rays  
results-  final outputs of the respective x-ray  
plot_infected_area.py- main code that is to be executed.  
pneumonia_predict_model.py - code for the model.  
vgg16.h5 - This is the trained model which we will use predict pneumonia.    


# Methods

find_infection()- calls all the functions in order and decides the affected region.  
track_left()- inspects the left lung to extract the longest sequence.  
track_right()- inspects the right lung to extract the longest sequence.  
plot_infection()- draws a circle around the Infected area.  

# Link to data

https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
