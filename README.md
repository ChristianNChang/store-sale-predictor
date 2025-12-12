This project trains a Multi Layer Perception Neural Network to predict store sales using date features + one-hot encoded categorical features, trains on log1p(sales), and provides a script to plot training/validation RMSE plus the final MAPE. 

data_prep.py — loads train.csv and test.csv, creates date features, one-hot encodes inputs, and builds train/validation DataLoaders. 

model_architecture.py — defines the sales_model MLP. 

train.py — trains the model saving best weights, training history, and MAPE

predict.py — loads the model and generates predictions for test.csv placing them in test_predictions.csv. 

visualization.py — loads and plots RMSE curves and MAPE

dependencies:
train.csv and test.csv placed in project folder
numpy, pandas, torch (PyTorch) and matplotlib installed

running:
after getting all the dependencies run data_prep.py, train.py, and then visualization.py to see model training information
run predict.py to get sales predictions on test.csv