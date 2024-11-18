This project uses a Random Forest Algorithm to predict hydrate formation in industrial pipelines. 

By analyzing CSV files provided by the company, the model evaluates key variables such as:

Volume Difference: The difference between the instantaneous and setpoint gas meter volumes.
Valve Percent Open: The openness of the injection valve.


**How It Works**

Load Data: Combines multiple CSV files provided by the company.
Preprocess Data: Handles missing values.
Engineers features relevant to hydrate formation.
Train Model: Fits a Random Forest Classifier to predict hydrate presence (Hydrate_Detected).
Evaluate Model: Generates classification metrics.

Validates predictions using test data and cross-validation.
Visualize Predictions: Scatterplots highlight relationships between key features and predictions.
