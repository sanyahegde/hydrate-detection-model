# 💧 Hydrate Detection Model Using Random Forest

This project implements a **Random Forest classifier** to predict **hydrate formation** in industrial gas pipelines using real-world CSV data. The model leverages **dynamic feature engineering** and statistical analysis to flag potential hydrate events based on pipeline measurements such as volume differences and valve openness.

---

##  Key Features

### Dynamic Feature Engineering
- Calculates `Volume_Difference` as the delta between instantaneous and setpoint gas meter readings.
- Applies statistical analysis (e.g. 95th percentile thresholds) to determine hydrate-related anomalies.

###  Random Forest Classifier
- Uses an ensemble of decision trees to classify hydrate presence.
- Balances class weights to handle imbalanced datasets.
- Performs **cross-validation** for robustness.

### Visualization
- Creates scatterplots of `Volume_Difference` vs. `Valve Percent Open`, with predictions color-coded (hydrate vs. no hydrate).
- Makes model outputs interpretable for domain experts.

### Data Handling
- Reads multiple CSV files and combines them into a single dataset.
- Handles missing values and engineers key features.
- Splits data into training and test sets for evaluation.

### Model Performance
- Outputs classification reports with **accuracy, precision, recall, and F1 scores**.
- Validates performance on unseen datasets like `Fearless.csv`.

---

## How It Works

1. **Load Data**  
   - Reads and merges CSV files from the working directory.
  
2. **Preprocess Data**  
   - Cleans missing values and engineers features (`Volume_Difference`, etc.).

3. **Train Model**  
   - Fits a Random Forest classifier to predict `Hydrate_Detected`.

4. **Evaluate Model**  
   - Produces classification metrics and validates on a separate test set.

5. **Visualize Predictions**  
   - Generates scatterplots to highlight patterns in hydrate vs. non-hydrate cases.

---

## 📂 Usage

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/hydrate-detection-model.git
   cd hydrate-detection-model
