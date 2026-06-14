\# House Price Prediction using Machine Learning



\## Intern Information

\- \*\*Intern ID:\*\* \[CITS779]

\- \*\*Full Name:\*\* \[VANA HARITHA]

\- \*\*No. of Weeks:\*\* 2 Weeks

\- \*\*Project Name:\*\* House Price Prediction



\## Project Scope

This project builds a machine learning model to predict house prices using the California Housing dataset (20,640 records, 8 features). Multiple regression algorithms are compared, and the best model is saved for future predictions. The project includes data exploration, model training, evaluation, and a prediction script for new inputs.



\## Technologies Used

\- Python 3.8+

\- pandas, numpy

\- matplotlib, seaborn

\- scikit-learn (Linear Regression, Ridge, Lasso, Random Forest)

\- joblib (model persistence)



\## Project Structure

house\_price\_prediction/

├── src/

│ ├── download\_dataset.py

│ ├── train\_model.py

│ └── predict\_price.py

├── data/

│ └── housing.csv

├── models/

│ ├── best\_model.pkl

│ └── scaler.pkl

├── outputs/

│ ├── correlation\_heatmap.png

│ ├── target\_distribution.png

│ ├── actual\_vs\_predicted.png

│ └── model\_comparison.csv

├── screenshots/

│ ├── correlation\_heatmap.png

│ ├── target\_distribution.png

│ ├── actual\_vs\_predicted.png

│ └── price\_prediction\_output.png

├── README.md

└── requirements.txt



\## How to Run

1\. Install dependencies: `pip install -r requirements.txt`

2\. Download dataset: `python src/download\_dataset.py`

3\. Train models: `python src/train\_model.py`

4\. Predict price: `python src/predict\_price.py`



\## Results

Best model: Random Forest Regressor



| Model | R² Score | RMSE |

|-------|----------|------|

| Linear Regression | 0.575 | \~0.85 |

| Ridge Regression | 0.575 | \~0.85 |

| Lasso Regression | 0.556 | \~0.87 |

| Random Forest | 0.812 | \~0.52 |


## Output Images

| Image | Description |
|-------|-------------|
| ![Correlation Heatmap](https://raw.githubusercontent.com/Haritha-Vana/house-price-prediction/main/screenshots/correlation_heatmap.png) | **Correlation Heatmap** – Shows relationships between features. MedInc (median income) has the highest positive correlation with house price (MedHouseVal). |
| ![Target Distribution](https://raw.githubusercontent.com/Haritha-Vana/house-price-prediction/main/screenshots/target_distribution.png) | **Target Distribution** – Histogram of house prices. Most values are between 1 and 3 (i.e., $100,000 to $300,000). |
| ![Actual vs Predicted](https://raw.githubusercontent.com/Haritha-Vana/house-price-prediction/main/screenshots/actual_vs_predicted.png) | **Actual vs Predicted (Random Forest)** – Scatter plot showing predictions vs true values. R² score = 0.812, indicating good model performance. |
| ![Prediction Output](https://raw.githubusercontent.com/Haritha-Vana/house-price-prediction/main/screenshots/price_prediction_output.png) | **Prediction Example** – Terminal output showing a sample house predicted price of $234,699. |



\### Prediction Example

!\[Prediction Output](screenshots/price\_prediction\_output.png)



\## GitHub Repository

\[https://github.com/Haritha-Vana/house-price-prediction.git]



