# 🏡 House Price Prediction — Machine Learning Project

## 📘 Overview
This project predicts the **sale prices of houses** based on their features using **Machine Learning regression models**.
It is based on the famous [Kaggle “House Prices: Advanced Regression Techniques”](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) dataset.

You’ll learn the **complete ML workflow** — from loading and cleaning data to training models, evaluating performance, and generating predictions for submission.

---

## 🎯 Project Objectives
- Perform **data cleaning and preprocessing** on real-world housing data.
- Explore relationships between features and sale price.
- Train and compare multiple **regression algorithms**.
- Evaluate model performance using metrics (MAE, RMSE, R²).
- Generate predictions and save them for **Kaggle submission**.

---

## 🧠 Key Learnings
- Hands-on experience with **pandas**, **NumPy**, **Matplotlib**, **Seaborn**, and **scikit-learn**.
- Understanding the difference between **Linear Regression**, **Decision Tree**, and **Random Forest** models.
- Practice in **data preprocessing**, **feature selection**, and **model evaluation**.
- How to **save models** and **make predictions on unseen data**.

---

## ⚙️ Tech Stack
| Category | Tools/Libraries |
|-----------|----------------|
| Language | Python |
| Data Handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn |
| Model Saving | joblib |
| Environment | Jupyter Notebook / Google Colab |

---

## 📂 Project Structure
```
house-price-prediction/
│
├── train.csv                 # Training data from Kaggle
├── test.csv                  # Test data from Kaggle
├── house_price_prediction.ipynb   # Main Jupyter Notebook
├── submission.csv            # Output predictions for Kaggle
├── house_price_model.pkl     # Saved trained model
└── README.md                 # Project documentation (this file)
```

---

## 🧰 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/house-price-prediction.git
cd house-price-prediction
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### 3. Download Dataset
Download from Kaggle:
👉 [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

Then place the files `train.csv` and `test.csv` inside your project folder.

### 4. Run the Notebook
```bash
jupyter notebook house_price_prediction.ipynb
```

Run all cells to see outputs and results.

---

## 🔍 Workflow Summary
1. Load the dataset
2. Explore the data (EDA)
3. Select useful numeric features
4. Visualize correlations
5. Split data into training and validation sets
6. Train models (Linear, Decision Tree, Random Forest)
7. Evaluate performance
8. Predict on test data
9. Save model and generate submission.csv

---

## 📈 Example Output
| Model | MAE | RMSE | R² |
|--------|------|------|----|
| Linear Regression | 24567.1 | 34567.8 | 0.73 |
| Decision Tree | 18045.3 | 29012.5 | 0.81 |
| Random Forest | **15234.7** | **25543.9** | **0.86** |

*(Values may vary depending on random seed and parameters.)*

---

## 💾 Output Files
- `submission.csv` → file ready for Kaggle submission.
- `house_price_model.pkl` → trained Random Forest model (can be reloaded for prediction).

Example code:
```python
import joblib
model = joblib.load("house_price_model.pkl")
price = model.predict([[7, 1800, 2, 480, 900, 850, 2, 6, 2005]])
print("Predicted Price:", price)
```

---

## 🚀 Next Steps
- Add more features and categorical encoding.
- Try advanced models like **XGBoost** or **LightGBM**.
- Perform **hyperparameter tuning** with `GridSearchCV`.
- Build a **Flask/Streamlit web app** for interactive prediction.

---

## 📚 References
- [Kaggle Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Hands-On Machine Learning by Aurélien Géron](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

---

## 🧑‍💻 Author
**Your Name**
Fresher in Data Analytics | Aspiring Machine Learning Engineer  
📧 [your.email@example.com]  
🌐 [LinkedIn / Portfolio link]
