# 🫁 Lung Cancer Prediction System Using Machine Learning

## 📌 Overview

The Lung Cancer Prediction System is a machine learning-based web application developed to predict the possibility of lung cancer using patient health and lifestyle details. The system uses the Random Forest algorithm for accurate prediction and provides results through a user-friendly Flask web interface.

## 🎯 Objectives

* Predict lung cancer risk using machine learning techniques.
* Improve prediction accuracy using data preprocessing and feature selection.
* Provide an easy-to-use web-based interface for users.
* Generate fast and real-time prediction results.

## 💻 Technologies Used

* Python
* Flask
* HTML, CSS, Bootstrap
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Joblib

## 🤖 Machine Learning Algorithm

The project uses the Random Forest Classifier because of its high accuracy and reliability in classification problems. The model is trained using clinical and lifestyle-related features.

## ✨ Features

* User-friendly web interface
* Real-time lung cancer prediction
* Data preprocessing and normalization
* Accurate prediction using Random Forest
* Easy integration with datasets

## 📊 Dataset

The dataset contains health-related attributes such as:

* Age
* Smoking habits
* Chest pain
* Shortness of breath
* Fatigue
* Alcohol consumption
* Coughing

## ⚙️ System Architecture

1. User enters health details through the web form.
2. Input data is processed and normalized.
3. The trained Random Forest model predicts the result.
4. Prediction result is displayed to the user.

## 🚀 Installation Steps

1. Install Python.
2. Install required libraries using:

```bash
pip install -r requirements.txt
```

3. Run the Flask application:

```bash
python app.py
```

4. Open the browser and visit:

```bash
http://127.0.0.1:5000/
```

## 📁 Project Structure

```text
Lung-Cancer-Prediction/
│
├── static/
├── templates/
├── dataset/
├── model/
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

## 📈 Results

* Model Accuracy: 88.71%
* ROC-AUC Score: 0.884
* Fast prediction response
* User-friendly interface

## ✅ Advantages

* Early prediction support
* Easy to use
* Accurate results
* Web-based accessibility

## 🔮 Future Enhancements

* Integration with real-time hospital databases
* Deployment on cloud platforms
* Improved prediction using deep learning
* Mobile application support

## 📌 Conclusion

The Lung Cancer Prediction System successfully predicts lung cancer risk using machine learning techniques. The integration of Random Forest with a Flask web application provides accurate, fast, and user-friendly prediction results.

## 📚 References

1. IEEE Access Journals
2. Scikit-learn Documentation
3. Flask Documentation
4. Kaggle Lung Cancer Dataset
