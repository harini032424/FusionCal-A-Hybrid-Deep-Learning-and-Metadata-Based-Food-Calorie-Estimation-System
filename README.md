
# 🍱 FusionCal-A-Hybrid-Deep-Learning-and-Metadata-Based-Food-Calorie-Estimation-System

## Project Overview
An intelligent system that combines Deep Learning, Machine Learning, and Big Data technologies to estimate calorie content in food images. This project is designed for Big Data Analytics (BDA) course submission.

## 🌟 Key Features

- **Deep Learning Feature Extraction**: Uses ResNet50 to extract visual features from food images
- **Machine Learning Prediction**: Random Forest model trained on 80/20 split for accurate calorie estimation
- **MongoDB Integration**: Stores unstructured image data and prediction history
- **Interactive UI**: Streamlit-based interface with real-time predictions
- **Analytics Dashboard**: Visual insights using Plotly charts
- **Persistent History**: Tracks all predictions with timestamps
- **Responsive Design**: Blue-themed UI with modern components

## 🛠️ Technical Architecture

1. **Data Storage (MongoDB)**
   - Unstructured image data storage
   - Feature vectors from ResNet50
   - Prediction history with timestamps
   - Analytics aggregations

2. **Model Pipeline**
   - ResNet50 for feature extraction
   - Random Forest for calorie prediction
   - 80/20 train/test split
   - Performance metrics tracking

3. **Web Interface**
   - Streamlit for interactive UI
   - Plotly for data visualization
   - Real-time predictions
   - Historical analytics

## 📊 Data Management

- **Training Data**: Split 80/20 for robust model evaluation
- **Image Storage**: Unstructured data in MongoDB
- **Feature Storage**: High-dimensional vectors from ResNet50
- **History Tracking**: All predictions with metadata

## 🚀 Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure MongoDB:
   - Install MongoDB locally or use MongoDB Atlas
   - Set connection string in environment variables

3. Run the application:
   ```bash
   streamlit run web_app/app.py
   ```

## 📌 Project Structure

```
food-calorie-estimation/
├── db/
│   └── db_connection.py     # MongoDB connection handling
├── models/
│   ├── cnn_regressor.py    # CNN model architecture
│   ├── feature_extraction.py # Feature extraction logic
│   └── train_rf_model.py   # Random Forest training (80/20 split)
├── web_app/
│   └── app.py             # Streamlit UI and main application
└── utils/
    └── preprocessing.py   # Image preprocessing utilities
```

## 🎯 Usage

1. Navigate to "Predict Calories" page
2. Upload a food image
3. Adjust portion size if needed
4. Get instant calorie prediction
5. View prediction history and analytics

## 📈 Analytics

- Real-time prediction tracking
- Distribution of calorie ranges
- Top predicted foods
- Historical trends
- Category-wise analysis

## 👨‍💻 Developer

- **Name**: [Your Name]
- **Roll No**: [Your Roll No]
- **Department**: [Your Department]

## 📝 Note

This project demonstrates the practical application of:
- Unstructured data handling with MongoDB
- Deep Learning feature extraction
- Machine Learning prediction models
- Interactive data visualization
- Big Data analytics principles
=======



