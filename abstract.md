# FusionCal: A Hybrid Deep Learning and Metadata-Based Food Calorie Estimation System

## Abstract

FusionCal presents an innovative approach to automated food calorie estimation by combining deep learning techniques with metadata analysis. This system addresses the growing need for efficient and accurate dietary monitoring tools in health and nutrition applications. The project implements a hybrid architecture that leverages both visual features and contextual information to provide precise calorie estimates for food items.

### Key Components:

1. **Deep Learning Core:**
   - Utilizes ResNet50 architecture for robust feature extraction
   - Pre-trained on ImageNet for transfer learning advantages
   - Custom dense projection layer for specialized food feature representation

2. **Machine Learning Integration:**
   - Random Forest regressor for final calorie prediction
   - Combines visual features with portion size metadata
   - Optimized for real-world food image variations

3. **Data Management:**
   - MongoDB integration for scalable data storage
   - Real-time analytics and prediction logging
   - Efficient image and metadata handling

4. **User Interface:**
   - Streamlit-based interactive web application
   - Real-time calorie estimation
   - Intuitive visualization dashboard
   - Historical prediction tracking

### Technical Implementation:

The system processes food images through a dual-pipeline architecture:
1. Visual Analysis: Images are processed through ResNet50 to extract 512-dimensional feature vectors
2. Metadata Integration: Portion size information is combined with visual features
3. Prediction: Random Forest model generates final calorie estimates

### Results:

The system demonstrates robust performance in real-world scenarios, offering:
- Quick and accurate calorie estimations
- User-friendly interface
- Comprehensive analytics
- Scalable architecture for future enhancements

### Applications:

- Personal dietary monitoring
- Nutritional analysis
- Health and wellness applications
- Food service industry

This project contributes to the field of automated dietary assessment by providing a practical, accurate, and user-friendly solution for food calorie estimation, combining the power of deep learning with traditional machine learning approaches.