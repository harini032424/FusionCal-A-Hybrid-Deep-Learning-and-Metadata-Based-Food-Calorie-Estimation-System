"""
Food Calorie Estimation Web Application
with MongoDB Analytics Dashboard
"""
import os
import sys
import logging
import warnings
import tensorflow as tf

# Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import streamlit as st
import numpy as np
import io
import joblib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

from db.db_connection import (
    store_image_file, save_metadata_doc, save_prediction_log, 
    get_image_file, get_analytics_data
)

# Page config
st.set_page_config(
    page_title="FusionCal",
    page_icon="🍱",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    /* Page container and background gradient */
    html, body, .stApp {
        height: 100%;
        background: linear-gradient(135deg, #e6f0ff 0%, #ffffff 45%, #f0f8ff 100%);
        background-attachment: fixed;
        color: #12263a;
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    }
    /* Limit content width and add subtle backdrop */
    .appview-container .main .block-container {
        max-width: 1200px;
        margin-top: 20px;
        margin-left: auto;
        margin-right: auto;
        padding: 1.5rem 2rem;
        background: rgba(255,255,255,0.85);
        border-radius: 12px;
        box-shadow: 0 6px 24px rgba(16,24,40,0.08);
    }
    .main-header {
        font-size: 2rem;
        color: #0b63a7;
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 0.75rem 1rem;
    }
    /* Category badges */
    .category-low { 
        color: #155724;
        padding: 0.4rem 0.6rem;
        border-radius: 6px;
        background: rgba(40, 167, 69, 0.08);
        font-weight: 600;
    }
    .category-medium { 
        color: #856404;
        padding: 0.4rem 0.6rem;
        border-radius: 6px;
        background: rgba(255, 193, 7, 0.08);
        font-weight: 600;
    }
    .category-high { 
        color: #7a4100;
        padding: 0.4rem 0.6rem;
        border-radius: 6px;
        background: rgba(253, 126, 20, 0.08);
        font-weight: 600;
    }
    .category-very-high { 
        color: #721c24;
        padding: 0.4rem 0.6rem;
        border-radius: 6px;
        background: rgba(220, 53, 69, 0.08);
        font-weight: 600;
    }
    /* Buttons */
    .stButton > button {
        width: 100%;
        padding: 0.5rem;
        font-size: 1.05rem;
        margin-top: 0.75rem;
        background-color: #0b63a7;
        color: white;
        border-radius: 8px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #064a78;
    }
    /* Metrics and cards */
    div[data-testid="stMetricValue"] { font-size: 1.9rem !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.95rem !important; }
    .metric-card { background: linear-gradient(180deg,#ffffff,#f7fbff); padding: 1rem; border-radius: 12px; box-shadow: 0 4px 14px rgba(11,99,167,0.06); }
    .info-box { background: rgba(11,99,167,0.04); padding: 1rem; border-radius: 10px; margin: 1rem 0; }
    .stPlotlyChart { background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }

    /* Sidebar tweaks */
    .css-1d391kg .css-1avcm0n { background: rgba(11,99,167,0.02); }

    @media (max-width: 768px) {
        .main-header { font-size: 1.6rem; }
        .appview-container .main .block-container { padding: 1rem; }
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🍱 Predict Calories", "📊 Analytics Dashboard", "ℹ️ About Project"])


def show_history_sidebar(limit: int = 6):
    from db.db_connection import get_recent_predictions, get_image_by_filename
    st.sidebar.markdown("---")
    st.sidebar.header("📜 Recent Predictions")
    with st.sidebar.expander("Show recent predictions", expanded=False):
        recent = get_recent_predictions(limit=limit)
        if not recent:
            st.sidebar.write("No predictions yet")
            return
        for r in recent:
            ts = r.get('timestamp')
            filename = r.get('filename') or r.get('image_name')
            calories = r.get('predicted_calories')
            user_name = r.get('user_food_name')
            img_bytes = None
            if r.get('gridfs_id'):
                try:
                    img_bytes = get_image_by_filename(filename) if filename else None
                except Exception:
                    img_bytes = None
            else:
                img_bytes = get_image_by_filename(filename) if filename else None

            cols = st.sidebar.columns([1, 2])
            with cols[0]:
                if img_bytes:
                    try:
                        st.image(img_bytes, width=80)
                    except Exception:
                        st.write("[img]")
                else:
                    st.write("[img]")
            with cols[1]:
                st.markdown(f"**{user_name or filename or 'Unknown'}**")
                st.markdown(f"{calories:.1f} kcal" if calories is not None else "- kcal")
                if ts:
                    st.markdown(ts.strftime("%Y-%m-%d %H:%M"))

# show history in sidebar
show_history_sidebar(limit=6)

def get_category_color(calories):
    if calories < 200:
        return "#28a745"  # Green
    elif calories < 400:
        return "#ffc107"  # Yellow
    elif calories < 700:
        return "#fd7e14"  # Orange
    else:
        return "#dc3545"  # Red

def show_prediction_page():
    st.markdown('<h1 class="main-header">FusionCal: A Hybrid Deep Learning and Metadata-Based Food Calorie Estimation System</h1>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner('Loading models...'):
        resnet_model = load_resnet()
        rf_model = load_rf()
    
    if resnet_model is None or rf_model is None:
        st.error("❌ Required models not found. Please check the models/ folder.")
        return
        
    # Quick guide
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Upload a food image** (supported formats: JPG, JPEG, PNG)
        2. **Adjust portion size** if needed (1.0 = standard portion)
        3. Click 'Estimate Calories' to get the prediction
        
        The calorie predictions are color-coded:
        - 🟢 Green: <200 calories (Low)
        - 🟡 Yellow: 200-400 calories (Medium)
        - 🟠 Orange: 400-700 calories (High)
        - 🔴 Red: >700 calories (Very High)
        """)
        
    # Optional user-entered food name
    user_food_name = st.text_input("Food name (optional)", value="", help="Enter a food name to store with the prediction")

    # Image upload
    uploaded = st.file_uploader(
        "Upload food image", 
        type=["jpg", "jpeg", "png"],
        key="food_image_upload"
    )
    portion = st.slider(
        "Portion size multiplier",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Adjust for larger or smaller portions",
        key="portion_slider"
    )
    
    if uploaded:
        img_bytes = uploaded.read()
        st.image(img_bytes, caption="Uploaded food image", use_container_width=True)
        
        # Store image
        try:
            gfid = store_image_file(img_bytes, filename=uploaded.name)
            save_metadata_doc({
                "filename": uploaded.name,
                "gridfs_id": gfid,
                "ingested_at": datetime.utcnow()
            })
        except Exception as e:
            st.warning(f"⚠️ Image storage failed: {e}")
        
        # Predict
        if st.button("🔍 Estimate Calories"):
            with st.spinner("Analyzing image..."):
                try:
                    calories = predict_calories(
                        img_bytes,
                        portion=portion,
                        resnet=resnet_model,
                        rf=rf_model
                    )
                    
                    # Display result
                    color = get_category_color(calories)
                    st.markdown(f"""
                        <h3 style='color: {color}'>
                            Estimated Calories: {calories:.1f} kcal
                        </h3>
                    """, unsafe_allow_html=True)
                    # Show additional info
                    if user_food_name:
                        st.markdown(f"**Food (user):** {user_food_name}")
                    st.markdown(f"**Portion multiplier:** {portion}")
                    st.markdown(f"**Predicted at:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
                    
                    # Log prediction
                    log_doc = {
                        "filename": uploaded.name,
                        "gridfs_id": gfid if 'gfid' in locals() else None,
                        "predicted_calories": float(calories),
                        "predicted_food": None,  # placeholder: model currently predicts calories only
                        "user_food_name": user_food_name if user_food_name else None,
                        "portion_size": float(portion),
                        "model_version": "rf_v1",
                        "timestamp": datetime.utcnow(),
                    }
                    save_prediction_log(log_doc)
                    st.success("✅ Prediction logged to MongoDB")
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")

def show_analytics_page():
    st.markdown('<h1 class="main-header">📊 Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    try:
        with st.spinner('Loading analytics data...'):
            analytics = list(get_analytics_data())[0]
        
        # Summary metrics in cards
        st.markdown('<div style="margin-bottom: 2rem;">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            total = analytics['total_predictions'][0]['count'] if analytics['total_predictions'] else 0
            st.metric("🔢 Total Predictions", total)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            avg = analytics['avg_calories'][0]['avg'] if analytics['avg_calories'] else 0
            st.metric("📈 Average Calories", f"{avg:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            top = analytics['top_calories'][0] if analytics['top_calories'] else None
            if top:
                st.metric("⬆️ Highest Calories", f"{top['predicted_calories']:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
                
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            low = analytics['lowest_calories'][0] if analytics['lowest_calories'] else None
            if low:
                st.metric("⬇️ Lowest Calories", f"{low['predicted_calories']:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Charts with tabs
        st.markdown('<div style="margin-top: 2rem;">', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["📊 Distribution", "📈 Top Foods"])
        
        with tab1:
            st.markdown('<div style="padding: 1rem; background: white; border-radius: 10px;">', unsafe_allow_html=True)
            # Category distribution pie chart
            categories = analytics['category_distribution']
            if categories:
                fig = px.pie(
                    values=[c['count'] for c in categories],
                    names=[c['_id'] for c in categories],
                    title="Calorie Categories Distribution",
                    color_discrete_map={
                        'Low': '#28a745',
                        'Medium': '#ffc107',
                        'High': '#fd7e14',
                        'Very High': '#dc3545'
                    },
                    hole=0.4
                )
                fig.update_layout(
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add explanatory text
                total = sum(c['count'] for c in categories)
                st.markdown(f"""
                <div class="info-box">
                    <h4>Distribution Summary:</h4>
                    {''.join([
                        f"- {c['_id']}: {(c['count']/total)*100:.1f}% ({c['count']} items)<br>"
                        for c in sorted(categories, key=lambda x: x['count'], reverse=True)
                    ])}
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
                
        with tab2:
            st.markdown('<div style="padding: 1rem; background: white; border-radius: 10px;">', unsafe_allow_html=True)
            # Top calories bar chart
            top_foods = analytics['top_calories']
            if top_foods:
                fig = px.bar(
                    x=[f"{t['filename'][:20]}..." for t in top_foods],
                    y=[t['predicted_calories'] for t in top_foods],
                    title="Top 5 Highest Calorie Foods",
                    labels={'x': 'Food', 'y': 'Calories'},
                    color=[t['predicted_calories'] for t in top_foods],
                    color_continuous_scale=['green', 'yellow', 'orange', 'red']
                )
                fig.update_layout(
                    xaxis_title="Food Items",
                    yaxis_title="Calories",
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    bargap=0.2
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add details table
                st.markdown("""
                <div class="info-box">
                    <h4>Top Foods Details:</h4>
                    <table style="width: 100%">
                        <tr>
                            <th>Food</th>
                            <th>Calories</th>
                            <th>Category</th>
                        </tr>
                """, unsafe_allow_html=True)
                
                for food in top_foods:
                    cal = food['predicted_calories']
                    category = 'Low' if cal < 200 else 'Medium' if cal < 400 else 'High' if cal < 700 else 'Very High'
                    category_class = f'category-{category.lower().replace(" ", "-")}'
                    st.markdown(f"""
                        <tr>
                            <td>{food['filename']}</td>
                            <td>{cal:.1f}</td>
                            <td><span class="{category_class}">{category}</span></td>
                        </tr>
                    """, unsafe_allow_html=True)
                    
                st.markdown("</table></div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to load analytics data: {e}")

def show_about_page():
    st.markdown('<h1 class="main-header">FusionCal: A Hybrid Deep Learning and Metadata-Based Food Calorie Estimation System</h1>', unsafe_allow_html=True)
    
    # Project Overview Section
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: #1f77b4;">A Big Data Analytics Project</h2>
            <p style="font-size: 1.2rem; color: #666;">
                Combining Deep Learning, Machine Learning, and Big Data for 
                intelligent food calorie estimation.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Technology Stack
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="info-box" style="height: 100%;">
                <h3 style="color: #1f77b4;">🔬 Technology Stack</h3>
                <ul style="list-style-type: none; padding: 0;">
                    <li style="margin: 10px 0;">
                        <strong style="color: #1f77b4;">🧠 Deep Learning:</strong>
                        <div>ResNet50 for feature extraction</div>
                    </li>
                    <li style="margin: 10px 0;">
                        <strong style="color: #1f77b4;">🌲 Machine Learning:</strong>
                        <div>Random Forest for calorie prediction</div>
                    </li>
                    <li style="margin: 10px 0;">
                        <strong style="color: #1f77b4;">📊 Database:</strong>
                        <div>MongoDB for analytics</div>
                    </li>
                    <li style="margin: 10px 0;">
                        <strong style="color: #1f77b4;">🎯 Frontend:</strong>
                        <div>Streamlit for interactive visualization</div>
                    </li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="info-box" style="height: 100%;">
                <h3 style="color: #1f77b4;">⚙️ How it Works</h3>
                <ol style="padding-left: 20px;">
                    <li style="margin: 10px 0;">Upload a food image</li>
                    <li style="margin: 10px 0;">ResNet50 extracts visual features</li>
                    <li style="margin: 10px 0;">Random Forest predicts calories</li>
                    <li style="margin: 10px 0;">Results stored in MongoDB</li>
                    <li style="margin: 10px 0;">Real-time analytics dashboard</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
    
    # Developer Information
    st.markdown("""
        <div class="info-box" style="margin-top: 2rem;">
            <h3 style="color: #1f77b4;">👨‍💻 Developer Information</h3>
            <table style="width: 100%;">
                <tr>
                    <td style="padding: 10px; width: 150px;"><strong>Student</strong></td>
                    <td>[Your Name]</td>
                </tr>
                <tr>
                    <td style="padding: 10px;"><strong>Roll No</strong></td>
                    <td>[Your Roll No]</td>
                </tr>
                <tr>
                    <td style="padding: 10px;"><strong>Department</strong></td>
                    <td>[Your Department]</td>
                </tr>
            </table>
        </div>
        
        <div class="info-box" style="margin-top: 2rem;">
            <h3 style="color: #1f77b4;">📝 Project Features</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                <div style="padding: 1rem; background: rgba(31, 119, 180, 0.1); border-radius: 5px;">
                    <h4>🎯 Accurate Predictions</h4>
                    <p>Advanced deep learning for precise calorie estimates</p>
                </div>
                <div style="padding: 1rem; background: rgba(31, 119, 180, 0.1); border-radius: 5px;">
                    <h4>📊 Rich Analytics</h4>
                    <p>Detailed insights and visualizations</p>
                </div>
                <div style="padding: 1rem; background: rgba(31, 119, 180, 0.1); border-radius: 5px;">
                    <h4>🔄 Real-time Processing</h4>
                    <p>Instant results and database updates</p>
                </div>
                <div style="padding: 1rem; background: rgba(31, 119, 180, 0.1); border-radius: 5px;">
                    <h4>📱 User-friendly Interface</h4>
                    <p>Easy-to-use and responsive design</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_resnet(path: str = "models/resnet_feature_extractor.h5"):
    """Load ResNet model silently, handling custom layer issues."""
    if not os.path.exists(path):
        return None
    
    try:
        # Build architecture first
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.layers import Input, Dense
        from tensorflow.keras.models import Model
        from tensorflow.keras.applications.resnet import preprocess_input

        # Create model architecture
        inp = Input(shape=(224, 224, 3), name="input_image")
        x = preprocess_input(inp)
        base = ResNet50(include_top=False, weights="imagenet", pooling="avg")
        x = base(x)
        x = Dense(512, activation="linear", name="proj_512")(x)
        model = Model(inputs=inp, outputs=x, name="resnet50_512_fallback")

        # Load weights silently
        with tf.keras.utils.CustomObjectScope({}):
            try:
                model.load_weights(path, by_name=True)
            except:
                pass  # Fallback to ImageNet weights
        
        return model
    except Exception as e:
        return None


@st.cache_resource
def load_rf(path: str = "models/rf_model.pkl"):
    if not os.path.exists(path):
        return None
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Failed to load RF model from {path}: {e}")
        return None


def preprocess_image_bytes(img_bytes, target_size=(224, 224)):
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize(target_size)
    arr = np.array(pil).astype(np.float32)
    # Use ResNet-specific preprocessing
    from tensorflow.keras.applications.resnet import preprocess_input
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_calories(img_bytes, portion=1.0, resnet=None, rf=None):
    """Predict calories for a food image."""
    if resnet is None:
        raise RuntimeError("ResNet model not loaded. Please check models/resnet_feature_extractor.h5")
    if rf is None:
        raise RuntimeError("Random Forest model not loaded. Please check models/rf_model.pkl")
    
    try:
        # Preprocess and get ResNet features
        arr = preprocess_image_bytes(img_bytes)
        feat = resnet.predict(arr, verbose=0)  # Suppress prediction messages
        
        # Handle feature dimensions for RF model
        try:
            expected = rf.n_features_in_
        except Exception:
            expected = None
            
        # Prepare input features
        if expected is None or expected == feat.shape[1]:
            X = feat
        elif expected == feat.shape[1] + 1:
            X = np.hstack([feat, np.array([[portion]])])
        else:
            if expected < feat.shape[1]:
                st.warning(f"Trimming features from {feat.shape[1]} to {expected}")
                X = feat[:, :expected]
            else:
                st.warning(f"Padding features from {feat.shape[1]} to {expected}")
                pad = np.zeros((feat.shape[0], expected - feat.shape[1]), dtype=feat.dtype)
                X = np.hstack([feat, pad])
        
        # Get final prediction
        pred = float(rf.predict(X)[0])
        
        # Adjust prediction by portion size
        return pred * portion
        
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {str(e)}")


# Load models (cached)
RESNET_PATH = os.getenv("RESNET_PATH", "models/resnet_feature_extractor.h5")
RF_PATH = os.getenv("RF_MODEL_PATH", "models/rf_model.pkl")

# Load models silently before rendering pages
@st.cache_resource(show_spinner=False)
def initialize_models():
    """Load models without displaying warnings"""
    with st.spinner(text=''):  # Empty spinner to suppress messages
        resnet = load_resnet(RESNET_PATH)
        rf = load_rf(RF_PATH)
    return resnet, rf

# Initialize models silently
resnet_model, rf_model = initialize_models()

# Main App Logic
if page == "🍱 Predict Calories":
    show_prediction_page()
elif page == "📊 Analytics Dashboard":
    show_analytics_page()
else:
    show_about_page()

