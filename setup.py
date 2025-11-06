from setuptools import setup, find_packages

setup(
    name="food_calorie_estimation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "tensorflow",
        "keras",
        "scikit-learn",
        "numpy",
        "pandas",
        "pillow",
        "joblib",
        "pymongo",
        "streamlit",
        "mongomock"
    ],
)