# MongoDB Setup Guide

## Installation (Windows)

1. Download MongoDB Community Server:
   - Visit: https://www.mongodb.com/try/download/community
   - Select "Windows" and "msi" package
   - Download the latest version

2. Install MongoDB:
   - Run the downloaded .msi installer
   - Choose "Complete" installation
   - Install MongoDB Compass (optional GUI)
   - Complete the installation

3. Data directory is already created at:
   ```
   C:\data\db
   ```

## Starting MongoDB

1. Start MongoDB server (in a new PowerShell window with admin rights):
   ```powershell
   "C:\Program Files\MongoDB\Server\{version}\bin\mongod.exe" --dbpath="C:\data\db"
   ```
   Replace `{version}` with your installed version (e.g., 7.0)

2. Test connection (in another PowerShell):
   ```powershell
   "C:\Program Files\MongoDB\Server\{version}\bin\mongosh.exe"
   ```

3. The app will automatically connect to MongoDB at `mongodb://localhost:27017`

## Verifying Installation

The app uses these collections in `food_calorie_db`:
- `features`: Stores ResNet feature vectors
- `images`: Stores uploaded images
- `predictions`: Stores calorie predictions

To verify setup:
1. Start MongoDB
2. Run Streamlit app
3. Upload and predict an image
4. Check MongoDB Compass to see stored data

## Troubleshooting

If MongoDB won't start:
1. Ensure the service isn't already running:
   ```powershell
   Get-Service -Name MongoDB
   ```
2. Check logs in `C:\data\db\mongod.log`
3. Verify permissions on `C:\data\db`