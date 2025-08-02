@echo off
echo Installing required packages for Audio Classification Streamlit App...
echo.

:: Install packages from requirements.txt
pip install -r requirements.txt

echo.
echo Installation completed!
echo.
echo To run the Streamlit app, use the following command:
echo streamlit run streamlit_app.py
echo.
pause
