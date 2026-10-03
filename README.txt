PROJECT TRACKING AI CHATBOT - GROQ + STREAMLIT

1. Keep these files together:
   app.py
   Project_Tracking_Sheet_with_Dashboard.xlsx
   requirements.txt
   .env (create this yourself)
   run_chatbot.bat

2. Create .env in the same folder as app.py:
   GROQ_API_KEY=YOUR_GROQ_API_KEY
   GROQ_MODEL=openai/gpt-oss-20b

3. Install Python 3.11 or 3.12 if possible. Then open PowerShell in this folder:
   python -m pip install --upgrade pip setuptools wheel
   python -m pip install -r requirements.txt

4. Start:
   python -m streamlit run app.py
   OR double-click run_chatbot.bat

5. The app automatically looks for the Excel file beside app.py. It also provides an upload fallback if the workbook is missing.

Supported examples:
- overall project summary
- all projects of any employee
- full details of any employee/project
- projects needing attention
- projects achieving target
- trends / improving / declining
- forecast any employee/project for a named future month
- forecast next N months
- natural-language AI explanations using Groq

Forecast method:
70% linear trend + 30% three-month moving average, with an indicative 95% interval.
Forecasts are estimates, not guarantees.
