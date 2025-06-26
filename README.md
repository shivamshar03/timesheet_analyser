# 🧠 Project Report & Employee Timesheet Analysis

A powerful Streamlit-based dashboard that automates the generation of structured project reports and employee performance summaries using Groq AI (`llama3-8b-8192`) via LangChain. It also includes interactive visualizations using Plotly.

---

## 📌 Features

- 📤 Upload multiple `.txt` timesheet files
- 🤖 Uses LLMs to:
  - Summarize project progress
  - Analyze individual employee contributions
  - Rate performance and working hours
  - Identify blockers and next steps
- 📊 Generates interactive charts for:
  - Average working hours per employee
  - Performance ratings
- 📥 Download the generated report as a text file

---

## 📂 Folder Structure
```
project-timesheet-analysis/
├── app.py # Main Streamlit app
├── requirements.txt # Python dependencies
├── README.md # Project overview
└── .env # API key configuration

```

---

## ⚙️ Installation

1. **Clone this repo:**

```bash
git clone https://github.com/your-username/project-timesheet-analysis.git
cd project-timesheet-analysis
```
2. **Install dependencies:**

```bash
pip install -r requirements.txt
Set up .env file:
```
3. **Create a .env file in the project root and add your Groq API key:**

```
GROQ_API_KEY=your_groq_api_key_here

```
4. **Run the app:**
```bash
streamlit run app.py
```

## 📤 Sample Input Format
Upload one or more .txt files containing raw daily reports in plain English, like:

Name: John Doe
Date: 24/06/2025

Project: GameX
- Fixed animation bug
- Integrated leaderboard API

Login: 9:00 AM
Logout: 6:00 PM
📊 Visualizations
Bar chart for average working hours

Bar chart for performance ratings

These are auto-generated based on CSV data embedded in the LLM response.

🧠 Powered By

- Streamlit
- LangChain
- Groq LLMs
- Plotly
- Pandas

## 📄 License
This project is licensed under the MIT License.

## ✨ Author
Shivam Sharma
Founder, NexHub
Python & AI Developer
