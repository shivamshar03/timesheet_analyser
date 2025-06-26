import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
import plotly.express as px
from io import StringIO
import re

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Set up LangChain + Groq LLM
llm = ChatGroq(
    temperature=0.9,
    groq_api_key=groq_api_key,
    model_name="llama3-8b-8192"
)

# Prompt Template for LLM
prompt_template = PromptTemplate(
    input_variables=["timesheet_data"],
    template="""
You are a Project Manager reviewing daily timesheet reports submitted by a team of game developers.

Based on the data provided below, generate the following:

1. **Project Summary**:
   - List all ongoing projects.
   - For each project, use the project name as a heading and summarize progress in bullet points.
   - Mention contributors for each project.

2. **Employee Performance Analysis**:
   For each employee:
   - Name and list of project contributions.
   - Key achievements or issues.
   - Honest performance rating (out of 5).
   - Login/logout times and average working hours.

3. **Work Hours Table**:
   Table with: Employee Name | Login Time | Logout Time | Avg Working Hours

4. **Highlights & Recommendations**:
   - Mention breakthroughs, blockers, collaboration.
   - Suggest improvements or task reassignments.

5. 📊 At the end, output chart data as CSV like this:
--RAW DATA FOR VISUALIZATION--
employee_name,average_hours,rating
Shubhanshu Shrimali,9.5,4.5
...
--END OF DATA--

perl
Copy
Edit
⚠️ Only include numeric values. Omit rows with missing or "varies" data.

-- RAW TIMESHEET DATA START --
{timesheet_data}
-- RAW TIMESHEET DATA END --
"""
)

chain = LLMChain(llm=llm, prompt=prompt_template)

# Extract CSV block from the LLM result
def extract_csv_block(text):
    pattern = r"--RAW DATA FOR VISUALIZATION--\s*(.*?)\s*--END OF DATA--"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

# Plot charts using Plotly
def plot_charts(csv_data):
    try:
        df = pd.read_csv(StringIO(csv_data))
        df.columns = [col.strip().lower() for col in df.columns]

        if "name" in df.columns:
            df.rename(columns={"name": "employee_name"}, inplace=True)

        if "employee_name" not in df.columns or "average_hours" not in df.columns or "rating" not in df.columns:
            st.warning("📛 CSV format incorrect. Columns missing.")
            return

        df = df[pd.to_numeric(df["average_hours"], errors="coerce").notnull()]
        df["average_hours"] = df["average_hours"].astype(float)
        df["rating"] = df["rating"].astype(float)

        st.subheader("📊 Average Working Hours per Employee")
        fig1 = px.bar(
            df,
            x="employee_name",
            y="average_hours",
            color="average_hours",
            color_continuous_scale="Blues",
            title="Average Working Hours"
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("⭐ Performance Ratings")
        fig2 = px.bar(
            df,
            x="employee_name",
            y="rating",
            color="rating",
            color_continuous_scale="Oranges",
            title="Employee Ratings (Out of 5)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Failed to generate charts: {e}")

# Streamlit UI
st.set_page_config(page_title="🧠 Project Report Generator", layout="wide")
st.title("AI-Powered Timesheet Analysis")

uploaded_files = st.file_uploader("📤 Upload One or More Timesheet Text Files", type=["txt"], accept_multiple_files=True)

if uploaded_files:
    all_data = ""
    for file in uploaded_files:
        try:
            all_data += file.read().decode("utf-8") + "\n"
        except Exception as e:
            st.warning(f"❌ Failed to read {file.name}: {e}")

    if all_data.strip():
        with st.spinner(" Generating report..."):
            try:
                result = chain.invoke({"timesheet_data": all_data})
                report_text = result["text"]
                csv_data = extract_csv_block(report_text)

                st.subheader("📋 Generated Project Report")
                st.markdown(report_text)

                st.download_button("📥 Download Full Report", data=report_text, file_name="project_report.txt")

                if csv_data:
                    plot_charts(csv_data)
                else:
                    st.warning("⚠️ No chart data available in the response.")

            except Exception as e:
                st.error(f"⚠️ Error during AI processing: {e}")
    else:
        st.warning("📝 No readable content found in uploaded files.")
else:
    st.info("Please upload one or more `.txt` files containing timesheet logs.")