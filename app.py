import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
import plotly.express as px

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(temperature=0.9, model_name="llama3-8b-8192", groq_api_key=groq_api_key)

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["timesheet_data"],
    template="""
You are a Project Manager reviewing daily timesheet reports submitted by a team of game developers.

Based on the data provided below, generate the following:

1. Project Summary:
   - List all ongoing projects.
   - Summarize each project’s key updates and progress in bullet points , it should be like main heading (project name ) then poject highlighted  subpoints.
   - Mention contributors for each project.

2. Employee Performance Analysis:
   For each employee:
   - Mention their name and project contributions.
   - Highlight key achievements or issues.
   - Provide a very honest performance rating (out of 5 stars).
   - Note login and logout times, and calculate average working hours.

3. Work Hours Table:
   A table showing average login duration per day per employee with observations.

4. Important Highlights and Recommendations:
   - Mention technical breakthroughs, collaboration insights, and any blockers.
   - Suggest next steps, improvements, or adjustments in work assignments.

-- RAW TIMESHEET DATA START --
{timesheet_data}
-- RAW TIMESHEET DATA END --

Also, include at the bottom:
--RAW DATA FOR VISUALIZATION--
employee_name,average_hours,rating
... (based on your analysis)
--END OF DATA--

Give a professional and structured report.
"""
)

# LangChain LLM Chain
chain = LLMChain(llm=llm, prompt=prompt_template)

# Streamlit UI
st.set_page_config(page_title="Timesheet Dashboard - Upload & Analyze", layout="wide")
st.title("📤 Upload Timesheet & Generate Report")

uploaded_file = st.file_uploader("Upload your `.txt` timesheet file", type=["txt"])

if uploaded_file is not None:
    timesheet_data = uploaded_file.read().decode("utf-8")

    # Save file in session_state to be reused in other pages
    st.session_state["uploaded_file_name"] = uploaded_file.name
    st.session_state["uploaded_file_bytes"] = uploaded_file.getvalue()

    with st.spinner("Generating report..."):
        result = chain.invoke({"timesheet_data": timesheet_data})

    # Extract raw report and CSV data
    report_text = result.get("text", "")
    st.subheader("📋 Generated Report")
    st.markdown(report_text)

    # Attempt to parse RAW DATA section
    try:
        csv_start = report_text.index("--RAW DATA FOR VISUALIZATION--") + len("--RAW DATA FOR VISUALIZATION--")
        csv_end = report_text.index("--END OF DATA--")
        raw_csv = report_text[csv_start:csv_end].strip()

        from io import StringIO
        df = pd.read_csv(StringIO(raw_csv))

        st.subheader("📊 Visual Insights")
        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.bar(df, x="employee_name", y="average_hours", color="employee_name", title="Average Working Hours")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.bar(df, x="employee_name", y="rating", color="employee_name", title="Performance Ratings")
            st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.warning(f"⚠️ Could not parse visualization data: {e}")

    st.download_button(
        label="📥 Download Full Report",
        data=report_text,
        file_name="project_report.txt",
        mime="text/plain"
    )
else:
    st.info("Please upload a valid `.txt` file containing daily report logs.")
