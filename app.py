import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DEFAULT_XLSX = BASE_DIR / "Project_Tracking_Sheet_with_Dashboard.xlsx"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

st.set_page_config(page_title="Project Tracking AI Chatbot", page_icon="🤖", layout="wide")

# ============================================================
# NORMALIZATION / DATA LOADING
# ============================================================

def norm_col(c):
    if isinstance(c, pd.Timestamp):
        return c.strftime("%b-%y")
    if isinstance(c, __import__('datetime').datetime):
        return c.strftime("%b-%y")
    s = str(c).replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

ALIASES = {
    "S.No": ["S.No", "S No", "Sr No", "Serial No"],
    "Participant": ["Participant", "Participant's Name", "Participants Name", "Name"],
    "Employee Code": ["Employee Code", "Employee\nCode", "Emp Code"],
    "Location": ["Location", "Location / Unit", "Unit", "Location/Unit"],
    "Department": ["Department", "Dept", " ", "Unnamed: 5"],
    "Project": ["Project", "Project Theme / Title", "Project Theme", "Project Title"],
    "Direction": ["Direction", "Improvement Direction"],
    "UOM": ["UOM", "Baseline UOM"],
    "Baseline Period": ["Baseline Period"],
    "Baseline": ["Baseline", "Baseline Data"],
    "Target": ["Target"],
    "Average Actual": ["Average Actual", "Average (Actual)"],
    "Progress": ["Progress", "Progress %"],
    "Status": ["Status"],
    "Project File": ["Project File"],
    "Review Date": ["Review Date", "Review date"],
    "Comment": ["Comment", "Comments"],
    "Target Date": ["Target Date"],
}

MONTH_COLUMNS = [f"{m}-26" for m in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]]


def find_excel():
    candidates = [
        DEFAULT_XLSX,
        BASE_DIR / "Project Tracking Sheet with Dashboard.xlsx",
        BASE_DIR / "Project_Tracking_Sheet_with_Dashboard.xlsm",
    ]
    for p in candidates:
        if p.exists():
            return p
    files = list(BASE_DIR.glob("*.xlsx"))
    if len(files) == 1:
        return files[0]
    return None


def standardize_columns(df):
    original = list(df.columns)
    cleaned = [norm_col(c) for c in original]
    df.columns = cleaned

    rename = {}
    for canonical, variants in ALIASES.items():
        for v in variants:
            nv = norm_col(v)
            if nv in df.columns:
                rename[nv] = canonical
                break
    df = df.rename(columns=rename)

    # Duplicate columns after renaming: retain first non-empty version.
    if df.columns.duplicated().any():
        out = pd.DataFrame(index=df.index)
        for c in dict.fromkeys(df.columns):
            same = df.loc[:, df.columns == c]
            out[c] = same.bfill(axis=1).iloc[:, 0]
        df = out

    # Convert date-looking monthly headers into standard names.
    month_map = {}
    for c in list(df.columns):
        try:
            dt = pd.to_datetime(c, format="%b-%y")
            if dt.year == 2026 and dt.month <= 12:
                month_map[c] = dt.strftime("%b-%y")
        except Exception:
            pass
    df = df.rename(columns=month_map)

    # If a blank department column is really a department column, keep it.
    if "Department" not in df.columns:
        blank = [c for c in df.columns if str(c).strip() == ""]
        if blank:
            df = df.rename(columns={blank[0]: "Department"})

    return df


@st.cache_data(show_spinner=False)
def read_excel(path_str):
    path = Path(path_str)
    # The user's workbook uses row 5 as the header (header=4).
    raw = pd.read_excel(path, sheet_name="Project Tracking", header=4)
    return standardize_columns(raw)


excel_path = find_excel()

if excel_path is None:
    st.title("🤖 Project Tracking AI Chatbot")
    st.warning("Excel file was not found beside app.py. Upload the workbook below.")
    uploaded = st.file_uploader("Upload Project_Tracking_Sheet_with_Dashboard.xlsx", type=["xlsx", "xlsm"])
    if uploaded is None:
        st.stop()
    temp = BASE_DIR / "_uploaded_project_tracking.xlsx"
    temp.write_bytes(uploaded.getbuffer())
    excel_path = temp

try:
    df = read_excel(str(excel_path))
except Exception as e:
    st.error(f"Could not read the Excel workbook: {e}")
    st.stop()

# Numeric/date cleanup
for c in MONTH_COLUMNS + ["Baseline", "Target", "Average Actual", "Progress"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

for c in ["Review Date", "Target Date"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

# Ensure expected columns exist so queries don't crash.
for c in ["Participant", "Project", "Department", "Location", "Direction", "UOM", "Status", "Comment"]:
    if c not in df.columns:
        df[c] = ""

AVAILABLE_MONTHS = [c for c in MONTH_COLUMNS if c in df.columns]
MONTH_DATES = {c: pd.to_datetime(c, format="%b-%y") for c in AVAILABLE_MONTHS}
LAST_DATA_MONTH = max(MONTH_DATES.values()) if MONTH_DATES else pd.Timestamp("2026-09-01")

# ============================================================
# GENERAL HELPERS
# ============================================================

def clean(v):
    if pd.isna(v):
        return ""
    return str(v).strip()


def qnorm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def direction_higher_better(row):
    d = clean(row.get("Direction", "")).lower()
    if any(x in d for x in ["reduce", "reduction", "decrease", "lower", "minimize", "less"]):
        return False
    return True


def target_met(row, actual_key="Average Actual"):
    actual = pd.to_numeric(pd.Series([row.get(actual_key, np.nan)]), errors="coerce").iloc[0]
    target = pd.to_numeric(pd.Series([row.get("Target", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(actual) or pd.isna(target):
        return None
    return bool(actual >= target) if direction_higher_better(row) else bool(actual <= target)


def unique_values(col):
    return sorted(df[col].dropna().astype(str).str.strip().replace("", np.nan).dropna().unique().tolist()) if col in df.columns else []


def find_person(text):
    t = text.lower()
    people = sorted(unique_values("Participant"), key=len, reverse=True)
    for p in people:
        if p.lower() in t:
            return p
    # Match full name word sets.
    words = set(qnorm(text).split())
    for p in people:
        pw = set(qnorm(p).split())
        if pw and pw.issubset(words):
            return p
    return None


def find_project(text):
    t = text.lower()
    projects = sorted(unique_values("Project"), key=len, reverse=True)
    for p in projects:
        if p.lower() in t:
            return p
    return None


def search_df(text):
    terms = [x for x in qnorm(text).split() if x not in {
        "show", "give", "me", "all", "the", "details", "detail", "of", "project", "projects",
        "for", "what", "will", "achieve", "forecast", "predict", "in", "month", "target", "summary"
    }]
    if not terms:
        return df.copy()
    searchable = [c for c in ["Participant", "Project", "Department", "Location", "Direction", "UOM", "Status", "Comment", "Batch"] if c in df.columns]
    text_series = pd.Series("", index=df.index)
    for c in searchable:
        text_series = text_series + " " + df[c].fillna("").astype(str).str.lower()
    mask = pd.Series(True, index=df.index)
    for term in terms:
        mask &= text_series.str.contains(re.escape(term), regex=True, na=False)
    result = df[mask]
    if result.empty:
        # OR search as fallback
        mask = pd.Series(False, index=df.index)
        for term in terms:
            mask |= text_series.str.contains(re.escape(term), regex=True, na=False)
        result = df[mask]
    return result

# ============================================================
# SUMMARY / ANALYTICS
# ============================================================

def summary_data():
    total = len(df)
    achieved = 0
    attention = 0
    nodata = 0
    for _, row in df.iterrows():
        r = target_met(row)
        if r is True:
            achieved += 1
        elif r is False:
            attention += 1
        else:
            nodata += 1
    avg_progress = pd.to_numeric(df.get("Progress", pd.Series(dtype=float)), errors="coerce").mean()
    status = df["Status"].fillna("No Data").astype(str).value_counts().to_dict()
    dept = df["Department"].fillna("Unknown").astype(str).replace("", "Unknown").value_counts().to_dict()
    return {"total": total, "achieved": achieved, "attention": attention, "nodata": nodata,
            "avg_progress": avg_progress, "status": status, "department": dept}


def trend_table(rows):
    out = []
    for _, r in rows.iterrows():
        vals = [(m, r.get(m, np.nan)) for m in AVAILABLE_MONTHS if not pd.isna(r.get(m, np.nan))]
        if len(vals) >= 2:
            y = np.array([float(v) for _, v in vals])
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            change = y[-1] - y[0]
        else:
            slope = np.nan
            change = np.nan
        out.append({"Participant": clean(r.get("Participant")), "Project": clean(r.get("Project")),
                    "Latest Actual": vals[-1][1] if vals else np.nan, "Change First→Latest": change, "Monthly Trend": slope})
    return pd.DataFrame(out)

# ============================================================
# FORECAST
# ============================================================

def parse_month(text):
    patterns = [
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+20\d{2}\b",
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[- ]20\d{2}\b"
    ]
    for pat in patterns:
        m = re.search(pat, text.lower())
        if m:
            try:
                return pd.to_datetime(m.group(0), errors="coerce")
            except Exception:
                pass
    return None


def months_between(a, b):
    return (b.year - a.year) * 12 + b.month - a.month


def forecast_row(row, target_date):
    vals = []
    dates = []
    for m in AVAILABLE_MONTHS:
        v = row.get(m, np.nan)
        if not pd.isna(v):
            vals.append(float(v))
            dates.append(MONTH_DATES[m])
    if not vals:
        return None
    y = np.array(vals, dtype=float)
    steps = max(1, months_between(LAST_DATA_MONTH, target_date))
    if len(y) == 1:
        pred = y[-1]
        lower = upper = pred
        method = "Last actual"
    else:
        x = np.arange(len(y), dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        trend = intercept + slope * (len(y) - 1 + steps)
        ma = float(np.mean(y[-3:]))
        pred = 0.70 * trend + 0.30 * ma
        fitted = intercept + slope * x
        resid = y - fitted
        sd = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
        margin = 1.96 * sd * np.sqrt(steps)
        lower, upper = pred - margin, pred + margin
        method = "70% linear trend + 30% 3-month moving average"
    target = pd.to_numeric(pd.Series([row.get("Target", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(target):
        status = "Target unavailable"
    else:
        good = pred >= target if direction_higher_better(row) else pred <= target
        status = "Forecast meets target" if good else "Forecast misses target"
    return {"Participant": clean(row.get("Participant")), "Project": clean(row.get("Project")),
            "UOM": clean(row.get("UOM")), "Direction": clean(row.get("Direction")),
            "Target": target, "Forecast": pred, "Lower 95%": lower, "Upper 95%": upper,
            "Target Status": status, "Method": method}


def forecast_horizon(text):
    target = parse_month(text)
    if target is not None:
        return [target]
    m = re.search(r"next\s+(\d+)\s+months?", text.lower())
    if m:
        n = max(1, min(int(m.group(1)), 24))
        return [LAST_DATA_MONTH + pd.DateOffset(months=i) for i in range(1, n + 1)]
    return [LAST_DATA_MONTH + pd.DateOffset(months=1)]

# ============================================================
# QUERY ROUTER
# ============================================================

def is_forecast(q):
    return any(x in q.lower() for x in ["forecast", "predict", "projection", "will achieve", "expected", "future performance"])


def is_summary(q):
    return any(x in q.lower() for x in ["overall summary", "project summary", "total projects", "how many projects", "overall project status"])


def is_attention(q):
    return any(x in q.lower() for x in ["needs attention", "need attention", "below target", "behind target", "not achieved", "missed target"])


def is_achieved(q):
    return any(x in q.lower() for x in ["achieved target", "met target", "meeting target", "above target"])


def is_details(q):
    return any(x in q.lower() for x in ["full details", "complete details", "all details", "project details", "details of", "tell me about", "all information"])


def build_result(question):
    q = question.lower()

    if is_forecast(q):
        person = find_person(question)
        project = find_project(question)
        rows = df.copy()
        if person:
            rows = rows[rows["Participant"].fillna("").astype(str).str.lower() == person.lower()]
        if project:
            rows = rows[rows["Project"].fillna("").astype(str).str.lower() == project.lower()]
        if not person and not project:
            s = search_df(question)
            if not s.empty:
                rows = s
        dates = forecast_horizon(question)
        recs = []
        for d in dates:
            for _, row in rows.iterrows():
                r = forecast_row(row, d)
                if r:
                    r["Forecast Month"] = d.strftime("%b-%Y")
                    recs.append(r)
        return {"type": "forecast", "data": pd.DataFrame(recs), "person": person, "project": project}

    if is_summary(q):
        return {"type": "summary", "data": summary_data()}

    if is_attention(q):
        rows = df[[target_met(r) is False for _, r in df.iterrows()]]
        return {"type": "table", "title": "Projects Needing Attention", "data": rows}

    if is_achieved(q):
        rows = df[[target_met(r) is True for _, r in df.iterrows()]]
        return {"type": "table", "title": "Projects Achieving Target", "data": rows}

    if is_details(q):
        person = find_person(question)
        project = find_project(question)
        rows = df.copy()
        if person:
            rows = rows[rows["Participant"].fillna("").astype(str).str.lower() == person.lower()]
        if project:
            rows = rows[rows["Project"].fillna("").astype(str).str.lower() == project.lower()]
        if not person and not project:
            rows = search_df(question)
        return {"type": "details", "title": "Project Details", "data": rows}

    # Common analytical queries handled locally.
    if any(x in q for x in ["trend", "improving", "declining", "performance trend"]):
        rows = search_df(question)
        if rows.empty:
            rows = df
        return {"type": "table", "title": "Performance Trend", "data": trend_table(rows)}

    # Default: search relevant rows.
    rows = search_df(question)
    if rows.empty:
        rows = df.head(50)
    return {"type": "table", "title": "Relevant Project Data", "data": rows}

# ============================================================
# GROQ CONTEXT
# ============================================================

def compact_df(data, max_rows=100):
    if data is None or data.empty:
        return "No matching rows."
    x = data.copy().head(max_rows)
    for c in x.columns:
        if pd.api.types.is_datetime64_any_dtype(x[c]):
            x[c] = x[c].dt.strftime("%Y-%m-%d")
    return x.to_csv(index=False)


def context_for_groq(question, result):
    if result["type"] == "summary":
        return str(result["data"])
    return compact_df(result.get("data"), 100)


def groq_answer(question, result):
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return "Groq API key is not configured. The structured result above is still available. Add GROQ_API_KEY to `.env` to enable AI explanations."
    try:
        client = Groq(api_key=api_key)
        context = context_for_groq(question, result)
        prompt = f"""
You are the AI assistant for a manufacturing project-tracking workbook.
Answer the user's question using ONLY the supplied data.
Never invent numbers. Use Indian manufacturing/business language.
IMPORTANT LANGUAGE RULE: Respond ONLY in English. All headings, explanations, summaries, labels, tables, and forecast commentary must be in English. If the user asks in Hindi or another language, understand the question internally but still answer entirely in English. Do not translate the response into Hindi. Do not use Devanagari script.
If a forecast is shown, call it an estimate and mention the method only when useful.
Latest actual month is {LAST_DATA_MONTH.strftime('%B %Y')}.

USER QUESTION:
{question}

STRUCTURED DATA:
{context}
"""
        r = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Be accurate, concise and analytical. Do not fabricate data. ALWAYS respond entirely in English. Never use Hindi, Devanagari, or any non-English response text."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1800,
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"Groq could not answer the natural-language part. The structured result above is still valid. Error: {e}"

# ============================================================
# DISPLAY
# ============================================================

def display_result(result):
    typ = result["type"]
    if typ == "summary":
        s = result["data"]
        a,b,c,d = st.columns(4)
        a.metric("Total Projects", s["total"])
        b.metric("Achieved", s["achieved"])
        c.metric("Needs Attention", s["attention"])
        d.metric("Average Progress", "N/A" if pd.isna(s["avg_progress"]) else f"{s['avg_progress']:.1f}%")
        st.subheader("Department-wise Projects")
        st.dataframe(pd.DataFrame(list(s["department"].items()), columns=["Department", "Projects"]), use_container_width=True, hide_index=True)
        st.subheader("Status")
        st.dataframe(pd.DataFrame(list(s["status"].items()), columns=["Status", "Projects"]), use_container_width=True, hide_index=True)
        return
    data = result.get("data", pd.DataFrame()).copy()
    if data.empty:
        st.info("No matching data found.")
        return
    if typ == "forecast":
        for c in ["Target", "Forecast", "Lower 95%", "Upper 95%"]:
            if c in data.columns:
                data[c] = pd.to_numeric(data[c], errors="coerce").round(2)
        st.dataframe(data, use_container_width=True, hide_index=True)
        st.caption("Forecast method: 70% linear trend + 30% three-month moving average. The 95% interval is indicative, not a guarantee.")
        return
    # Put key columns first.
    preferred = ["S.No", "Participant", "Employee Code", "Batch", "Location", "Department", "Project", "Direction", "UOM", "Baseline Period", "Baseline", "Target"] + AVAILABLE_MONTHS + ["Average Actual", "Progress", "Status", "Review Date", "Comment", "Target Date", "Project File"]
    cols = [c for c in preferred if c in data.columns] + [c for c in data.columns if c not in preferred]
    st.dataframe(data[cols], use_container_width=True, hide_index=True)

# ============================================================
# UI
# ============================================================

st.title("🤖 Project Tracking AI Chatbot")
st.caption("Ask about any employee, project, department, target, trend, status or future forecast.")

c1, c2, c3 = st.columns(3)
c1.metric("Projects Loaded", len(df))
c2.metric("Latest Actual", LAST_DATA_MONTH.strftime("%b-%Y"))
c3.metric("Groq Model", GROQ_MODEL)

with st.sidebar:
    st.header("Controls")
    st.success(f"Excel loaded: {excel_path.name}")
    st.write(f"Rows: **{len(df)}**")
    st.write(f"Columns: **{len(df.columns)}**")
    st.divider()
    st.markdown("**Try:**")
    st.code("Give me overall project summary", language=None)
    st.code("Show all projects of Shantimoy Som", language=None)
    st.code("Give full details of Shantimoy Som", language=None)
    st.code("Which projects need attention?", language=None)
    st.code("What will Shantimoy Som achieve in November 2026?", language=None)
    st.code("Forecast the MBF project for the next 3 months", language=None)
    st.code("Show projects with improving trend", language=None)
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "result" in msg:
            display_result(msg["result"])
            if msg.get("answer"):
                st.markdown("### 🤖 AI Analysis")
                st.markdown(msg["answer"])
        else:
            st.markdown(msg["content"])

question = st.chat_input("Ask any question about your project tracker...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Analyzing workbook..."):
            result = build_result(question)
            display_result(result)
            answer = groq_answer(question, result)
            st.markdown("### 🤖 AI Analysis")
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "result": result, "answer": answer})
