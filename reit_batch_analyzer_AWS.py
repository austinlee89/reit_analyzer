import os
import time
import re
import boto3
import tempfile
import yfinance as yf
from datetime import datetime, timedelta
from decimal import Decimal
from collections import defaultdict
from google import genai
from google.genai import types

# --- CONFIGURATION ---
API_KEY = "insert-your-GeminiAPI-key-here"
S3_BUCKET_NAME = "reit-data"
DYNAMODB_TABLE_NAME = "reit-metrics"

# --- CONTROL FLAGS ---
# List tickers here to force re-analysis.
# Formats: "ALL", "TNT-UN", "TNT-UN:2023_Q3"
FORCE_REDO = ["TNT-UN"]

# --- TICKER MAPPING ---
CANADIAN_REITS = {
    "RioCan": "REI-UN",
    "Granite": "GRT-UN",
    "Allied": "AP-UN",
    "CAPREIT": "CAR-UN",
    "Dream_Industrial": "DIR-UN",
    "Dream_Office": "D-UN",
    "SmartCentres": "SRU-UN",
    "Choice_Properties": "CHP-UN",
    "Killam": "KMP-UN",
    "InterRent": "IIP-UN",
    "Boardwalk": "BEI-UN",
    "H_and_R": "HR-UN",
    "Chartwell": "CSH-UN",
    "NorthWest_Healthcare": "NWH-UN",
    "Crombie": "CRR-UN",
    "True_North": "TNT-UN" # Added TNT
}

# Initialize Clients
client = genai.Client(api_key=API_KEY)
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(DYNAMODB_TABLE_NAME)

# Load System Instruction
try:
    with open("system_instruction.txt", "r", encoding="utf-8") as f:
        SYSTEM_INSTRUCTION = f.read()
except FileNotFoundError:
    try:
        with open("system_instructions.txt", "r", encoding="utf-8") as f:
            SYSTEM_INSTRUCTION = f.read()
    except:
        print("Error: 'system_instruction.txt' not found.")
        exit()

# --- HELPER: DATE & PRICE FETCHING ---
def get_period_end_date(period_str):
    """Converts '2024_Q3' or '2023_YE' into a YYYY-MM-DD string."""
    match = re.match(r"(\d{4})_(Q\d|YE|Annual)", period_str, re.IGNORECASE)
    if not match:
        return None

    year = int(match.group(1))
    suffix = match.group(2).upper()

    if suffix == "Q1": return f"{year}-03-31"
    if suffix == "Q2": return f"{year}-06-30"
    if suffix == "Q3": return f"{year}-09-30"
    # Q4, YE (Year End), and Annual all map to Dec 31
    if suffix in ["Q4", "YE", "ANNUAL"]: return f"{year}-12-31"
    return None

def fetch_historical_price(ticker, date_str):
    """Fetches the closing price on (or near) the specific date using yfinance."""
    clean_ticker = ticker.replace('.UN', '-UN').replace(' ', '')
    if not clean_ticker.endswith('.TO'):
        clean_ticker += '.TO'

    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        start_date = target_date - timedelta(days=5)
        end_date_query = target_date + timedelta(days=1)

        ticker_obj = yf.Ticker(clean_ticker)
        hist = ticker_obj.history(start=start_date, end=end_date_query)

        if not hist.empty:
            price = hist['Close'].iloc[-1]
            return float(price)
    except Exception as e:
        print(f"    ⚠️ YFinance Error for {ticker}: {e}")
        return None
    return None

# --- HELPER: PARSE MARKDOWN TO DYNAMODB FORMAT ---
def parse_report_to_metrics(report_text, period_str):
    """
    Parses Markdown report into DynamoDB format.
    Handles mixed 3-column (Standard) and 4-column (Dual-Key) tables.
    """
    metrics_map = {}
    metadata = {}

    # 1. Extract Metadata
    name_match = re.search(r"\*\*Name:\*\*\s*(.*)", report_text)
    if name_match:
        metadata['Full_Name'] = name_match.group(1).strip()

    # 2. Define Key Map (Updated with New Fields from System Instructions)
    key_map = {
        # Earnings
        "Revenue": "Revenue",
        "Adj. EBITDA": "Adj_EBITDA",
        "FFO": "FFO_Total",  # New Raw $
        "AFFO": "AFFO_Total", # New Raw $
        "FFO per Unit": "FFO_per_Unit",
        "AFFO per Unit": "AFFO_per_Unit",
        "AFFO Payout Ratio": "Payout_Ratio",
        "P/FFO": "P_FFO",
        "P/AFFO": "P_AFFO",

        # Debt
        "Net Debt": "Net_Debt",
        "Debt-to-GBV": "Debt_to_GBV",
        "Net Debt-to-EBITDA": "Net_Debt_to_EBITDA",
        "Interest Coverage": "Interest_Coverage",

        # Book Value / NAV
        "Equity": "Equity",
        "Unit Count": "Unit_Count_Basic",
        "Unit Count (Diluted)": "Unit_Count_Diluted",
        "Share Price (Period End)": "Share_Price_Period_End", # Critical New Field
        "Market Capitalization": "Market_Cap",
        "Market Cap": "Market_Cap", # Robustness
        "Enterprise Value": "Enterprise_Value",
        "NAV per Unit": "NAV_per_Unit",
        "P/NAV (PBR)": "P_NAV",
        "Reported Cap Rate": "Reported_Cap_Rate",
        "Implied Stabilized NOI": "Implied_Stabilized_NOI",
        "Implied Cap Rate": "Implied_Cap_Rate",

        # Operational
        "Occupancy": "Occupancy",
        "WALT": "WALT",
        "SPNOI Growth": "SPNOI_Growth"
    }

    # 4. Line-by-Line Table Parsing
    lines = report_text.split('\n')

    for line in lines:
        line = line.strip()
        # Skip non-table lines or separator lines
        if not line.startswith('|') or '---' in line or 'Metric' in line:
            continue

        # Split by pipe | and strip whitespace
        parts = [p.strip() for p in line.split('|') if p.strip()]

        if not parts: continue

        # Identify Metric Name (Must be in **Bold**)
        metric_name_match = re.search(r"\*\*(.*?)\*\*", parts[0])
        if not metric_name_match:
            continue

        clean_name = metric_name_match.group(1).strip()

        # Fuzzy Key Match (Exact match first, then robust fallback)
        db_key = key_map.get(clean_name)

        if not db_key:
            continue

        # --- UPDATED CLEAN_DECIMAL (Handles accounting negative, *, %, LTM) ---
        def clean_decimal(v):
            if not v: return None
            v = str(v).strip()
            if v.upper() in ["N/A", "-", ""]: return None

            # Handle Accounting Negative (1.6) -> -1.6
            if '(' in v and ')' in v:
                v = v.replace('(', '-').replace(')', '')

            # Remove standard chars
            v = v.replace('$', '').replace('%', '').replace('x', '').replace(',', '')

            # Extract valid number using Regex (handles 19400* or 83000 (LTM))
            # Finds signed float/int at start of string
            match = re.search(r'-?\d+(\.\d+)?', v)
            if not match:
                return None

            try: return Decimal(match.group(0))
            except: return None

        # --- LOGIC FOR DUAL-KEY TABLES (4 COLUMNS) ---
        # Format: | Metric | 3M Value | 12M Value | Source |
        if len(parts) == 4:
            val_3m_str = parts[1]
            val_12m_str = parts[2]
            source_str = parts[3]

            entry = {"Source": source_str}
            val_3m = clean_decimal(val_3m_str)
            val_12m = clean_decimal(val_12m_str)

            if val_3m is not None: entry["Value_3M"] = val_3m

            # GUARDRAIL REMOVED: Now accepting the 2nd column value regardless of quarter
            # This allows YTD/LTM data to be saved in Value_12M bucket
            if val_12m is not None:
                entry["Value_12M"] = val_12m

            if "Value_3M" in entry or "Value_12M" in entry:
                metrics_map[db_key] = entry

        # --- LOGIC FOR STANDARD TABLES (3 COLUMNS) ---
        # Format: | Metric | Value | Source |
        elif len(parts) == 3:
            value_str = parts[1]
            source_str = parts[2]

            entry = {"Source": source_str}
            val = clean_decimal(value_str)

            if val is not None:
                entry["Value"] = val
                metrics_map[db_key] = entry

    return metrics_map, metadata

# --- CORE FUNCTION: ANALYZE PACKET ---
def analyze_packet(reit_name, reit_ticker, period, local_file_paths):
    print(f"    Using Gemini to analyze documents...")

    # 1. Fetch Price Context First
    price_context = ""
    end_date = get_period_end_date(period)
    if end_date:
        hist_price = fetch_historical_price(reit_ticker, end_date)
        if hist_price:
            print(f"    💲 Fetched historical price for {end_date}: ${hist_price:.2f}")
            # We inject this strictly into the prompt
            price_context = (
                f"\n\nCONTEXTUAL DATA (DO NOT ASK USER): "
                f"The trading price of {reit_ticker} on {end_date} was ${hist_price:.2f}. "
                f"Use this price to calculate Market Cap, P/FFO, and Implied Cap Rate if not explicitly stated."
            )
        else:
            print(f"    ⚠️ Could not fetch price for {end_date}. Analysis will rely on PDF data.")

    # 2. Upload Files
    uploaded_files = []
    try:
        for path in local_file_paths:
            uploaded_file = client.files.upload(file=path)
            uploaded_files.append(uploaded_file)

        for uf in uploaded_files:
            while True:
                file_check = client.files.get(name=uf.name)
                if file_check.state == "ACTIVE": break
                elif file_check.state == "FAILED": return None
                time.sleep(1)

        # 3. Construct Prompt with Price Context
        prompt_content = [
            SYSTEM_INSTRUCTION,
            *uploaded_files,
            f"Generate the Forensic Report for {reit_name} ({period}) based on these documents.{price_context}"
        ]

        # 4. Generate
        response = client.models.generate_content(
            model="gemini-3-flash-preview", # Use valid stable model
            contents=prompt_content,
            config=types.GenerateContentConfig(
                temperature=0.1
            )
        )
        return response.text
    except Exception as e:
        print(f"    Error: {e}")
        return None
    finally:
        for uf in uploaded_files:
            try: client.files.delete(name=uf.name)
            except: pass

# --- MAIN EXECUTION ---
def main():
    print("🚀 Starting Smart Cloud Pipeline...")

    # 1. Scan S3
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET_NAME)

    pdf_packets = defaultdict(list)
    existing_reports = set()

    print("🔍 Scanning S3 Bucket inventory...")
    for page in pages:
        if 'Contents' not in page: continue
        for obj in page['Contents']:
            key = obj['Key']
            if key.lower().endswith('.pdf'):
                parts = key.split('/')
                if len(parts) >= 3:
                    pdf_packets[(parts[0], parts[1])].append(key)
            elif key.lower().endswith('.md'):
                existing_reports.add(key)

    if not pdf_packets:
        print("No PDF packets found in S3.")
        return

    print(f"✅ Found {len(pdf_packets)} PDF packets.")

    # 2. Process Packets
    for (folder_name, period), s3_keys in pdf_packets.items():

        reit_ticker = CANADIAN_REITS.get(folder_name, folder_name)

        expected_md_filename = f"{reit_ticker}_{period}_Report.md"
        expected_s3_key = f"{folder_name}/{period}/{expected_md_filename}"
        exists = expected_s3_key in existing_reports

        # --- FORCE REDO LOGIC ---
        force_period_tag = f"{reit_ticker}:{period}" # E.g., AP-UN:2024_Q3

        is_force_all = "ALL" in FORCE_REDO
        is_force_ticker = reit_ticker in FORCE_REDO
        is_force_period = force_period_tag in FORCE_REDO

        if exists and not (is_force_all or is_force_ticker or is_force_period):
            print(f"⏩ Skipping {reit_ticker} ({period}) - Analysis already exists.")
            continue

        if is_force_period:
            print(f"🔄 Force Re-analyzing Period: {reit_ticker} ({period})...")
        elif is_force_ticker:
            print(f"🔄 Force Re-analyzing Ticker: {reit_ticker} ({period})...")
        elif is_force_all:
            print(f"🔄 Force Re-analyzing ALL: {reit_ticker} ({period})...")
        else:
            print(f"\n📂 Processing New Packet: {reit_ticker} ({period})")

        with tempfile.TemporaryDirectory() as temp_dir:
            local_paths = []

            # A. Download
            for key in s3_keys:
                filename = os.path.basename(key)
                local_path = os.path.join(temp_dir, filename)
                s3.download_file(S3_BUCKET_NAME, key, local_path)
                local_paths.append(local_path)

            # B. Analyze (Passing Ticker now)
            report_text = analyze_packet(folder_name, reit_ticker, period, local_paths)

            if report_text:
                # C. Upload
                print(f"    ⬆️ Uploading Report: {expected_s3_key}")
                s3.put_object(Body=report_text, Bucket=S3_BUCKET_NAME, Key=expected_s3_key)

                # D. DB Update
                print("    ⚙️ Updating DynamoDB...")
                # Pass 'period' to enable anti-hallucination guardrails
                metrics_map, metadata = parse_report_to_metrics(report_text, period)

                item = {
                    'REIT_Ticker': reit_ticker,
                    'Period_ISO': period,
                    'REIT_Name': metadata.get('Full_Name', folder_name),
                    'Analysis_Date': str(time.strftime("%Y-%m-%d")),
                    'S3_Link_Report': f"s3://{S3_BUCKET_NAME}/{expected_s3_key}",
                    'Metrics': metrics_map,
                }

                try:
                    table.put_item(Item=item)
                    print(f"    ✅ Success: {reit_ticker} ({period}) saved.")
                except Exception as e:
                    print(f"    ❌ DB Error: {e}")

        time.sleep(2)

if __name__ == "__main__":
    main()

