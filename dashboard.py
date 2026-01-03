import streamlit as st
import boto3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import os

st.set_page_config(page_title="REIT Analyst Pro", layout="wide", initial_sidebar_state="expanded")

# --- HACK: INJECT AWS CREDENTIALS FOR STREAMLIT CLOUD ---

if 'AWS_ACCESS_KEY_ID' in st.secrets:
    os.environ['AWS_ACCESS_KEY_ID'] = st.secrets['AWS_ACCESS_KEY_ID']
    os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets['AWS_SECRET_ACCESS_KEY']
    os.environ['AWS_DEFAULT_REGION'] = st.secrets['AWS_REGION']
    
# --- CONFIGURATION ---
TABLE_NAME = "reit-metrics"
AWS_REGION = "us-west-2"  # Update to your region

# --- 1. CONNECT TO DYNAMODB ---
@st.cache_resource
def get_dynamo_resource():
    return boto3.resource('dynamodb', region_name=AWS_REGION)

table = get_dynamo_resource().Table(TABLE_NAME)

# --- 2. DATA FETCHING ---
def fetch_all_data():
    """Fetches all items from DynamoDB table, handling pagination."""
    response = table.scan()
    data = response['Items']
    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        data.extend(response['Items'])
    return data

def parse_period_to_date(period_str):
    try:
        parts = period_str.split('_')
        year = int(parts[0])
        suffix = parts[1].upper()

        if suffix == 'Q1': return datetime(year, 3, 31)
        if suffix == 'Q2': return datetime(year, 6, 30)
        if suffix == 'Q3': return datetime(year, 9, 30)
        return datetime(year, 12, 31)
    except:
        return None

# --- 3. SPLIT NORMALIZATION ENGINE (UNIT COUNT METHOD) ---
def normalize_splits_by_units(df):
    """
    Detects share consolidations/splits by looking for massive cliffs in Unit Counts.
    Returns:
        df: The normalized dataframe.
        split_info: A dictionary of {Ticker: "Detected 1-for-X split around [Date]"} for display.
    """
    if df.empty or 'Unit_Count_Diluted' not in df.columns:
        return df, {}

    per_unit_cols = ['FFO_per_Unit', 'AFFO_per_Unit', 'NAV_per_Unit', 'Share_Price_Period_End']
    count_cols = ['Unit_Count_Basic', 'Unit_Count_Diluted']

    adjusted_frames = []
    split_info_map = {}

    for reit_ticker in df['REIT'].unique():
        reit_df = df[df['REIT'] == reit_ticker].copy()
        reit_df = reit_df.sort_values('Period_Date', ascending=False)

        current_factor = 1.0
        unit_values = reit_df['Unit_Count_Diluted'].values
        dates = reit_df['Period_Date'].values
        factors = [1.0] * len(reit_df)

        detected_event = None

        for i in range(len(reit_df) - 1):
            newer_units = unit_values[i]
            older_units = unit_values[i+1]

            if pd.isna(newer_units) or pd.isna(older_units) or newer_units == 0 or older_units == 0:
                factors[i+1] = current_factor
                continue

            ratio = older_units / newer_units

            if ratio > 1.8:
                current_factor *= ratio
                if not detected_event:
                    d = pd.to_datetime(dates[i]).strftime('%Y-%m-%d')
                    detected_event = f"Detected 1-for-{ratio:.2f} Consolidation around {d}"
            elif ratio < 0.6:
                current_factor *= ratio
                if not detected_event:
                    d = pd.to_datetime(dates[i]).strftime('%Y-%m-%d')
                    detected_event = f"Detected {1/ratio:.2f}-for-1 Split around {d}"

            factors[i+1] = current_factor

        reit_df['Split_Factor'] = factors
        if detected_event:
            split_info_map[reit_ticker] = detected_event

        for col in per_unit_cols:
            targets = [col, f"{col}_3M", f"{col}_12M", f"{col}_Annual_Actual", f"{col}_Q_Annualized"]
            for t in targets:
                if t in reit_df.columns: reit_df[t] = reit_df[t] * reit_df['Split_Factor']

        for col in count_cols:
            if col in reit_df.columns: reit_df[col] = reit_df[col] / reit_df['Split_Factor']

        adjusted_frames.append(reit_df)

    if not adjusted_frames: return df, {}
    return pd.concat(adjusted_frames), split_info_map

# --- PROCESS DATA ---
def process_data(items):
    rows = []
    for item in items:
        period = item.get('Period_ISO', '')
        parts = period.split('_')
        year = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
        suffix = parts[1] if len(parts) > 1 else ""
        is_quarterly = 'Q' in period.upper()

        row = {
            'REIT': item.get('REIT_Ticker'),
            'Period': period,
            'Year': year,
            'Quarter_Suffix': suffix,
            'Summary': item.get('Analyst_Summary'),
            'Link_Report': item.get('S3_Link_Report'),
            'Is_Quarterly': is_quarterly,
            'Is_Projected': False
        }
        row['Period_Date'] = parse_period_to_date(period)

        metrics = item.get('Metrics', {})
        for key, val_obj in metrics.items():
            if not val_obj: continue
            val_3m = float(val_obj['Value_3M']) if 'Value_3M' in val_obj else None
            val_12m = float(val_obj['Value_12M']) if 'Value_12M' in val_obj else None
            val_generic = float(val_obj['Value']) if 'Value' in val_obj else None

            if val_3m is not None: row[f"{key}_3M"] = val_3m
            if val_12m is not None: row[f"{key}_12M"] = val_12m
            if val_generic is not None: row[key] = val_generic
            row[f"{key}_Source"] = val_obj.get('Source', '')

            flow_metrics = ['FFO_per_Unit', 'AFFO_per_Unit', 'Revenue', 'Adj_EBITDA', 'FFO_Total', 'AFFO_Total']
            base_q = val_3m if val_3m is not None else val_generic
            if base_q is not None:
                if key in flow_metrics: row[f"{key}_Q_Annualized"] = base_q * 4
                else: row[f"{key}_Q_Annualized"] = base_q

            base_a = val_12m if val_12m is not None else val_generic
            if base_a is not None: row[f"{key}_Annual_Actual"] = base_a

        rows.append(row)
    return pd.DataFrame(rows)

def format_period_axis(df):
    if not df.empty and 'Period' in df.columns:
        df['Period_Clean'] = df['Period'].apply(lambda x: f"{x.split('_')[1]}-{x.split('_')[0][2:]}" if len(x.split('_'))==2 else x)
        df['Hover_Label'] = df['Period_Clean']
    return df

# --- VALUATION ENGINES ---
@st.cache_data(ttl=3600)
def calculate_historical_valuation(ticker_symbol, fundamentals_df, metric='FFO_per_Unit'):
    clean_ticker = ticker_symbol.replace('.UN', '-UN').replace(' ', '')
    if clean_ticker == 'NET-UN': clean_ticker += '.V'
    elif not clean_ticker.endswith('.TO') and not clean_ticker.endswith('.V'): clean_ticker += '.TO'

    start_date = fundamentals_df['Period_Date'].min() - timedelta(days=90)
    try:
        stock = yf.Ticker(clean_ticker)
        prices = stock.history(start=start_date, interval="1wk")
        if prices.empty: return None, None
        prices = prices[['Close']].reset_index()
        prices['Date'] = prices['Date'].dt.tz_localize(None)
    except: return None, None

    col_name = f"{metric}_Q_Annualized"
    if col_name not in fundamentals_df.columns: col_name = metric

    fund_data = fundamentals_df[['Period_Date', col_name]].copy()
    fund_data = fund_data.sort_values('Period_Date').set_index('Period_Date')
    fund_daily = fund_data.resample('D').ffill()

    merged = pd.merge_asof(prices, fund_daily, left_on='Date', right_index=True, direction='backward')
    merged['Multiple'] = merged['Close'] / merged[col_name]
    merged = merged.dropna(subset=['Multiple'])

    stats = {
        'mean': merged['Multiple'].mean(),
        'upper_1std': merged['Multiple'].mean() + merged['Multiple'].std(),
        'lower_1std': merged['Multiple'].mean() - merged['Multiple'].std()
    }
    return merged, stats

@st.cache_data(ttl=3600)
def fetch_live_market_data(ticker_symbol):
    clean_ticker = ticker_symbol.replace('.UN', '-UN').replace(' ', '')
    if clean_ticker == 'NET-UN': clean_ticker += '.V'
    elif not clean_ticker.endswith('.TO') and not clean_ticker.endswith('.V'): clean_ticker += '.TO'
    try:
        info = yf.Ticker(clean_ticker).info
        return {
            "Live_Price": info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose'),
            "Live_Yield": info.get('dividendYield', 0) if info.get('dividendYield') else 0
        }
    except: return None

@st.cache_data(ttl=3600)
def calculate_live_implied_cap(ticker_symbol, fundamentals_df):
    clean_ticker = ticker_symbol.replace('.UN', '-UN').replace(' ', '')
    if clean_ticker == 'NET-UN': clean_ticker += '.V'
    elif not clean_ticker.endswith('.TO') and not clean_ticker.endswith('.V'): clean_ticker += '.TO'

    start_date = fundamentals_df['Period_Date'].min() - timedelta(days=90)
    try:
        stock = yf.Ticker(clean_ticker)
        prices = stock.history(start=start_date, interval="1wk")
        if prices.empty: return None
        prices = prices[['Close']].reset_index()
        prices['Date'] = prices['Date'].dt.tz_localize(None)
    except: return None

    req_cols = ['Period_Date', 'NAV_per_Unit', 'Debt_to_GBV', 'Reported_Cap_Rate']
    available_cols = [c for c in req_cols if c in fundamentals_df.columns]
    if len(available_cols) != len(req_cols): return None

    df_calc = fundamentals_df[req_cols].copy().sort_values('Period_Date')
    df_calc['Debt_Pct'] = df_calc['Debt_to_GBV'] / 100.0
    df_calc['Cap_Rate_Dec'] = df_calc['Reported_Cap_Rate'] / 100.0
    df_calc['GAV_per_Unit'] = df_calc['NAV_per_Unit'] / (1 - df_calc['Debt_Pct'])
    df_calc['Debt_per_Unit'] = df_calc['GAV_per_Unit'] - df_calc['NAV_per_Unit']
    df_calc['NOI_per_Unit'] = df_calc['GAV_per_Unit'] * df_calc['Cap_Rate_Dec']

    fund_daily = df_calc.set_index('Period_Date').resample('D').ffill()
    merged = pd.merge_asof(prices, fund_daily, left_on='Date', right_index=True, direction='backward')
    merged['P_NAV_Ratio'] = merged['Close'] / merged['NAV_per_Unit']
    merged['Live_Implied_Cap'] = (merged['NOI_per_Unit'] / (merged['Close'] + merged['Debt_per_Unit'])) * 100
    return merged.dropna(subset=['P_NAV_Ratio', 'Live_Implied_Cap'])

# --- 3. HELPER FUNCTIONS ---
def add_year_shading(fig, df):
    if df.empty: return fig
    date_col = 'Date' if 'Date' in df.columns else 'Period_Date'
    if date_col not in df.columns: return fig

    min_date = df[date_col].min()
    max_date = df[date_col].max()
    min_year, max_year = min_date.year, max_date.year

    for year in range(min_year, max_year + 1):
        if year % 2 == 0:
            fig.add_vrect(x0=datetime(year, 1, 1), x1=datetime(year, 12, 31), fillcolor="gray", opacity=0.1, layer="below", line_width=0)
        fig.add_annotation(x=datetime(year, 7, 1), y=0, yref="paper", text=str(year), showarrow=False, yshift=-25, font=dict(size=18, color="gray"))
    return fig

def calculate_cagr(df, metric, periods_count):
    try:
        if len(df) < 2: return 0.0
        start_val, end_val = df.iloc[0][metric], df.iloc[-1][metric]
        return ((end_val - start_val) / start_val) * 100 if start_val != 0 else 0.0
    except: return 0.0

# --- DASHBOARD UI CONFIG ---
st.set_page_config(page_title="REIT Analyst Pro", layout="wide", initial_sidebar_state="expanded")

with st.spinner("Initializing Analyst Terminal..."):
    raw_items = fetch_all_data()
    raw_df = process_data(raw_items)
    df, split_info_map = normalize_splits_by_units(raw_df)
    df = format_period_axis(df)

if df.empty:
    st.error("No data found in DynamoDB."); st.stop()

st.sidebar.title("🔍 REIT Selector")
view_mode = st.sidebar.radio("View Mode", ["Single REIT Deep Dive", "Head-to-Head Comparison"])
all_reits = sorted(df['REIT'].unique().tolist())

color_palette = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
reit_color_map = {reit: color_palette[i % len(color_palette)] for i, reit in enumerate(all_reits)}

if view_mode == "Single REIT Deep Dive":
    selected_reit = st.sidebar.selectbox("Select REIT", all_reits)
    filtered_df = df[df['REIT'] == selected_reit].copy()
    comparison_reits = [selected_reit]
else:
    comparison_reits = st.sidebar.multiselect("Select REITs", all_reits, default=all_reits[:2])
    filtered_df = df[df['REIT'].isin(comparison_reits)].copy()

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Global Visual Settings")
use_annualized = st.sidebar.checkbox("Annualize Quarterly Data", value=True, help="Multiplies Q1-Q3 FFO/AFFO/Rev by 4.")
freq_option = st.sidebar.radio("Data Frequency", ["Annual Only", "Quarterly Only"], index=0)

# --- GLOBAL FILTERING (MULTI-REIT AWARE) ---
chart_df = filtered_df.copy()
max_year = chart_df['Year'].max()

if freq_option == "Annual Only":
    annual_data = chart_df[chart_df['Quarter_Suffix'].isin(['Q4', 'YE', 'Annual', 'ANNUAL'])].copy()
    if not annual_data.empty:
        annual_data['Hover_Label'] = annual_data['Year'].astype(str) + " Annual"

    final_annual_frames = [annual_data]
    for reit in comparison_reits:
        reit_subset = chart_df[chart_df['REIT'] == reit]
        if reit_subset.empty: continue
        reit_max_year = reit_subset['Year'].max()
        has_annual = False
        if not annual_data.empty:
            has_annual = not annual_data[(annual_data['REIT'] == reit) & (annual_data['Year'] == reit_max_year)].empty
        if not has_annual:
            latest_q = reit_subset[reit_subset['Year'] == reit_max_year].iloc[[-1]].copy()
            if not latest_q.empty:
                latest_q['Is_Projected'] = True
                suffix = latest_q['Quarter_Suffix'].iloc[0] if 'Quarter_Suffix' in latest_q.columns else "Q?"
                latest_q['Hover_Label'] = f"{reit_max_year} Projected ({suffix})"
                final_annual_frames.append(latest_q)
    chart_df = pd.concat(final_annual_frames)
    if not chart_df.empty:
        chart_df['Period_Date'] = chart_df['Year'].apply(lambda y: datetime(y, 7, 1))

elif freq_option == "Quarterly Only":
    year_counts = chart_df.groupby(['REIT', 'Year']).size()
    valid_years = year_counts[(year_counts >= 4) | (year_counts.index.get_level_values('Year') == max_year)].reset_index()
    valid_years['Valid'] = True
    chart_df = pd.merge(chart_df, valid_years[['REIT', 'Year', 'Valid']], on=['REIT', 'Year'], how='inner')
    chart_df['Hover_Label'] = chart_df['Period_Clean']

# --- HEADER ---
st.title(f"📊 {comparison_reits[0] if len(comparison_reits)==1 else 'Multi-REIT'} Analysis")
if len(comparison_reits) == 1:
    reit = filtered_df.iloc[0]['REIT']
    latest = filtered_df.iloc[-1]
    mkt = fetch_live_market_data(reit)
    price = mkt['Live_Price'] if mkt and mkt['Live_Price'] else (latest.get('FFO_per_Unit', 0) * 10)
    src = "Live" if mkt and mkt['Live_Price'] else "Est"
    div_yield = mkt['Live_Yield'] if mkt and mkt['Live_Yield'] else 0
    ffo_metric = 'FFO_per_Unit_Q_Annualized' if 'FFO_per_Unit_Q_Annualized' in latest else 'FFO_per_Unit'
    affo_metric = 'AFFO_per_Unit_Q_Annualized' if 'AFFO_per_Unit_Q_Annualized' in latest else 'AFFO_per_Unit'
    affo_annual = latest.get(affo_metric, 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Stock Price", f"${price:.2f}", src)
    c2.metric("Yield", f"{div_yield:.2f}%")
    c3.metric("P/AFFO", f"{price/affo_annual:.1f}x" if affo_annual > 0 else "N/A")
    c4.metric("P/NAV", f"{price/latest.get('NAV_per_Unit', 1):.2f}x" if latest.get('NAV_per_Unit') else "N/A")
    c5.metric("Implied Cap", f"{latest.get('Implied_Cap_Rate', 0):.1f}%")
    st.markdown("---")

for r in comparison_reits:
    if r in split_info_map:
        st.warning(f"⚠️ **{r}:** {split_info_map[r]}. Historical per-unit metrics normalized.")

tab_perf, tab_health, tab_value, tab_source = st.tabs(["📈 Performance", "❤️ Health", "💎 Valuation", "📂 Forensic Audit"])

with tab_perf:
    st.subheader("Earnings & Growth Engine")
    col_ctrl, col_chart = st.columns([1, 3])
    with col_ctrl:
        st.info("Select Metric to Visualize")
        metric_map = {
            "FFO per Unit": "FFO_per_Unit", "AFFO per Unit": "AFFO_per_Unit",
            "FFO (Total)": "FFO_Total", "AFFO (Total)": "AFFO_Total",
            "Revenue (Total)": "Revenue", "Adj. EBITDA": "Adj_EBITDA",
            "AFFO Payout Ratio": "Payout_Ratio", "Historical P/FFO": "P_FFO_Historical",
            "Historical P/AFFO": "P_AFFO_Historical"
        }
        selected_metric_label = st.radio("Primary Metric", list(metric_map.keys()))
        base_metric = metric_map[selected_metric_label]

        show_bands = False
        if base_metric in ["P_FFO_Historical", "P_AFFO_Historical"]:
            show_bands = st.checkbox("Show Valuation Bands (Mean ± 1σ)", value=False)

        plot_metric = f"{base_metric}_Plot"
        if freq_option == "Annual Only":
            if f"{base_metric}_Annual_Actual" in chart_df.columns: chart_df[plot_metric] = chart_df[f"{base_metric}_Annual_Actual"]
            else: chart_df[plot_metric] = np.nan
            if f"{base_metric}_Q_Annualized" in chart_df.columns:
                chart_df.loc[chart_df['Is_Projected'] == True, plot_metric] = chart_df.loc[chart_df['Is_Projected'] == True, f"{base_metric}_Q_Annualized"]
            if base_metric in chart_df.columns: chart_df[plot_metric] = chart_df[plot_metric].fillna(chart_df[base_metric])
        else:
            if use_annualized:
                 if f"{base_metric}_Q_Annualized" in chart_df.columns: plot_metric = f"{base_metric}_Q_Annualized"
                 else: plot_metric = base_metric
            else:
                 if f"{base_metric}_3M" in chart_df.columns: plot_metric = f"{base_metric}_3M"
                 else: plot_metric = base_metric

        if len(comparison_reits) == 1 and base_metric not in ["P_FFO_Historical", "P_AFFO_Historical"]:
            growth = calculate_cagr(chart_df, plot_metric, len(chart_df))
            st.metric(f"Total Growth", f"{growth:.1f}%")

    with col_chart:
        if base_metric in ["P_FFO_Historical", "P_AFFO_Historical"]:
            target_fund_metric = "FFO_per_Unit" if base_metric == "P_FFO_Historical" else "AFFO_per_Unit"
            metric_label_clean = "P/FFO" if base_metric == "P_FFO_Historical" else "P/AFFO"
            fig_hist = go.Figure()
            global_min_date = datetime.now()
            global_max_date = datetime(2000, 1, 1)

            for r in comparison_reits:
                hist_data, stats = calculate_historical_valuation(r, filtered_df[filtered_df['REIT']==r], target_fund_metric)
                if hist_data is not None:
                    min_d = hist_data['Date'].min()
                    max_d = hist_data['Date'].max()
                    if min_d < global_min_date: global_min_date = min_d
                    if max_d > global_max_date: global_max_date = max_d
                    color = reit_color_map[r]
                    fig_hist.add_trace(go.Scatter(x=hist_data['Date'], y=hist_data['Multiple'], mode='lines', name=f"{r} {metric_label_clean}", line=dict(width=2, color=color)))
                    if show_bands:
                        fig_hist.add_hline(y=stats['mean'], line_dash="dot", line_color=color, annotation_text=f"{r} Avg: {stats['mean']:.1f}x", annotation_position="top left")
                        fig_hist.add_hrect(y0=stats['lower_1std'], y1=stats['upper_1std'], fillcolor=color, opacity=0.1, line_width=0)

            fig_hist = add_year_shading(fig_hist, pd.DataFrame({'Date': [global_min_date, global_max_date]}))
            fig_hist.update_layout(title=f"Weekly {selected_metric_label}", xaxis=dict(showgrid=False, showticklabels=False, range=[global_min_date, global_max_date]), yaxis=dict(showgrid=True, title="Multiple (x)"), autosize=True, height=400, font=dict(size=18), margin=dict(l=40, r=40, t=40, b=40))
            st.plotly_chart(fig_hist, config={'responsive': True}, theme=None)
        else:
            if plot_metric in chart_df.columns and not chart_df.empty:
                fig_perf = go.Figure()
                for reit in comparison_reits:
                    subset = chart_df[chart_df['REIT'] == reit]
                    if subset.empty: continue
                    actuals = subset[subset['Is_Projected'] == False]
                    projections = subset[subset['Is_Projected'] == True]
                    color = reit_color_map[reit]
                    if freq_option == "Annual Only":
                        fig_perf.add_trace(go.Bar(x=actuals['Period_Date'], y=actuals[plot_metric], name=reit, marker_color=color, customdata=actuals['Hover_Label'], hovertemplate="%{customdata}<br>%{y:,.2f}<extra></extra>"))
                        if not projections.empty: fig_perf.add_trace(go.Bar(x=projections['Period_Date'], y=projections[plot_metric], name=f"{reit} (Proj)", marker=dict(color=color, opacity=0.6, pattern_shape='/'), customdata=projections['Hover_Label'], hovertemplate="%{customdata}<br>%{y:,.2f}<extra></extra>"))
                    else:
                        fig_perf.add_trace(go.Scatter(x=subset['Period_Date'], y=subset[plot_metric], mode='lines+markers', name=reit, line=dict(width=3, color=color), marker=dict(size=8), customdata=subset['Hover_Label'], hovertemplate="%{customdata}<br>%{y:,.2f}<extra></extra>"))
                fig_perf = add_year_shading(fig_perf, chart_df)
                fig_perf.update_layout(title=f"Historical {selected_metric_label}", hovermode="x unified", xaxis=dict(showgrid=False, showticklabels=False), yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'), autosize=True, height=350, margin=dict(l=40, r=40, t=40, b=40), font=dict(size=18))
                st.plotly_chart(fig_perf, config={'responsive': True}, theme=None)
            else: st.warning(f"Metric '{selected_metric_label}' not found.")

    st.subheader("Portfolio Stability Metrics")
    c1, c2 = st.columns(2)
    occ_col = 'Occupancy_Plot'
    chart_df[occ_col] = np.nan
    if freq_option == "Annual Only":
        if 'Occupancy_Annual_Actual' in chart_df.columns: chart_df[occ_col] = chart_df['Occupancy_Annual_Actual']
        if 'Occupancy_Q_Annualized' in chart_df.columns: chart_df[occ_col] = chart_df[occ_col].fillna(chart_df['Occupancy_Q_Annualized'])
    else:
        if 'Occupancy_Q_Annualized' in chart_df.columns: chart_df[occ_col] = chart_df['Occupancy_Q_Annualized']
    if 'Occupancy' in chart_df.columns: chart_df[occ_col] = chart_df[occ_col].fillna(chart_df['Occupancy'])

    if occ_col in chart_df.columns:
        fig_occ = go.Figure()
        for reit in comparison_reits:
            subset = chart_df[chart_df['REIT'] == reit]
            if subset.empty: continue
            actuals = subset[subset['Is_Projected'] == False]
            projections = subset[subset['Is_Projected'] == True]
            color = reit_color_map[reit]
            if freq_option == "Annual Only":
                 fig_occ.add_trace(go.Bar(x=actuals['Period_Date'], y=actuals[occ_col], name=reit, marker_color=color, customdata=actuals['Hover_Label'], hovertemplate="%{customdata}<br>%{y:.2f}%<extra></extra>"))
                 if not projections.empty: fig_occ.add_trace(go.Bar(x=projections['Period_Date'], y=projections[occ_col], name=f"{reit} (Proj)", marker=dict(color=color, opacity=0.6, pattern_shape='/'), customdata=projections['Hover_Label'], hovertemplate="%{customdata}<br>%{y:.2f}%<extra></extra>"))
            else:
                 fig_occ.add_trace(go.Scatter(x=subset['Period_Date'], y=subset[occ_col], fill='tozeroy', name=reit, line=dict(color=color), customdata=subset['Hover_Label'], hovertemplate="%{customdata}<br>%{y:.2f}%<extra></extra>"))
        fig_occ.update_layout(title="Occupancy %", yaxis_range=[50, 100], autosize=True, height=350, xaxis=dict(showgrid=False, showticklabels=False), font=dict(size=18))
        fig_occ.add_hline(y=95, line_dash="dot", line_color="green", annotation_text="Target")
        fig_occ = add_year_shading(fig_occ, chart_df)
        c1.plotly_chart(fig_occ, config={'responsive': True}, theme=None)

    spnoi_col = 'SPNOI_Plot'
    chart_df[spnoi_col] = np.nan
    if freq_option == "Annual Only":
        if 'SPNOI_Growth_Annual_Actual' in chart_df.columns: chart_df[spnoi_col] = chart_df['SPNOI_Growth_Annual_Actual']
        if 'SPNOI_Growth_Q_Annualized' in chart_df.columns:
            chart_df.loc[chart_df['Is_Projected'] == True, spnoi_col] = chart_df.loc[chart_df['Is_Projected'] == True, 'SPNOI_Growth_Q_Annualized']
            chart_df[spnoi_col] = chart_df[spnoi_col].fillna(chart_df['SPNOI_Growth_Q_Annualized'])
    else:
        if 'SPNOI_Growth_Q_Annualized' in chart_df.columns: chart_df[spnoi_col] = chart_df['SPNOI_Growth_Q_Annualized']
    if 'SPNOI_Growth' in chart_df.columns: chart_df[spnoi_col] = chart_df[spnoi_col].fillna(chart_df['SPNOI_Growth'])

    if spnoi_col in chart_df.columns:
        fig_spnoi = go.Figure()
        for reit in comparison_reits:
            subset = chart_df[chart_df['REIT'] == reit]
            if subset.empty: continue
            actuals = subset[subset['Is_Projected'] == False]
            projections = subset[subset['Is_Projected'] == True]
            color = reit_color_map[reit]
            if len(comparison_reits) == 1:
                color_logic = subset[spnoi_col].apply(lambda x: '#00CC96' if x >= 0 else '#EF553B')
                marker_dict = dict(color=color_logic)
                marker_proj = dict(color=color_logic, opacity=0.6, pattern_shape='/')
            else:
                marker_dict = dict(color=color)
                marker_proj = dict(color=color, opacity=0.6, pattern_shape='/')
            if freq_option == "Annual Only":
                fig_spnoi.add_trace(go.Bar(x=actuals['Period_Date'], y=actuals[spnoi_col], marker=marker_dict, name=reit, customdata=actuals['Hover_Label'], hovertemplate="%{customdata}<br>%{y:.2f}%<extra></extra>"))
                if not projections.empty: fig_spnoi.add_trace(go.Bar(x=projections['Period_Date'], y=projections[spnoi_col], marker=marker_proj, name=f"{reit} (Proj)", customdata=projections['Hover_Label'], hovertemplate="%{customdata}<br>%{y:.2f}%<extra></extra>"))
            else:
                fig_spnoi.add_trace(go.Bar(x=subset['Period_Date'], y=subset[spnoi_col], marker=dict(color=color), name=reit, customdata=subset['Hover_Label'], hovertemplate="%{customdata}<br>%{y:.2f}%<extra></extra>"))
        fig_spnoi = add_year_shading(fig_spnoi, chart_df)
        fig_spnoi.update_layout(title="Same Property NOI Growth %", yaxis=dict(title="% Growth"), autosize=True, height=350, xaxis=dict(showgrid=False, showticklabels=False), font=dict(size=18))
        c2.plotly_chart(fig_spnoi, config={'responsive': True}, theme=None)

with tab_health:
    st.subheader("Solvency & Leverage Analysis")
    if 'Net_Debt_to_EBITDA' in chart_df.columns:
        fig_ebitda = go.Figure()
        for reit in comparison_reits:
            subset = chart_df[chart_df['REIT'] == reit]
            actuals = subset[subset['Is_Projected'] == False]
            projections = subset[subset['Is_Projected'] == True]
            color = reit_color_map[reit]
            if freq_option == "Annual Only":
                fig_ebitda.add_trace(go.Bar(x=actuals['Period_Date'], y=actuals['Net_Debt_to_EBITDA'], name=reit, marker_color=color, customdata=actuals['Hover_Label'], hovertemplate="%{customdata}<br>%{y:.1f}x<extra></extra>"))
                if not projections.empty: fig_ebitda.add_trace(go.Bar(x=projections['Period_Date'], y=projections['Net_Debt_to_EBITDA'], marker=dict(color=color, opacity=0.6, pattern_shape='/'), name=f"{reit} (Proj)", customdata=projections['Hover_Label'], hovertemplate="%{customdata}<br>%{y:.1f}x<extra></extra>"))
            else: fig_ebitda.add_trace(go.Scatter(x=subset['Period_Date'], y=subset['Net_Debt_to_EBITDA'], mode='lines+markers', name=reit, line=dict(color=color), customdata=subset['Hover_Label'], hovertemplate="%{customdata}<br>%{y:.1f}x<extra></extra>"))
        fig_ebitda.add_hrect(y0=0, y1=8, fillcolor="green", opacity=0.1, layer="below", line_width=0, annotation_text="Safe")
        fig_ebitda.add_hrect(y0=10, y1=15, fillcolor="red", opacity=0.1, layer="below", line_width=0, annotation_text="Risk")
        fig_ebitda = add_year_shading(fig_ebitda, chart_df)
        fig_ebitda.update_layout(title="Net Debt to EBITDA", autosize=True, height=450, xaxis=dict(showgrid=False, showticklabels=False), font=dict(size=18))
        st.plotly_chart(fig_ebitda, config={'responsive': True}, theme=None)

    if 'Debt_to_GBV' in chart_df.columns:
        fig_gbv = go.Figure()
        for reit in comparison_reits:
            subset = chart_df[chart_df['REIT'] == reit]
            actuals = subset[subset['Is_Projected'] == False]
            projections = subset[subset['Is_Projected'] == True]
            color = reit_color_map[reit]
            if freq_option == "Annual Only":
                fig_gbv.add_trace(go.Bar(x=actuals['Period_Date'], y=actuals['Debt_to_GBV'], name=reit, marker_color=color, customdata=actuals['Hover_Label'], hovertemplate="%{customdata}<br>%{y:.1f}%<extra></extra>"))
                if not projections.empty: fig_gbv.add_trace(go.Bar(x=projections['Period_Date'], y=projections['Debt_to_GBV'], marker=dict(color=color, opacity=0.6, pattern_shape='/'), name=f"{reit} (Proj)", customdata=projections['Hover_Label'], hovertemplate="%{customdata}<br>%{y:.1f}%<extra></extra>"))
            else: fig_gbv.add_trace(go.Scatter(x=subset['Period_Date'], y=subset['Debt_to_GBV'], mode='lines+markers', name=reit, line=dict(color=color), customdata=subset['Hover_Label'], hovertemplate="%{customdata}<br>%{y:.1f}%<extra></extra>"))
        fig_gbv.add_hline(y=50, line_dash="dot", line_color="orange", annotation_text="Threshold")
        fig_gbv = add_year_shading(fig_gbv, chart_df)
        fig_gbv.update_layout(title="Debt to Gross Book Value %", autosize=True, height=450, xaxis=dict(showgrid=False, showticklabels=False), font=dict(size=18))
        st.plotly_chart(fig_gbv, config={'responsive': True}, theme=None)

    if 'Interest_Coverage' in chart_df.columns:
        fig_int = go.Figure()
        for reit in comparison_reits:
            subset = chart_df[chart_df['REIT'] == reit]
            actuals = subset[subset['Is_Projected'] == False]
            projections = subset[subset['Is_Projected'] == True]
            color = reit_color_map[reit]
            if freq_option == "Annual Only":
                fig_int.add_trace(go.Bar(x=actuals['Period_Date'], y=actuals['Interest_Coverage'], name=reit, marker_color=color, customdata=actuals['Hover_Label'], hovertemplate="%{customdata}<br>%{y:.2f}x<extra></extra>"))
                if not projections.empty: fig_int.add_trace(go.Bar(x=projections['Period_Date'], y=projections['Interest_Coverage'], marker=dict(color=color, opacity=0.6, pattern_shape='/'), name=f"{reit} (Proj)", customdata=projections['Hover_Label'], hovertemplate="%{customdata}<br>%{y:.2f}x<extra></extra>"))
            else: fig_int.add_trace(go.Scatter(x=subset['Period_Date'], y=subset['Interest_Coverage'], mode='lines+markers', name=reit, line=dict(color=color), customdata=subset['Hover_Label'], hovertemplate="%{customdata}<br>%{y:.2f}x<extra></extra>"))
        fig_int.add_hline(y=2.0, line_dash="dot", line_color="green", annotation_text="Safe > 2.0x")
        fig_int = add_year_shading(fig_int, chart_df)
        fig_int.update_layout(title="Interest Coverage Ratio", autosize=True, height=450, xaxis=dict(showgrid=False, showticklabels=False), font=dict(size=18))
        st.plotly_chart(fig_int, config={'responsive': True}, theme=None)

with tab_value:
    st.subheader("Asset Value vs. Market Price")
    if 'NAV_per_Unit' in chart_df.columns and 'Reported_Cap_Rate' in chart_df.columns:
        fig_nav_cap = make_subplots(specs=[[{"secondary_y": True}]])
        is_multi = len(comparison_reits) > 1
        yaxis_title = "Normalized NAV (Start=100)" if is_multi else "NAV per Unit ($)"

        for reit in comparison_reits:
            subset = chart_df[chart_df['REIT'] == reit].copy()
            if subset.empty: continue
            subset = subset.sort_values('Period_Date')
            if is_multi:
                start_val = subset['NAV_per_Unit'].iloc[0]
                if start_val > 0:
                    subset['NAV_Indexed'] = (subset['NAV_per_Unit'] / start_val) * 100
                    nav_col = 'NAV_Indexed'
                    hover_template = "%{customdata}<br>Index: %{y:.1f}<br>Raw NAV: $%{text:.2f}<extra></extra>"
                else:
                    subset['NAV_Indexed'] = 0
                    nav_col = 'NAV_Indexed'
                    hover_template = "%{customdata}<br>Raw NAV: $%{text:.2f}<extra></extra>"
            else:
                nav_col = 'NAV_per_Unit'
                hover_template = "%{customdata}<br>NAV: $%{y:.2f}<extra></extra>"

            actuals = subset[subset['Is_Projected'] == False]
            projections = subset[subset['Is_Projected'] == True]
            color = reit_color_map[reit]

            # --- APPLY TEXT FORMATTING (Round to 2 decimals) ---
            text_actuals = actuals[nav_col].apply(lambda x: f"{x:.2f}")
            text_projs = projections[nav_col].apply(lambda x: f"{x:.2f}")

            if freq_option == "Annual Only":
                fig_nav_cap.add_trace(go.Bar(x=actuals['Period_Date'], y=actuals[nav_col], name=f"{reit} NAV", marker_color=color, customdata=actuals['Hover_Label'], text=text_actuals, textfont=dict(color='white'), hovertemplate=hover_template), secondary_y=False)
                if not projections.empty: fig_nav_cap.add_trace(go.Bar(x=projections['Period_Date'], y=projections[nav_col], name=f"{reit} NAV (Proj)", marker=dict(color=color, opacity=0.5, pattern_shape='/'), customdata=projections['Hover_Label'], text=text_projs, textfont=dict(color='white'), hovertemplate=hover_template), secondary_y=False)
            else:
                text_subset = subset[nav_col].apply(lambda x: f"{x:.2f}")
                fig_nav_cap.add_trace(go.Bar(x=subset['Period_Date'], y=subset[nav_col], name=f"{reit} NAV", marker_color=color, customdata=subset['Hover_Label'], text=text_subset, textfont=dict(color='white'), hovertemplate=hover_template), secondary_y=False)

            cap_subset = subset[subset['Reported_Cap_Rate'] > 0]
            if not cap_subset.empty:
                fig_nav_cap.add_trace(go.Scatter(x=cap_subset['Period_Date'], y=cap_subset['Reported_Cap_Rate'], name=f"{reit} Cap Rate", mode='lines+markers', line=dict(color=color, width=3, dash='dot'), marker=dict(size=10, line=dict(width=2, color='white')), connectgaps=False, customdata=cap_subset['Hover_Label'], hovertemplate="%{customdata}<br>Cap: %{y:.2f}%<extra></extra>"), secondary_y=True)

        fig_nav_cap = add_year_shading(fig_nav_cap, chart_df)
        # --- MOVED LEGEND TO RIGHT ---
        fig_nav_cap.update_layout(title="NAV per Unit & Reported Cap Rate", autosize=True, height=450, xaxis=dict(showgrid=False, showticklabels=False, showline=False), yaxis=dict(title=dict(text=yaxis_title, font=dict(color="gray")), showgrid=True), yaxis2=dict(title=dict(text="Reported Cap Rate (%)", font=dict(color="gray")), showgrid=False, range=[-1, 10], zeroline=False), font=dict(size=18), legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02))
        st.plotly_chart(fig_nav_cap, config={'responsive': True}, theme=None)
        if is_multi: st.caption("Data indexed to 100 at the start of the reporting period to compare relative growth.")

    st.markdown("---")
    st.subheader("Market Sentiment Tracker")
    sentiment_mode = st.radio("Select Metric:", ["Price / NAV Ratio", "Implied Cap Rate"], horizontal=True)
    fig_market = go.Figure()
    global_min_date = datetime.now()
    global_max_date = datetime(2000, 1, 1)

    for reit in comparison_reits:
        live_cap_df = calculate_live_implied_cap(reit, filtered_df[filtered_df['REIT']==reit])
        if live_cap_df is not None:
            min_d = live_cap_df['Date'].min()
            max_d = live_cap_df['Date'].max()
            if min_d < global_min_date: global_min_date = min_d
            if max_d > global_max_date: global_max_date = max_d
            color = reit_color_map[reit]
            if sentiment_mode == "Price / NAV Ratio":
                fig_market.add_trace(go.Scatter(x=live_cap_df['Date'], y=live_cap_df['P_NAV_Ratio'], name=f"{reit} P/NAV", mode='lines', line=dict(color=color, width=2)))
            else:
                fig_market.add_trace(go.Scatter(x=live_cap_df['Date'], y=live_cap_df['Live_Implied_Cap'], name=f"{reit} Implied Cap", mode='lines', line=dict(color=color, width=2, dash='dot')))

    fig_market = add_year_shading(fig_market, pd.DataFrame({'Date': [global_min_date, global_max_date]}))
    y_title = "Price / NAV (x)" if sentiment_mode == "Price / NAV Ratio" else "Implied Cap Rate (%)"
    if sentiment_mode == "Price / NAV Ratio":
        fig_market.add_hline(y=1.0, line_dash="solid", line_color="gray", opacity=0.5, annotation_text="NAV Par (1.0x)")

    # --- MOVED LEGEND TO RIGHT ---
    fig_market.update_layout(title=f"Historical {sentiment_mode}", autosize=True, height=450, xaxis=dict(showgrid=False, showticklabels=False, range=[global_min_date, global_max_date]), yaxis=dict(title=y_title), font=dict(size=18), legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02))
    st.plotly_chart(fig_market, config={'responsive': True}, theme=None)

    st.markdown("---")
    st.subheader("Capital Structure & Dilution")
    unit_col = 'Unit_Count_Diluted'
    if unit_col not in chart_df.columns: unit_col = 'Unit_Count_Basic'

    if unit_col in chart_df.columns:
        fig_units = go.Figure()
        is_multi = len(comparison_reits) > 1
        y_title_units = "Normalized Unit Count (Start=100)" if is_multi else "Units"

        for reit in comparison_reits:
            subset = chart_df[chart_df['REIT'] == reit].copy()
            if subset.empty: continue
            subset = subset.sort_values('Period_Date')

            if is_multi:
                start_units = subset[unit_col].iloc[0]
                if start_units > 0:
                    subset['Units_Indexed'] = (subset[unit_col] / start_units) * 100
                    plot_unit_col = 'Units_Indexed'
                    hover_template = "%{customdata}<br>Index: %{y:.1f}<br>Raw Units: %{text:,.0f}<extra></extra>"
                else:
                    subset['Units_Indexed'] = 0
                    plot_unit_col = 'Units_Indexed'
                    hover_template = "%{customdata}<br>Raw Units: %{text:,.0f}<extra></extra>"
            else:
                plot_unit_col = unit_col
                hover_template = "%{customdata}<br>Units: %{y:,.0f}<extra></extra>"

            actuals = subset[subset['Is_Projected'] == False]
            projections = subset[subset['Is_Projected'] == True]
            color = reit_color_map[reit]

            if freq_option == "Annual Only":
                fig_units.add_trace(go.Bar(x=actuals['Period_Date'], y=actuals[plot_unit_col], name=f"{reit} Units", marker_color=color, customdata=actuals['Hover_Label'], text=actuals[unit_col], hovertemplate=hover_template))
                if not projections.empty: fig_units.add_trace(go.Bar(x=projections['Period_Date'], y=projections[plot_unit_col], name=f"{reit} Units (Proj)", marker=dict(color=color, opacity=0.5, pattern_shape='/'), customdata=projections['Hover_Label'], text=projections[unit_col], hovertemplate=hover_template))
            else:
                fig_units.add_trace(go.Scatter(x=subset['Period_Date'], y=subset[plot_unit_col], mode='lines+markers', name=f"{reit} Units", line=dict(color=color, width=3), customdata=subset['Hover_Label'], text=subset[unit_col], hovertemplate=hover_template))

        fig_units = add_year_shading(fig_units, chart_df)
        # --- MOVED LEGEND TO RIGHT ---
        fig_units.update_layout(title="Total Units Outstanding (Diluted)", autosize=True, height=400, xaxis=dict(showgrid=False, showticklabels=False), yaxis=dict(title=y_title_units, showgrid=True), font=dict(size=18), legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02))
        st.plotly_chart(fig_units, config={'responsive': True}, theme=None)
        st.markdown("**Note:** Tracks shareholder dilution. Rising bars indicate equity issuance (dilutive); falling bars indicate buybacks (accretive).")
        if is_multi: st.caption("Data indexed to 100 at the start of the reporting period to compare relative dilution.")

# --- TAB 4: FORENSIC AUDIT ---
with tab_source:
    st.subheader("📂 Forensic Report Viewer")

    col_sel, col_empty = st.columns([1, 3])
    with col_sel:
        if len(comparison_reits) == 1:
            periods = sorted(filtered_df['Period_Clean'].unique().tolist(), reverse=True)
            selected_period = st.selectbox("Select Reporting Period to Audit:", periods)
            audit_row = filtered_df[filtered_df['Period_Clean'] == selected_period].iloc[0]
        else:
            st.warning("Please select a Single REIT to view detailed Forensic Reports.")
            st.stop()

    def render_section_table(title, row_data, metric_configs):
        table_rows = []
        for label, key_prefix, is_dual in metric_configs:
            raw_3m = row_data.get(f"{key_prefix}_3M")
            raw_12m = row_data.get(f"{key_prefix}_12M")
            raw_generic = row_data.get(key_prefix)
            factor = row_data.get('Split_Factor', 1.0)
            is_per_unit = 'per_Unit' in key_prefix or 'Share_Price' in key_prefix or 'NAV_per_Unit' in key_prefix
            is_count = 'Unit_Count' in key_prefix
            source = row_data.get(f"{key_prefix}_Source", "")
            row_dict = {"Metric": label}

            def safe_str(val): return f"{val:,.2f}" if pd.notnull(val) else "-"

            if is_dual:
                if factor != 1.0 and (is_per_unit or is_count):
                    if is_count:
                        val_3m_display = safe_str(raw_3m * factor)
                        val_12m_display = safe_str(raw_12m * factor)
                        row_dict["Adjusted (3M)"] = f"{raw_3m:,.0f}" if pd.notnull(raw_3m) else "-"
                    else:
                        val_3m_display = safe_str(raw_3m / factor)
                        val_12m_display = safe_str(raw_12m / factor)
                        row_dict["Adjusted (3M)"] = f"{raw_3m:,.2f}" if pd.notnull(raw_3m) else "-"
                else:
                    val_3m_display = safe_str(raw_3m)
                    val_12m_display = safe_str(raw_12m)
                row_dict["Value (3M/Qtr)"] = val_3m_display
                row_dict["Value (12M/Ann)"] = val_12m_display
            else:
                if factor != 1.0 and (is_per_unit or is_count):
                    if is_count:
                        val_display = safe_str(raw_generic * factor)
                        row_dict["Adjusted Value"] = f"{raw_generic:,.0f}" if pd.notnull(raw_generic) else "-"
                    else:
                        val_display = safe_str(raw_generic / factor)
                        row_dict["Adjusted Value"] = f"{raw_generic:,.2f}" if pd.notnull(raw_generic) else "-"
                else:
                    val_display = safe_str(raw_generic)
                row_dict["Value"] = val_display

            row_dict["Source / Citation"] = str(source) if source else ""
            table_rows.append(row_dict)
        return pd.DataFrame(table_rows)

    st.markdown("#### 1. EARNINGS & VALUATION")
    df_earn = render_section_table("Earnings", audit_row, [
        ("Revenue", "Revenue", True),
        ("Adj. EBITDA", "Adj_EBITDA", True),
        ("FFO", "FFO_Total", True),
        ("AFFO", "AFFO_Total", True),
        ("FFO per Unit", "FFO_per_Unit", True),
        ("AFFO per Unit", "AFFO_per_Unit", True),
        ("Payout Ratio %", "Payout_Ratio", True),
        ("P/FFO (x)", "P_FFO", True),
        ("P/AFFO (x)", "P_AFFO", True),
    ])
    st.table(df_earn)

    st.markdown("#### 2. DEBT")
    df_debt = render_section_table("Debt", audit_row, [
        ("Net Debt ($)", "Net_Debt", False),
        ("Debt-to-GBV (%)", "Debt_to_GBV", False),
        ("Net Debt/EBITDA (x)", "Net_Debt_to_EBITDA", False),
        ("Interest Cov. (x)", "Interest_Coverage", False),
    ])
    st.table(df_debt)

    st.markdown("#### 3. BOOK VALUE & NAV")
    df_nav = render_section_table("NAV", audit_row, [
        ("Equity ($)", "Equity", False),
        ("Unit Count", "Unit_Count_Basic", False),
        ("Diluted Units", "Unit_Count_Diluted", False),
        ("Market Cap ($)", "Market_Cap", False),
        ("Enterprise Val ($)", "Enterprise_Value", False),
        ("NAV per Unit ($)", "NAV_per_Unit", False),
        ("P/NAV (x)", "P_NAV", False),
        ("Reported Cap (%)", "Reported_Cap_Rate", False),
        ("Implied Cap (%)", "Implied_Cap_Rate", False),
    ])
    st.table(df_nav)

    st.markdown("#### 4. OPERATIONAL HEALTH")
    df_ops = render_section_table("Ops", audit_row, [
        ("Occupancy (%)", "Occupancy", False),
        ("WALT (Years)", "WALT", False),
        ("SPNOI Growth (%)", "SPNOI_Growth", True),
    ])
    st.table(df_ops)

    st.markdown("---")
    raw_link = audit_row.get('Link_Report', 'N/A')
    display_link = raw_link.replace('s3://reit-data/', '') if raw_link else "N/A"
    st.markdown(f"**Data Origin:** `{display_link}`")
