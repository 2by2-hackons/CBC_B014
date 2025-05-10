import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Load Excel
file = 'Infosys.xlsx'
xls = pd.ExcelFile(file)

# Keywords for sections
keywords = {
    'pnl': ['Sales', 'Expenses', 'Operating Profit', 'Net profit'],
    'bs': ['Equity Share Capital', 'Reserves', 'Borrowings', 'Fixed Assets'],
    'cf': ['Cash from Operating Activity', 'Cash from Investing Activity', 'Cash from Financing Activity']
}

# Function to find data start row
def find_data_start_row(df, keys):
    for idx, row in df.iterrows():
        if any(any(k.lower() in str(cell).lower() for k in keys) for cell in row):
            return idx
    return None

# Load and clean sheet
def process_sheet(sheet_name, keys):
    raw_df = xls.parse(sheet_name, header=None)
    start_row = find_data_start_row(raw_df, keys)
    if start_row is None:
        return pd.DataFrame()
    df = xls.parse(sheet_name, header=start_row)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.rename(columns={df.columns[0]: 'Metric'}).dropna(subset=['Metric'])
    df['Metric'] = df['Metric'].astype(str).str.lower().str.strip()
    df = df[df['Metric'].apply(lambda x: any(k.lower() in x for k in keys))]
    df = df.set_index('Metric')
    df = df.T
    df.columns = [col.strip() for col in df.columns]
    df = df.applymap(lambda x: re.sub(r'[^\d.-]', '', str(x)))
    df = df.apply(pd.to_numeric, errors='coerce')
    df.index.name = 'Year'
    return df

# Process all three sheets
pnl_df = process_sheet('Profit & Loss', keywords['pnl'])
bs_df = process_sheet('Balance Sheet', keywords['bs'])
cf_df = process_sheet('Cash Flow', keywords['cf'])

# Combine and summarize
combined = pd.concat([pnl_df, bs_df, cf_df], axis=1)

print("Combined Financial Data (Preview):")
print(combined.head())

# Save combined data
combined.to_csv("financial_summary.csv")

# Plot insights
if not pnl_df.empty:
    pnl_df.plot(kind='bar', figsize=(10, 6), title="P&L Trends")
    plt.tight_layout()
    plt.savefig("pnl_summary.png")
    plt.show()

if not cf_df.empty:
    cf_df.plot(kind='bar', figsize=(10, 6), title="Cash Flow Trends")
    plt.tight_layout()
    plt.savefig("cf_summary.png")
    plt.show()

if not bs_df.empty:
    bs_df.plot(kind='bar', figsize=(10, 6), title="Balance Sheet Trends")
    plt.tight_layout()
    plt.savefig("bs_summary.png")
    plt.show()

# Optional: Generate basic insights
def generate_basic_insights(df, name):
    if df.empty:
        return f"No {name} data available."
    insights = []
    numeric_cols = df.columns[df.dtypes != 'object']
    for col in numeric_cols:
        latest = df[col].dropna().iloc[-1] if not df[col].dropna().empty else None
        trend = df[col].dropna().pct_change().mean() * 100 if df[col].dropna().shape[0] > 1 else None
        if latest is not None:
            insights.append(f"{col.title()}: Latest value is {latest:,.2f}.")
        if trend is not None:
            trend_desc = "increasing" if trend > 0 else "decreasing"
            insights.append(f"{col.title()} shows a {trend_desc} trend of {trend:.2f}%.")
    return "\n".join(insights)

print("\nProfit & Loss Insights:\n", generate_basic_insights(pnl_df, "P&L"))
print("\n Balance Sheet Insights:\n", generate_basic_insights(bs_df, "Balance Sheet"))
print("\n Cash Flow Insights:\n", generate_basic_insights(cf_df, "Cash Flow"))