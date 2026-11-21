"""
Blood pressure dataset - data cleaning with NumPy + pandas
Input : Blood_Pressure_Dirty_Data.xlsx
Output: Blood_Pressure_Clean_Data.xlsx
"""
import json, re
import numpy as np
import pandas as pd

INPUT_FILE = "Blood_Pressure_Dirty_Data.xlsx"
OUTPUT_FILE = "Blood_Pressure_Clean_Data.xlsx"

df = pd.read_excel(INPUT_FILE, dtype=object)
print("Raw shape:", df.shape)
log = {"raw_rows": len(df)}

# 1. trim spaces, unify missing markers
MISSING = {"", "n/a", "na", "null", "none", "-", "nan"}
def to_nan(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = re.sub(r"\s+", " ", str(x)).strip()
    return np.nan if s.lower() in MISSING else s
df = df.map(to_nan)               # applymap() on pandas < 2.1

# 2. duplicates
df = df.drop_duplicates()
df = df.drop_duplicates(subset="Person_ID", keep="first")
log["duplicates_removed"] = log["raw_rows"] - len(df)

# 3. text columns
df["Name"] = df["Name"].str.title().fillna("Unknown")

gender_map = {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}
df["Gender"] = df["Gender"].str.lower().map(gender_map)

city_map = {"ghaziabad": "Ghaziabad", "ghazibad": "Ghaziabad", "gzb": "Ghaziabad",
            "noida": "Noida", "delhi": "Delhi", "new delhi": "Delhi",
            "gurgaon": "Gurgaon", "gurugram": "Gurgaon",
            "faridabad": "Faridabad", "faridabd": "Faridabad"}
df["City"] = df["City"].str.lower().map(city_map).fillna("Unknown")

diet_map = {"healthy": "Healthy", "good": "Healthy",
            "average": "Average", "avg": "Average", "moderate": "Average",
            "junk": "Junk", "poor": "Junk", "junk food": "Junk"}
df["Diet_Quality"] = df["Diet_Quality"].str.lower().map(diet_map)

# 4. dates in mixed formats
def parse_date(s):
    if pd.isna(s):
        return pd.NaT
    s = str(s).split(" 00:00")[0]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d.%m.%Y", "%d %b %Y", "%m/%d/%Y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except ValueError:
            continue
    return pd.NaT
df["Checkup_Date"] = df["Checkup_Date"].map(parse_date)

# 5. numeric columns (units, words, comma decimals)
AGE_WORDS = {"twenty five": 25, "thirty": 30, "forty": 40, "fifty": 50}

def to_number(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower().strip()
    if s in AGE_WORDS:
        return float(AGE_WORDS[s])
    if s in {"none", "nil", "zero"}:
        return 0.0
    s = s.replace(",", ".")
    m = re.search(r"-?\d+(\.\d+)?", s)
    if not m:
        return np.nan
    v = float(m.group())
    if "lb" in s:                  # pounds -> kg
        v = v / 2.20462
    elif "min" in s:               # minutes -> hours
        v = v / 60
    return v

for col in ["Age", "Weight_kg", "Exercise_Hours_Week", "Systolic_BP", "Diastolic_BP"]:
    df[col] = df[col].map(to_number)

# 6. impossible values -> NaN
df["Age"] = df["Age"].where(df["Age"].between(18, 100))
df["Weight_kg"] = df["Weight_kg"].where(df["Weight_kg"].between(30, 250))
df["Exercise_Hours_Week"] = df["Exercise_Hours_Week"].where(df["Exercise_Hours_Week"].between(0, 40))
df["Systolic_BP"] = df["Systolic_BP"].where(df["Systolic_BP"].between(70, 250))
df["Diastolic_BP"] = df["Diastolic_BP"].where(df["Diastolic_BP"].between(40, 150))

# 7. swapped systolic/diastolic (systolic must be higher)
swap = df["Systolic_BP"] < df["Diastolic_BP"]
df.loc[swap, ["Systolic_BP", "Diastolic_BP"]] = df.loc[swap, ["Diastolic_BP", "Systolic_BP"]].values
log["bp_swaps_fixed"] = int(swap.sum())

# 8. missing values
before = len(df)
df = df.dropna(subset=["Systolic_BP"])        # target must not be invented
log["rows_dropped_no_target"] = before - len(df)

for col in ["Age", "Weight_kg", "Exercise_Hours_Week", "Diastolic_BP"]:
    log[f"missing_{col}_filled"] = int(df[col].isna().sum())
log["missing_Gender_filled"] = int(df["Gender"].isna().sum())
log["missing_Diet_filled"] = int(df["Diet_Quality"].isna().sum())

df["Weight_kg"] = df["Weight_kg"].fillna(df.groupby("Gender")["Weight_kg"].transform("median"))
for col in ["Age", "Weight_kg", "Exercise_Hours_Week", "Diastolic_BP"]:
    df[col] = df[col].fillna(df[col].median())
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Diet_Quality"] = df["Diet_Quality"].fillna(df["Diet_Quality"].mode()[0])

# 9. final types, extra feature, order
df["Age"] = df["Age"].round().astype(int)
df["Weight_kg"] = df["Weight_kg"].round(1)
df["Exercise_Hours_Week"] = df["Exercise_Hours_Week"].round(1)
df["Systolic_BP"] = df["Systolic_BP"].round().astype(int)
df["Diastolic_BP"] = df["Diastolic_BP"].round().astype(int)
df["Diet_Quality"] = pd.Categorical(df["Diet_Quality"], ["Healthy", "Average", "Junk"], ordered=True)
df["BP_Category"] = pd.cut(df["Systolic_BP"], [0, 119, 129, 139, 300],
                           labels=["Normal", "Elevated", "High (Stage 1)", "High (Stage 2)"])
df = df.sort_values(["Checkup_Date", "Person_ID"]).reset_index(drop=True)
df["Checkup_Date"] = df["Checkup_Date"].dt.date
log["clean_rows"] = len(df)

df.to_excel(OUTPUT_FILE, index=False)
json.dump(log, open("cleaning_log.json", "w"))
print("Clean shape:", df.shape)
print(log)
print(df.describe(include="all").T)
