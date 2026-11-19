import sqlite3
import json
import os
from sklearn.datasets import fetch_california_housing

os.makedirs("data", exist_ok=True)

# 1. Load real California Housing dataset (20,640 rows) offline
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# 2. Clean column names for SQL
df.columns = [col.strip().lower().replace(" ", "_").replace("-", "_") for col in df.columns]

# 3. Save into SQLite database
conn = sqlite3.connect("data/company_data.db")
df.to_sql("housing", conn, if_exists="replace", index=False)
conn.close()

# 4. Create Schema RAG rules
columns_list = ", ".join(df.columns)
schema_docs = [
    {
        "table": "housing",
        "description": f"Real housing market dataset with {len(df)} rows. Columns: {columns_list}. medinc is median income, houseage is median house age, averooms is average rooms, population is block population, medhouseval is median house value (in $100,000s)."
    },
    {
        "table": "business_rules",
        "description": "Always use the table name 'housing' and exact lowercase column names. Use AVG() for averages, MAX() for highest, and MIN() for lowest."
    }
]

with open("data/schema_docs.json", "w") as f:
    json.dump(schema_docs, f, indent=4)

print(f"Success! Loaded Housing dataset ({len(df)} rows) into data/company_data.db")
print("Columns:", list(df.columns))
