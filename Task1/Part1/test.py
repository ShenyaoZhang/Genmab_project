import requests
import pandas as pd
import json

url = "https://api.fda.gov/drug/event.json"

params = {
    "search": 'patient.drug.medicinalproduct:"epcoritamab" AND patient.reaction.reactionmeddrapt:"cytokine release syndrome"',
    "limit": 1000
}

r = requests.get(url, params=params).json()

# Save raw JSON response locally
with open("fda_drug_events.json", "w") as f:
    json.dump(r, f, indent=2)
print("Raw JSON saved to fda_drug_events.json")

# # Process and save as CSV
# df = pd.json_normalize(r["results"])
# df.to_csv("fda_drug_events.csv", index=False)
# print(f"DataFrame saved to fda_drug_events.csv ({len(df)} rows)")

# df.head()
