# : Student Marks Analysis using Pandas
import pandas as pd

data = {"Student": ["A", "B", "C", "D", "E"],
        "Marks": [65, 72, 81, 56, 90]}

df = pd.DataFrame(data)
print("Student Data:\n", df)
print("Average Marks:", df["Marks"].mean())
print("Highest Marks:", df["Marks"].max())
