import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project folder and output folder
folder = Path(__file__).resolve().parent
output = folder / "environmental_graphs"
output.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def get_dataset_type(filename):
    name = filename.lower()

    if "air" in name:
        return "Air Quality"
    elif "water" in name:
        return "Water Quality"
    elif "energy" in name:
        return "Energy"
    elif "weather" in name:
        return "Weather"
    elif "waste" in name:
        return "Waste Management"

    return "Other"


def find_column(df, keywords):
    """Find a column using likely column-name keywords."""
    for col in df.columns:
        normalized = (
            str(col).lower()
            .replace("_", " ")
            .replace("-", " ")
        )
        if any(word in normalized for word in keywords):
            return col
    return None


def save_bar_chart(data, x, y, title, filename):
    plt.figure(figsize=(9, 5))
    plt.bar(data[x].astype(str), data[y])
    plt.title(title)
    plt.xlabel(str(x))
    plt.ylabel(str(y))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output / filename, dpi=150)
    plt.close()


def analyze_dataset(file):
    df = pd.read_excel(file)
    df = df.copy()
    df.columns = df.columns.map(str).str.strip()
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    kind = get_dataset_type(file.name)

    print(f"\n{'=' * 50}")
    print(f"Dataset: {file.name}")
    print(f"Type: {kind}")
    print(f"Rows and columns: {df.shape}")
    print("Columns:", df.columns.tolist())

    # Convert likely measurement columns to numeric
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0:
                if converted.notna().sum() >= df[col].notna().sum() * 0.8:
                    df[col] = converted

    numeric = df.select_dtypes(include="number")

    # Save descriptive statistics
    if not numeric.empty:
        numeric.describe().round(2).to_csv(
            output / f"{file.stem}_statistics.csv"
        )

        # Histograms for measurements
        for col in numeric.columns:
            values = numeric[col].dropna()

            if not values.empty:
                plt.figure(figsize=(7, 4))
                plt.hist(values, bins=15, edgecolor="black")
                plt.title(f"{kind}: {col} Distribution")
                plt.xlabel(str(col))
                plt.ylabel("Frequency")
                plt.tight_layout()
                plt.savefig(
                    output / f"{file.stem}_{col}_histogram.png",
                    dpi=150
                )
                plt.close()

        # Correlation heatmap
        if numeric.shape[1] >= 2:
            plt.figure(figsize=(9, 6))
            sns.heatmap(
                numeric.corr(),
                annot=True,
                cmap="coolwarm",
                fmt=".2f"
            )
            plt.title(f"{kind}: Correlation Heatmap")
            plt.tight_layout()
            plt.savefig(
                output / f"{file.stem}_correlation.png",
                dpi=150
            )
            plt.close()

    # Find common location, date and measurement columns
    city_col = find_column(
        df, ["city", "district", "location", "station", "region"]
    )
    date_col = find_column(df, ["date", "datetime", "timestamp"])
    aqi_col = find_column(df, ["aqi", "air quality index"])
    ph_col = find_column(df, ["ph"])
    energy_col = find_column(
        df, ["energy consumption", "energy generation",
             "electricity", "energy", "power"]
    )
    temperature_col = find_column(
        df, ["temperature", "temp"]
    )
    rainfall_col = find_column(
        df, ["rainfall", "precipitation"]
    )
    waste_col = find_column(
        df, ["waste generated", "waste generation",
             "waste collected", "recycled", "waste"]
    )

    # Air Quality: average AQI by location
    if kind == "Air Quality" and aqi_col and city_col:
        temp = df[[city_col, aqi_col]].copy()
        temp[aqi_col] = pd.to_numeric(temp[aqi_col], errors="coerce")
        temp = temp.dropna()

        if not temp.empty:
            summary = temp.groupby(city_col)[aqi_col].mean().reset_index()
            save_bar_chart(
                summary, city_col, aqi_col,
                "Average AQI by Location",
                f"{file.stem}_average_AQI.png"
            )

    # Water Quality: average pH by location
    if kind == "Water Quality" and ph_col and city_col:
        temp = df[[city_col, ph_col]].copy()
        temp[ph_col] = pd.to_numeric(temp[ph_col], errors="coerce")
        temp = temp.dropna()

        if not temp.empty:
            summary = temp.groupby(city_col)[ph_col].mean().reset_index()
            save_bar_chart(
                summary, city_col, ph_col,
                "Average pH by Location",
                f"{file.stem}_average_pH.png"
            )

    # Energy: energy measurements by location
    if kind == "Energy" and energy_col and city_col:
        temp = df[[city_col, energy_col]].copy()
        temp[energy_col] = pd.to_numeric(temp[energy_col], errors="coerce")
        temp = temp.dropna()

        if not temp.empty:
            summary = temp.groupby(city_col)[energy_col].mean().reset_index()
            save_bar_chart(
                summary, city_col, energy_col,
                "Average Energy Measurement by Location",
                f"{file.stem}_energy_by_location.png"
            )

    # Weather: temperature trend over time
    if kind == "Weather" and temperature_col and date_col:
        temp = df[[date_col, temperature_col]].copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        temp[temperature_col] = pd.to_numeric(temp[temperature_col], errors="coerce")
        temp = temp.dropna().sort_values(date_col)

        if not temp.empty:
            plt.figure(figsize=(9, 5))
            plt.plot(
                temp[date_col], temp[temperature_col],
                marker="o"
            )
            plt.title("Temperature Over Time")
            plt.xlabel("Date")
            plt.ylabel("Temperature")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(
                output / f"{file.stem}_temperature_trend.png",
                dpi=150
            )
            plt.close()

    # Weather: rainfall by date
    if kind == "Weather" and rainfall_col and date_col:
        temp = df[[date_col, rainfall_col]].copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        temp[rainfall_col] = pd.to_numeric(temp[rainfall_col], errors="coerce")
        temp = temp.dropna()

        if not temp.empty:
            summary = (
                temp.groupby(temp[date_col].dt.to_period("M"))[rainfall_col]
                .sum()
                .reset_index()
            )
            summary[date_col] = summary[date_col].astype(str)

            save_bar_chart(
                summary, date_col, rainfall_col,
                "Monthly Rainfall",
                f"{file.stem}_monthly_rainfall.png"
            )

    # Waste: average waste measurement by location
    if kind == "Waste Management" and waste_col and city_col:
        temp = df[[city_col, waste_col]].copy()
        temp[waste_col] = pd.to_numeric(temp[waste_col], errors="coerce")
        temp = temp.dropna()

        if not temp.empty:
            summary = temp.groupby(city_col)[waste_col].mean().reset_index()
            save_bar_chart(
                summary, city_col, waste_col,
                "Average Waste Measurement by Location",
                f"{file.stem}_waste_by_location.png"
            )

    print("Analysis completed for:", file.name)


# Process all Excel files in the project folder
excel_files = [
    file for file in folder.glob("*.xlsx")
    if "ANALYSIS_RESULTS" not in file.name.upper()
]

if not excel_files:
    print("No Excel files found in the project folder.")
else:
    for file in excel_files:
        try:
            analyze_dataset(file)
        except Exception as error:
            print(f"Could not analyze {file.name}: {error}")

print("\nAll datasets processed.")
print("Graphs and statistics saved in:", output.resolve())