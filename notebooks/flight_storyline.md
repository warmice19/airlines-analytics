# Notebook: flight_00.ipynb

## High-level purpose

Data preprocessing: load raw flight datasets, clean and transform them, engineer features, and persist a cleaned dataset for analysis.

## Step-by-step storyline

### Step 1 — (code)

**What the cell does (one-line):** import polars as pl

**Why this step:** Set up analysis environment: import libraries used for data manipulation and visualization.

**Outcome / observed output:** Python libraries are available to run subsequent data processing and plotting.

**Inference / what this enables for the story:** This establishes standard tooling for the notebook.

**Example snippet:**

```python
import polars as pl
import numpy as np
```

### Step 2 — (code)

**What the cell does (one-line):** airports = pl.read_csv("./../data/airports.csv")

**Why this step:** Load raw data into memory to begin preprocessing and analysis.

**Outcome / observed output:** Dataset is loaded into a DataFrame from ./../data/airports.csv. Observed shape: (322, 7). Observed output: Airports: (322, 7)
Airlines: (14, 2)
Flights: (5819079, 31)

**Inference / what this enables for the story:** With the data in-memory we can inspect structure, missing values, and prepare features.

**Example snippet:**

```python
airports = pl.read_csv("./../data/airports.csv")
airlines = pl.read_csv("./../data/airlines.csv")
flights = pl.read_csv("./../data/flights.csv")

print("Airports:", airports.shape)
print("Airlines:", airlines.shape)
print("Flights:", flights.shape)
```

### Step 3 — (code)

**What the cell does (one-line):** #--renaming and standardizing columns

**Why this step:** Perform a technical data transformation or check.

**Outcome / observed output:** Cell produces a transformation, check, or intermediate object.

**Inference / what this enables for the story:** Moves preprocessing forward toward analysis-ready data.

**Example snippet:**

```python
#--renaming and standardizing columns

airports = airports.rename({
    "IATA_CODE": "AIRPORT_CODE",
    "AIRPORT": "AIRPORT_NAME",
    "CITY": "CITY",
    "STATE": "STATE",
    "COUNTRY": "COUNTRY",
    "LATITUDE": "LAT",
    "LONGITUDE": "LON"
})

airlines = airlines.rename({
    "IATA_CODE": "AIRLINE_CODE",
    "AIRLINE": "AIRLINE_NAME"
})

flights = flights.rename({
    "YEAR": "YEAR",
    "MONTH": "MONTH",
    "DAY": "DAY",
    "DAY_OF_WEEK": "DAY_OF_WEEK",
    "AIRLINE": "AIRLINE_CODE",
    "FLIGHT_NUMBER": "FLIGHT_NUM",
    "TAIL_NUMBER": "TAIL_NUM",
    "ORIGIN_AIRPORT": "ORIGIN_CODE",
    "DESTINATION_AIRPORT": "DEST_CODE",
    "SCHEDULED_DEPARTURE": "SCHED_DEP",
    "DEPARTURE_TIME": "DEP_TIME",
    "DEPARTURE_DELAY": "DEP_DELAY",
    "TAXI_OUT": "TAXI_OUT",
    "WHEELS_OFF": "WHEELS_OFF",
    "SCHEDULED_TIME": "SCHED_TIME",
    "ELAPSED_TIME": "ELAPSED_TIME",
    "AIR_TIME": "AIR_TIME",
    "DISTANCE": "DISTANCE",
    "WHEELS_ON": "WHEELS_ON",
    "TAXI_IN": "TAXI_IN",
    "
```

### Step 4 — (code)

**What the cell does (one-line):** # -- converting delay columns to float and replacing null values

**Why this step:** Perform a technical data transformation or check.

**Outcome / observed output:** Cell produces a transformation, check, or intermediate object. Observed output: Cleaned flights: (5819079, 35)

**Inference / what this enables for the story:** Moves preprocessing forward toward analysis-ready data.

**Example snippet:**

```python
# -- converting delay columns to float and replacing null values
delay_cols = ["DEP_DELAY", "ARR_DELAY", "AIR_SYS_DELAY", "SEC_DELAY",
              "AIRLINE_DELAY", "LATE_AC_DELAY", "WEATHER_DELAY"]

flights = flights.with_columns([
    flights[col].fill_null(0).cast(pl.Float64) for col in delay_cols
])


# --- Convert Diverted & Cancelled to Boolean
flights = flights.with_columns([
    flights["DIVERTED"].cast(pl.Boolean),
    flights["CANCELLED"].cast(pl.Boolean)
])


# --- Clean cancellation reason ("null" → None)
flights = flights.with_columns(
    pl.when((flights["CANCEL_REASON"] == "null") | (flights["CANCEL_REASON"] == ""))
      .then(None)
      .otherwise(flights["CANCEL_REASON"])
      .alias("CANCEL_REASON")
)

# --- Parse scheduled/actual times (HHMM → datetime)
def parse_time(df, col):
    return (
        df[col]
        .cast(pl.Int32)
        .fill_null(-1)
        .cast(str)
        .str.zfill(4)
        .str.strptime(pl.Datetime, "%H%M", strict=False)
    )

flight
```

### Step 5 — (code)

**What the cell does (one-line):** flights.head()

**Why this step:** Preview rows of the data to inspect columns, sample values and obvious issues.

**Outcome / observed output:** Displays the first few rows so column names and sample records are visible. Output sample: shape: (5, 35) Observed output: shape: (5, 35)
┌──────┬───────┬─────┬─────────────┬───┬──────────────┬──────────────┬──────────────┬──────────────┐
│ YEAR ┆ MONTH ┆ DAY ┆ DAY_OF_WEEK ┆ … ┆ SCHED_DEP_TI ┆ DEP_TIME_CLE ┆ SCHED_ARR_TI ┆ ARR_TIME_CLE │
│ ---  ┆ ---   ┆ --- ┆ ---         ┆   ┆ ME           ┆ AN           ┆ ME           ┆ AN           │
│ i64  ┆ i64   ┆ i64 ┆ i64         ┆   ┆ ---          ┆ ---          ┆ ---          ┆ ---          │
│      ┆       ┆     ┆             ┆   ┆ datetime[μs] ┆ datetime[μs] ┆ datetime[μ

**Inference / what this enables for the story:** Quickly helps validate that data loaded correctly and reveals immediate cleaning needs.

**Example snippet:**

```python
flights.head()
```

### Step 6 — (code)

**What the cell does (one-line):** # --- merge datasets ---

**Why this step:** Combine related tables (airlines, airports, flights) for richer context (names, codes).

**Outcome / observed output:** A wider table that includes both flight metrics and human-readable metadata (e.g., airport names). Observed output: Final flights dataset: (5819079, 42)

**Inference / what this enables for the story:** Joined tables make plots and interpretations clearer (showing names instead of codes).

**Example snippet:**

```python
# --- merge datasets ---

flights = flights.join(airlines, on="AIRLINE_CODE", how="left")

flights = flights.join(
    airports.select(["AIRPORT_CODE", "AIRPORT_NAME", "CITY", "STATE"]),
    left_on="ORIGIN_CODE", right_on="AIRPORT_CODE", how="left"
).rename({
    "AIRPORT_NAME": "ORIGIN_AIRPORT_NAME",
    "CITY": "ORIGIN_CITY",
    "STATE": "ORIGIN_STATE"
})


flights = flights.join(
    airports.select(["AIRPORT_CODE", "AIRPORT_NAME", "CITY", "STATE"]),
    left_on="DEST_CODE", right_on="AIRPORT_CODE", how="left"
).rename({
    "AIRPORT_NAME": "DEST_AIRPORT_NAME",
    "CITY": "DEST_CITY",
    "STATE": "DEST_STATE"
})


print("Final flights dataset:", flights.shape)
```

### Step 7 — (code)

**What the cell does (one-line):** flight_data = flights

**Why this step:** Perform a technical data transformation or check.

**Outcome / observed output:** Cell produces a transformation, check, or intermediate object.

**Inference / what this enables for the story:** Moves preprocessing forward toward analysis-ready data.

**Example snippet:**

```python
flight_data = flights
```

### Step 8 — (code)

**What the cell does (one-line):** # -- data summary --

**Why this step:** Perform a technical data transformation or check.

**Outcome / observed output:** Cell produces a transformation, check, or intermediate object. Observed output: shape: (1, 4)
┌─────────────────┬────────────────────────┬─────────────────────────────┬───────────────┐
│ Unique Airlines ┆ Unique Origin Airports ┆ Unique Destination Airports ┆ Total Flights │
│ ---             ┆ ---                    ┆ ---                         ┆ ---           │
│ u32             ┆ u32                    ┆ u32                         ┆ u32           │
╞═════════════════╪════════════════════════╪═════════════════════════════╪═══════════════╡
│ 14              ┆ 628        

**Inference / what this enables for the story:** Moves preprocessing forward toward analysis-ready data.

**Example snippet:**

```python
# -- data summary -- 

print(
   flight_data.select([
    pl.col("AIRLINE_NAME").n_unique().alias("Unique Airlines"),
    pl.col("ORIGIN_CODE").n_unique().alias("Unique Origin Airports"),
    pl.col("DEST_CODE").n_unique().alias("Unique Destination Airports"),
    pl.col("FLIGHT_NUM").count().alias("Total Flights")
])
)
```

### Step 9 — (code)

**What the cell does (one-line):** flight_data.write_csv("./../data/cleaned_flight_data.csv")

**Why this step:** Perform a technical data transformation or check.

**Outcome / observed output:** Cell produces a transformation, check, or intermediate object.

**Inference / what this enables for the story:** Moves preprocessing forward toward analysis-ready data.

**Example snippet:**

```python
flight_data.write_csv("./../data/cleaned_flight_data.csv")
```

### Step 10 — (code)

**What the cell does (one-line):** (code cell)

**Why this step:** Perform a technical data transformation or check.

**Outcome / observed output:** Cell produces a transformation, check, or intermediate object.

**Inference / what this enables for the story:** Moves preprocessing forward toward analysis-ready data.


---

# Notebook: flight_01.ipynb

## High-level purpose

Visualizations & analysis: load the cleaned data, compute KPIs and build plots to surface insights about delays and flight patterns.

## Step-by-step storyline

### Step 1 — (code)

**What the cell does (one-line):** import polars as pl

**Why this step:** Set up analysis environment: import libraries used for data manipulation and visualization.

**Outcome / observed output:** Python libraries are available to run subsequent data processing and plotting.

**Inference / what this enables for the story:** This establishes standard tooling for the notebook.

**Example snippet:**

```python
import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

### Step 2 — (code)

**What the cell does (one-line):** df = pl.read_csv("./../data/cleaned_flight_data.csv",try_parse_dates=True)

**Why this step:** Load raw data into memory to begin preprocessing and analysis.

**Outcome / observed output:** Dataset is loaded into a DataFrame from ./../data/cleaned_flight_data.csv.

**Inference / what this enables for the story:** With the data in-memory we can inspect structure, missing values, and prepare features.

**Example snippet:**

```python
df = pl.read_csv("./../data/cleaned_flight_data.csv",try_parse_dates=True)
```

### Step 3 — (code)

**What the cell does (one-line):** sns.set_theme(style="whitegrid")

**Why this step:** Visualize distributions, trends and relationships to support insights-driven storytelling.

**Outcome / observed output:** Plots (histograms, bar charts, heatmaps, time-series) illustrating the metric of interest.

**Inference / what this enables for the story:** Visual evidence for claims — e.g., peak delay hours or airports with chronic delays.

**Example snippet:**

```python
sns.set_theme(style="whitegrid")
```

### Step 4 — (markdown)

**What the cell does (one-line):** ### 1. AIRLINE PERFORMANCE ANALYSIS

**Why this step:** Explains intent or documents the next step.

**Outcome / observed output:** No direct code output; provides context for following cells.

**Inference / what this enables for the story:** Helps readers understand why the next code step exists.

**Example snippet:**

```python
### 1. AIRLINE PERFORMANCE ANALYSIS
```

### Step 5 — (code)

**What the cell does (one-line):** kpi_basic = df.select([

**Why this step:** Perform a technical data transformation or check.

**Outcome / observed output:** Cell produces a transformation, check, or intermediate object. Observed output: shape: (1, 4)
┌─────────────────┬────────────────────────┬─────────────────────────────┬───────────────┐
│ Unique Airlines ┆ Unique Origin Airports ┆ Unique Destination Airports ┆ Total Flights │
│ ---             ┆ ---                    ┆ ---                         ┆ ---           │
│ u32             ┆ u32                    ┆ u32                         ┆ u32           │
╞═════════════════╪════════════════════════╪═════════════════════════════╪═══════════════╡
│ 14              ┆ 628        

**Inference / what this enables for the story:** Moves preprocessing forward toward analysis-ready data.

**Example snippet:**

```python
kpi_basic = df.select([
    pl.col("AIRLINE_NAME").n_unique().alias("Unique Airlines"),
    pl.col("ORIGIN_CODE").n_unique().alias("Unique Origin Airports"),
    pl.col("DEST_CODE").n_unique().alias("Unique Destination Airports"),
    pl.col("FLIGHT_NUM").count().alias("Total Flights")
])
kpi_basic
```

### Step 6 — (code)

**What the cell does (one-line):** # -- flights per airline --

**Why this step:** Set up analysis environment: import libraries used for data manipulation and visualization.

**Outcome / observed output:** Python libraries are available to run subsequent data processing and plotting. Observed output: <Figure size 1200x600 with 1 Axes>
[Contains one or more image outputs (plots).]

**Inference / what this enables for the story:** This establishes standard tooling for the notebook.

**Example snippet:**

```python
# -- flights per airline -- 
flights_per_airline = (
    df.group_by("AIRLINE_NAME")
    .agg(pl.count("FLIGHT_NUM").alias("Total Flights"))
    .sort("Total Flights", descending=True)
)

plt.figure(figsize=(12,6))
sns.barplot(data=flights_per_airline.to_pandas(), 
            x="AIRLINE_NAME", y="Total Flights", hue = "AIRLINE_NAME", palette="viridis")
plt.xticks(rotation= 90)
plt.title("Flights per Airline")
plt.show()
```

### Step 7 — (code)

**What the cell does (one-line):** # -- Newness of airplanes owned by airlines --

**Why this step:** Set up analysis environment: import libraries used for data manipulation and visualization.

**Outcome / observed output:** Python libraries are available to run subsequent data processing and plotting. Observed output: <Figure size 1200x600 with 1 Axes>
[Contains one or more image outputs (plots).]

**Inference / what this enables for the story:** This establishes standard tooling for the notebook.

**Example snippet:**

```python
# -- Newness of airplanes owned by airlines -- 

plt.figure(figsize=(12,6))
sns.countplot(data=df.to_pandas(), x="TAIL_NUM", order=df.to_pandas()["TAIL_NUM"].value_counts().head(20).index)
plt.xticks(rotation=90)
plt.title("Newness (Usage Frequency) of Airplanes by Airlines (Top 20 Tail Numbers)")
plt.ylabel("Number of Flights")
plt.show()
```

### Step 8 — (code)

**What the cell does (one-line):** # -- Airline split according to location (Origin State)

**Why this step:** Set up analysis environment: import libraries used for data manipulation and visualization.

**Outcome / observed output:** Python libraries are available to run subsequent data processing and plotting. Observed output: <Figure size 1200x600 with 1 Axes>
[Contains one or more image outputs (plots).]

**Inference / what this enables for the story:** This establishes standard tooling for the notebook.

**Example snippet:**

```python
# -- Airline split according to location (Origin State) 

plt.figure(figsize=(12,6))
sns.countplot(data=df.to_pandas(), x="ORIGIN_STATE", hue="AIRLINE_NAME")
plt.xticks(rotation=90)
plt.title("Airline Split According to Origin Location (State)")
plt.ylabel("Flights Count")
plt.show()
```

### Step 9 — (code)

**What the cell does (one-line):** # -- Average Departure Delay by Airline --

**Why this step:** Set up analysis environment: import libraries used for data manipulation and visualization.

**Outcome / observed output:** Python libraries are available to run subsequent data processing and plotting. Observed output: <Figure size 1200x600 with 1 Axes>
[Contains one or more image outputs (plots).]

**Inference / what this enables for the story:** This establishes standard tooling for the notebook.

**Example snippet:**

```python
# -- Average Departure Delay by Airline --
avg_dep_delay = (
    df.group_by("AIRLINE_NAME").agg(pl.col("DEP_DELAY").mean().alias("Avg Departure Delay (min)")).sort("Avg Departure Delay (min)", descending=True)
)

plt.figure(figsize=(12,6))
sns.barplot(data=avg_dep_delay.to_pandas(), 
            x="AIRLINE_NAME", y="Avg Departure Delay (min)", hue = "AIRLINE_NAME",  palette="magma")
plt.xticks(rotation=90)
plt.title("Average Departure Delay per Airline")
plt.show()
```

### Step 10 — (code)

**What the cell does (one-line):** #on-time % for different airlines

**Why this step:** Preview rows of the data to inspect columns, sample values and obvious issues.

**Outcome / observed output:** Displays the first few rows so column names and sample records are visible. Output sample: shape: (5, 4) Observed output: shape: (5, 4)
┌──────────────────────────────┬───────────────┬────────────────┬───────────┐
│ AIRLINE_NAME                 ┆ Total Flights ┆ OnTime Flights ┆ OnTime %  │
│ ---                          ┆ ---           ┆ ---            ┆ ---       │
│ str                          ┆ u32           ┆ u32            ┆ f64       │
╞══════════════════════════════╪═══════════════╪════════════════╪═══════════╡
│ Delta Air Lines Inc.         ┆ 875881        ┆ 625041         ┆ 71.361406 │
│ Alaska Airlines 

**Inference / what this enables for the story:** Quickly helps validate that data loaded correctly and reveals immediate cleaning needs.

**Example snippet:**

```python
#on-time % for different airlines

on_time_airline = (
    df.with_columns((pl.col("ARR_DELAY") <= 0).alias("OnTime"))
    .group_by("AIRLINE_NAME")
    .agg([
        pl.count("FLIGHT_NUM").alias("Total Flights"),
        pl.col("OnTime").sum().alias("OnTime Flights")
    ])
    .with_columns((pl.col("OnTime Flights") / pl.col("Total Flights") * 100).alias("OnTime %"))
    .sort("OnTime %", descending=True)
)

on_time_airline.head()
```

### Step 11 — (markdown)

**What the cell does (one-line):** ### 2. AIRPORT AND ROUTE ANALYSIS

**Why this step:** Explains intent or documents the next step.

**Outcome / observed output:** No direct code output; provides context for following cells.

**Inference / what this enables for the story:** Helps readers understand why the next code step exists.

**Example snippet:**

```python
### 2. AIRPORT AND ROUTE ANALYSIS
```

### Step 12 — (code)

**What the cell does (one-line):** # -- Top 10 Busiest Origin Airports --

**Why this step:** Set up analysis environment: import libraries used for data manipulation and visualization.

**Outcome / observed output:** Python libraries are available to run subsequent data processing and plotting. Observed output: <Figure size 1200x600 with 1 Axes>
[Contains one or more image outputs (plots).]

**Inference / what this enables for the story:** This establishes standard tooling for the notebook.

**Example snippet:**

```python
# -- Top 10 Busiest Origin Airports --
top_origin = (
    df.group_by("ORIGIN_CODE")
    .agg(pl.count("FLIGHT_NUM").alias("Total Flights"))
    .sort("Total Flights", descending=True)
    .head(10)
)

plt.figure(figsize=(12,6))
sns.barplot(data=top_origin.to_pandas(), x="ORIGIN_CODE", y="Total Flights",hue = "ORIGIN_CODE", palette="cubehelix")
plt.title("Top 10 Origin Airports")
plt.show()
```

### Step 13 — (code)

**What the cell does (one-line):** # -- Top 10 Busiest Destination Airports --

**Why this step:** Set up analysis environment: import libraries used for data manipulation and visualization.

**Outcome / observed output:** Python libraries are available to run subsequent data processing and plotting. Observed output: <Figure size 1200x600 with 1 Axes>
[Contains one or more image outputs (plots).]

**Inference / what this enables for the story:** This establishes standard tooling for the notebook.

**Example snippet:**

```python
# -- Top 10 Busiest Destination Airports -- 
top_dest = (
    df.group_by("DEST_CODE")
    .agg(pl.count("FLIGHT_NUM").alias("Total Flights"))
    .sort("Total Flights", descending=True)
    .head(10)
)

plt.figure(figsize=(12,6))
sns.barplot(data=top_dest.to_pandas(), x="DEST_CODE", y="Total Flights",hue = "DEST_CODE", palette="cubehelix_r")
plt.title("Top 10 Destination Airports")
plt.show()
```

### Step 14 — (code)

**What the cell does (one-line):** # -- Most Common Routes--

**Why this step:** Set up analysis environment: import libraries used for data manipulation and visualization.

**Outcome / observed output:** Python libraries are available to run subsequent data processing and plotting. Observed output: <Figure size 1200x600 with 1 Axes>
[Contains one or more image outputs (plots).]

**Inference / what this enables for the story:** This establishes standard tooling for the notebook.

**Example snippet:**

```python
# -- Most Common Routes--
routes = (
    df.with_columns((pl.col("ORIGIN_CODE") + " → " + pl.col("DEST_CODE")).alias("Route"))
    .group_by("Route")
    .agg(pl.count("FLIGHT_NUM").alias("Total Flights"))
    .sort("Total Flights", descending=True)
    .head(10)
)

plt.figure(figsize=(12,6))
sns.barplot(data=routes.to_pandas(), x="Route", y="Total Flights",hue = "Route",  palette="viridis")
plt.xticks(rotation=45, ha="right")
plt.title("Top 10 Routes by Flights")
plt.show()
```

### Step 15 — (code)

**What the cell does (one-line):** # -- Scheduled departure time bias by airport --

**Why this step:** Set up analysis environment: import libraries used for data manipulation and visualization.

**Outcome / observed output:** Python libraries are available to run subsequent data processing and plotting. Observed output: <Figure size 1200x600 with 1 Axes>
[Contains one or more image outputs (plots).]

**Inference / what this enables for the story:** This establishes standard tooling for the notebook.

**Example snippet:**

```python
# -- Scheduled departure time bias by airport -- 
plt.figure(figsize=(12,6))
sns.histplot(data=df.to_pandas(), x=df.to_pandas()["SCHED_DEP_TIME"].dt.hour, bins=24, kde=False)
plt.title("Scheduled Departure Time Distribution (Hourly)")
plt.xlabel("Scheduled Departure Hour")
plt.ylabel("Number of Flights")
plt.show()
```

### Step 16 — (code)

**What the cell does (one-line):** # -- Avg Taxi Out by Airport & Airline

**Why this step:** Set up analysis environment: import libraries used for data manipulation and visualization.

**Outcome / observed output:** Python libraries are available to run subsequent data processing and plotting.

**Inference / what this enables for the story:** This establishes standard tooling for the notebook.

**Example snippet:**

```python
# -- Avg Taxi Out by Airport & Airline
plt.figure(figsize=(14,6))
sns.barplot(data=df.to_pandas(), x="ORIGIN_CODE", y="TAXI_OUT", hue="AIRLINE_NAME", estimator="mean")
plt.xticks(rotation=45)
plt.title("Avg Taxi Out Time by Airport & Airline")
plt.show()
```

### Step 17 — (code)

**What the cell does (one-line):** # -- Avg Taxi in time by airport and airline

**Why this step:** Set up analysis environment: import libraries used for data manipulation and visualization.

**Outcome / observed output:** Python libraries are available to run subsequent data processing and plotting.

**Inference / what this enables for the story:** This establishes standard tooling for the notebook.

**Example snippet:**

```python
# -- Avg Taxi in time by airport and airline
plt.figure(figsize=(14,6))
sns.barplot(data=df.to_pandas(), x="DEST_CODE", y="TAXI_IN", hue="AIRLINE_NAME", estimator="mean")
plt.xticks(rotation=90)
plt.title("Avg Taxi In Time by Airport & Airline")
plt.show()
```

### Step 18 — (code)

**What the cell does (one-line):** # -- Avg Time from pushback to actual takeoff

**Why this step:** Set up analysis environment: import libraries used for data manipulation and visualization.

**Outcome / observed output:** Python libraries are available to run subsequent data processing and plotting.

**Inference / what this enables for the story:** This establishes standard tooling for the notebook.

**Example snippet:**

```python
# -- Avg Time from pushback to actual takeoff
plt.figure(figsize=(12,6))
sns.barplot(data=df.to_pandas(), x="ORIGIN_CODE", y=(df.to_pandas()["WHEELS_OFF"] - df.to_pandas()["DEP_TIME"]), estimator="mean")
plt.xticks(rotation=90)
plt.title("Avg Time Between Taxi Out & Wheels Off (by Airport)")
plt.show()
```

### Step 19 — (code)

**What the cell does (one-line):** # -- Avg Time from landing to gate

**Why this step:** Set up analysis environment: import libraries used for data manipulation and visualization.

**Outcome / observed output:** Python libraries are available to run subsequent data processing and plotting.

**Inference / what this enables for the story:** This establishes standard tooling for the notebook.

**Example snippet:**

```python
# -- Avg Time from landing to gate
plt.figure(figsize=(12,6))
sns.barplot(data=df.to_pandas(), x="DEST_CODE", y=(df.to_pandas()["ARR_TIME"] - df.to_pandas()["WHEELS_ON"]), estimator="mean")
plt.xticks(rotation=90)
plt.title("Avg Time Wheels On → Taxi In (by Airport)")
plt.show()
```

### Step 20 — (code)

**What the cell does (one-line):** (code cell)

**Why this step:** Perform a technical data transformation or check.

**Outcome / observed output:** Cell produces a transformation, check, or intermediate object.

**Inference / what this enables for the story:** Moves preprocessing forward toward analysis-ready data.

### Step 21 — (markdown)

**What the cell does (one-line):** ### 3. SEASONS AND TIME ANALYSIS

**Why this step:** Explains intent or documents the next step.

**Outcome / observed output:** No direct code output; provides context for following cells.

**Inference / what this enables for the story:** Helps readers understand why the next code step exists.

**Example snippet:**

```python
### 3. SEASONS AND TIME ANALYSIS
```

### Step 22 — (code)

**What the cell does (one-line):** # flights per month

**Why this step:** Set up analysis environment: import libraries used for data manipulation and visualization.

**Outcome / observed output:** Python libraries are available to run subsequent data processing and plotting. Observed output: <Figure size 1000x500 with 1 Axes>
[Contains one or more image outputs (plots).]

**Inference / what this enables for the story:** This establishes standard tooling for the notebook.

**Example snippet:**

```python
# flights per month 

flights_per_month = (
    df.group_by("MONTH")
    .agg(pl.count("FLIGHT_NUM").alias("Total Flights"))
    .sort("MONTH")
)

plt.figure(figsize=(10,5))
sns.lineplot(data=flights_per_month.to_pandas(), x="MONTH", y="Total Flights", marker="o")
plt.title("Flights per Month")
plt.show()
```

### Step 23 — (code)

**What the cell does (one-line):** plt.figure(figsize=(12,6))

**Why this step:** Set up analysis environment: import libraries used for data manipulation and visualization.

**Outcome / observed output:** Python libraries are available to run subsequent data processing and plotting. Observed output: <Figure size 1200x600 with 1 Axes>
[Contains one or more image outputs (plots).]

**Inference / what this enables for the story:** This establishes standard tooling for the notebook.

**Example snippet:**

```python
plt.figure(figsize=(12,6))
sns.countplot(data=df.to_pandas(), x="DAY_OF_WEEK",hue ="DAY_OF_WEEK",  palette="muted")
plt.title("Flights by Day of Week (1=Mon)")
plt.ylabel("Total Flights")
plt.show()
```

### Step 24 — (markdown)

**What the cell does (one-line):** ### 4. DELAYS AND CANCELLATIONS

**Why this step:** Explains intent or documents the next step.

**Outcome / observed output:** No direct code output; provides context for following cells.

**Inference / what this enables for the story:** Helps readers understand why the next code step exists.

**Example snippet:**

```python
### 4. DELAYS AND CANCELLATIONS
```

### Step 25 — (code)

**What the cell does (one-line):** #-- Overall on-time performance

**Why this step:** Perform a technical data transformation or check.

**Outcome / observed output:** Cell produces a transformation, check, or intermediate object. Observed output: shape: (1, 3)
┌───────────────┬────────────────┬───────────┐
│ Total Flights ┆ OnTime Flights ┆ OnTime %  │
│ ---           ┆ ---            ┆ ---       │
│ u32           ┆ u32            ┆ f64       │
╞═══════════════╪════════════════╪═══════════╡
│ 5819079       ┆ 3732183        ┆ 64.137005 │
└───────────────┴────────────────┴───────────┘
[HTML output present]

**Inference / what this enables for the story:** Moves preprocessing forward toward analysis-ready data.

**Example snippet:**

```python
#-- Overall on-time performance
on_time_perf = (
    df.with_columns((pl.col("ARR_DELAY") <= 0).alias("OnTime"))
    .select([
        pl.count("FLIGHT_NUM").alias("Total Flights"),
        pl.col("OnTime").sum().alias("OnTime Flights")
    ])
    .with_columns((pl.col("OnTime Flights") / pl.col("Total Flights") * 100).alias("OnTime %"))
)

on_time_perf
```

### Step 26 — (code)

**What the cell does (one-line):** # -- Avg Arrival Delay by Airline

**Why this step:** Set up analysis environment: import libraries used for data manipulation and visualization.

**Outcome / observed output:** Python libraries are available to run subsequent data processing and plotting. Observed output: <Figure size 1200x600 with 1 Axes>
[Contains one or more image outputs (plots).]

**Inference / what this enables for the story:** This establishes standard tooling for the notebook.

**Example snippet:**

```python
# -- Avg Arrival Delay by Airline
plt.figure(figsize=(12,6))
sns.barplot(data=df.to_pandas(), x="AIRLINE_NAME", y="ARR_DELAY", estimator="mean")
plt.xticks(rotation=90)
plt.title("Avg Arrival Delay by Airline")
plt.show()
```

### Step 27 — (code)

**What the cell does (one-line):** # -- Avg Departure Delay by Airport

**Why this step:** Set up analysis environment: import libraries used for data manipulation and visualization.

**Outcome / observed output:** Python libraries are available to run subsequent data processing and plotting.

**Inference / what this enables for the story:** This establishes standard tooling for the notebook.

**Example snippet:**

```python
# -- Avg Departure Delay by Airport
plt.figure(figsize=(12,6))
sns.barplot(data=df.to_pandas(), x="ORIGIN_CODE", y="DEP_DELAY", estimator="mean")
plt.xticks(rotation=90)
plt.title("Avg Departure Delay by Airport")
plt.show()
```

### Step 28 — (code)

**What the cell does (one-line):** # -- Avg Arrival Delay by Airport

**Why this step:** Set up analysis environment: import libraries used for data manipulation and visualization.

**Outcome / observed output:** Python libraries are available to run subsequent data processing and plotting.

**Inference / what this enables for the story:** This establishes standard tooling for the notebook.

**Example snippet:**

```python
# -- Avg Arrival Delay by Airport
plt.figure(figsize=(12,6))
sns.barplot(data=df.to_pandas(), x="DEST_CODE", y="ARR_DELAY", estimator="mean")
plt.xticks(rotation=90)
plt.title("Avg Arrival Delay by Airport")
plt.show()
```

### Step 29 — (code)

**What the cell does (one-line):** # Cancel rate of every airline by time

**Why this step:** Set up analysis environment: import libraries used for data manipulation and visualization.

**Outcome / observed output:** Python libraries are available to run subsequent data processing and plotting.

**Inference / what this enables for the story:** This establishes standard tooling for the notebook.

**Example snippet:**

```python
# Cancel rate of every airline by time
plt.figure(figsize=(12,6))
sns.countplot(data=df.to_pandas()[df.to_pandas()["CANCELLED"]==True], x="AIRLINE_NAME")
plt.xticks(rotation=90)
plt.title("Cancelled Flights by Airline")
plt.show()
```

### Step 30 — (code)

**What the cell does (one-line):** delay_reasons = df.select([

**Why this step:** Set up analysis environment: import libraries used for data manipulation and visualization.

**Outcome / observed output:** Python libraries are available to run subsequent data processing and plotting.

**Inference / what this enables for the story:** This establishes standard tooling for the notebook.

**Example snippet:**

```python
delay_reasons = df.select([
    pl.col("AIR_SYS_DELAY").sum().alias("Air System Delay"),
    pl.col("SEC_DELAY").sum().alias("Security Delay"),
    pl.col("AIRLINE_DELAY").sum().alias("Airline Delay"),
    pl.col("LATE_AC_DELAY").sum().alias("Late Aircraft Delay"),
    pl.col("WEATHER_DELAY").sum().alias("Weather Delay")
])

colors = sns.color_palette("pastel")

plt.figure(figsize=(8,8))
delay_reasons_pd = delay_reasons.to_pandas().T.reset_index()
delay_reasons_pd.columns = ["Reason", "Minutes"]

plt.pie(delay_reasons_pd["Minutes"], labels=delay_reasons_pd["Reason"],colors=colors, autopct='%1.1f%%')
plt.title("Delay Reasons Breakdown", fontsize=14, weight="bold")
plt.show()
```

### Step 31 — (code)

**What the cell does (one-line):** # -- Avg Departure Delay per Airline

**Why this step:** Set up analysis environment: import libraries used for data manipulation and visualization.

**Outcome / observed output:** Python libraries are available to run subsequent data processing and plotting.

**Inference / what this enables for the story:** This establishes standard tooling for the notebook.

**Example snippet:**

```python
# -- Avg Departure Delay per Airline
plt.figure(figsize=(12,6))
sns.barplot(data=df.to_pandas(), x="AIRLINE_NAME", y="DEP_DELAY", estimator="mean")
plt.xticks(rotation=90)
plt.title("Avg Departure Delay by Airline")
plt.show()
```

### Step 32 — (code)

**What the cell does (one-line):** # -- Avg Departure Delay by Time of Day -

**Why this step:** Set up analysis environment: import libraries used for data manipulation and visualization.

**Outcome / observed output:** Python libraries are available to run subsequent data processing and plotting.

**Inference / what this enables for the story:** This establishes standard tooling for the notebook.

**Example snippet:**

```python
# -- Avg Departure Delay by Time of Day -
plt.figure(figsize=(12,6))
sns.barplot(data=df.to_pandas(), x=df.to_pandas()["SCHED_DEP_TIME"].dt.hour, y="DEP_DELAY", estimator="mean")
plt.title("Avg Departure Delay by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Avg Departure Delay (min)")
plt.show()
```

### Step 33 — (code)

**What the cell does (one-line):** (code cell)

**Why this step:** Perform a technical data transformation or check.

**Outcome / observed output:** Cell produces a transformation, check, or intermediate object.

**Inference / what this enables for the story:** Moves preprocessing forward toward analysis-ready data.

### Step 34 — (code)

**What the cell does (one-line):** (code cell)

**Why this step:** Perform a technical data transformation or check.

**Outcome / observed output:** Cell produces a transformation, check, or intermediate object.

**Inference / what this enables for the story:** Moves preprocessing forward toward analysis-ready data.

### Step 35 — (markdown)

**What the cell does (one-line):** ### 5. MISC

**Why this step:** Explains intent or documents the next step.

**Outcome / observed output:** No direct code output; provides context for following cells.

**Inference / what this enables for the story:** Helps readers understand why the next code step exists.

**Example snippet:**

```python
### 5. MISC
```

### Step 36 — (code)

**What the cell does (one-line):** # -- Correlation Between Distance and Arrival Delay--

**Why this step:** Set up analysis environment: import libraries used for data manipulation and visualization.

**Outcome / observed output:** Python libraries are available to run subsequent data processing and plotting.

**Inference / what this enables for the story:** This establishes standard tooling for the notebook.

**Example snippet:**

```python
# -- Correlation Between Distance and Arrival Delay--
plt.figure(figsize=(10,6))
sns.scatterplot(data=df.to_pandas().sample(5000), x="DISTANCE", y="ARR_DELAY", alpha=0.4)
plt.title("Distance vs Arrival Delay")
plt.xlabel("Distance (miles)")
plt.ylabel("Arrival Delay (min)")
plt.show()
```

### Step 37 — (code)

**What the cell does (one-line):** # -- Taxi Times by Airport --

**Why this step:** Preview rows of the data to inspect columns, sample values and obvious issues.

**Outcome / observed output:** Displays the first few rows so column names and sample records are visible.

**Inference / what this enables for the story:** Quickly helps validate that data loaded correctly and reveals immediate cleaning needs.

**Example snippet:**

```python
# -- Taxi Times by Airport -- 
taxi_times = (
    df.group_by("ORIGIN_CODE")
    .agg([
        pl.col("TAXI_OUT").mean().alias("Avg Taxi Out"),
        pl.col("TAXI_IN").mean().alias("Avg Taxi In")
    ])
    .sort("Avg Taxi Out", descending=True)
    .head(10)
)

taxi_times
```

### Step 38 — (code)

**What the cell does (one-line):** # --Diverted Flights % --

**Why this step:** Perform a technical data transformation or check.

**Outcome / observed output:** Cell produces a transformation, check, or intermediate object.

**Inference / what this enables for the story:** Moves preprocessing forward toward analysis-ready data.

**Example snippet:**

```python
# --Diverted Flights % -- 
diverted = (
    df.select([
        pl.col("DIVERTED").sum().alias("Diverted Flights"),
        pl.count("FLIGHT_NUM").alias("Total Flights")
    ])
    .with_columns((pl.col("Diverted Flights") / pl.col("Total Flights") * 100).alias("Diverted %"))
)

diverted
```

### Step 39 — (code)

**What the cell does (one-line):** (code cell)

**Why this step:** Perform a technical data transformation or check.

**Outcome / observed output:** Cell produces a transformation, check, or intermediate object.

**Inference / what this enables for the story:** Moves preprocessing forward toward analysis-ready data.


---

