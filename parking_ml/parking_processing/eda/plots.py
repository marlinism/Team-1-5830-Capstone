import os
import duckdb
import matplotlib.pyplot as plt

def _savefig(outdir: str, name: str) -> None:
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, name), dpi=200)
    plt.close()

def run_eda(pattern: str, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    con = duckdb.connect(database=":memory:")

    print("EDA step starts:")

    """
    Row counts per day - Checks daily coverage stability & expose missing data
    """
    rows_by_day = con.execute(f"""
        SELECT CAST(ts15_utc AS DATE) AS day, COUNT(*) AS rows
        FROM read_csv_auto('{pattern}')
        GROUP BY 1 ORDER BY 1
    """).df()

    plt.figure()
    plt.plot(rows_by_day["day"], rows_by_day["rows"])
    plt.title("Row counts per day")
    plt.xlabel("day")
    plt.ylabel("rows")
    _savefig(outdir, "rows_per_day.png")

    """
    Row counts per week - Checks structural changes across the year
    """
    rows_by_week = con.execute(f"""
        SELECT DATE_TRUNC('week', ts15_utc) AS week, COUNT(*) AS rows
        FROM read_csv_auto('{pattern}')
        GROUP BY 1 ORDER BY 1
    """).df()

    plt.figure()
    plt.plot(rows_by_week["week"], rows_by_week["rows"])
    plt.title("Row counts per week")
    plt.xlabel("week")
    plt.ylabel("rows")
    _savefig(outdir, "rows_per_week.png")

    """
    Unique SourceElementKey per week: Measures node churn to ensure if 
    the graph node is stable
    """
    nodes_by_week = con.execute(f"""
        SELECT DATE_TRUNC('week', ts15_utc) AS week, COUNT(DISTINCT sourceelementkey) AS n_nodes
        FROM read_csv_auto('{pattern}')
        GROUP BY 1 ORDER BY 1
    """).df()

    plt.figure()
    plt.plot(nodes_by_week["week"], nodes_by_week["n_nodes"])
    plt.title("Unique SourceElementKey per week (node churn)")
    plt.xlabel("week")
    plt.ylabel("unique nodes")
    _savefig(outdir, "nodes_per_week.png")

    """
    Hour x day-of-week heatmap - Reveals predictable weekly occupancy
    """
    heat = con.execute(f"""
        SELECT
          EXTRACT('dow' FROM ts15_utc) AS dow,
          EXTRACT('hour' FROM ts15_utc) AS hour,
          AVG(occ_rate) AS mean_occ
        FROM read_csv_auto('{pattern}')
        GROUP BY 1,2
    """).df()

    pivot = heat.pivot(index="dow", columns="hour", values="mean_occ").sort_index()
    plt.figure(figsize=(10, 4))
    plt.imshow(pivot.values, aspect="auto")
    plt.title("Mean occ_rate by day-of-week × hour (ts15_utc)")
    plt.xlabel("hour")
    plt.ylabel("dow (0=Sun)")
    plt.colorbar(label="mean occ_rate")
    _savefig(outdir, "heatmap_occ_rate_dow_hour.png")

    """
    Full-rate by hour
    """
    full_by_hour = con.execute(f"""
        SELECT EXTRACT('hour' FROM ts15_utc) AS hour, AVG(is_full) AS full_rate
        FROM read_csv_auto('{pattern}')
        GROUP BY 1 ORDER BY 1
    """).df()

    plt.figure()
    plt.plot(full_by_hour["hour"], full_by_hour["full_rate"])
    plt.title("Full rate by hour (ts15_utc)")
    plt.xlabel("hour")
    plt.ylabel("P(full)")
    _savefig(outdir, "full_rate_by_hour.png")

    """
    Scatter Map: Nodes colored by mean occupancy rate
    """
    nodes = con.execute(f"""
        SELECT sourceelementkey,
               AVG(occ_rate) AS mean_occ,
               MEDIAN(lat) AS lat,
               MEDIAN(lon) AS lon
        FROM read_csv_auto('{pattern}')
        GROUP BY 1
    """).df()

    plt.figure(figsize=(6, 6))
    plt.scatter(nodes["lon"], nodes["lat"], s=2, c=nodes["mean_occ"])
    plt.title("Map: nodes colored by mean occ_rate")
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.colorbar(label="mean occ_rate")
    _savefig(outdir, "map_mean_occ_rate.png")

    print("EDA step completed!")
    print("EDA saved to:", outdir)