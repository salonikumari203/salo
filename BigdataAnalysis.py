import os
import time
import json
import argparse
import numpy as np
import pandas as pd

PAGES = ["home", "search", "product", "cart", "checkout", "help", "blog"]
EVENTS = ["view", "click", "add_to_cart", "purchase", "bounce"]
START_TS = pd.Timestamp("2025-01-01")
DAYS = 30
SESSION_GAP_MINUTES = 30

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["dask", "pyspark"], default="dask")
    ap.add_argument("--parts", type=int, default=8)
    ap.add_argument("--rows-per-part", type=int, default=250_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--plot", action="store_true", help="Show hourly plot (matplotlib required)")
    return ap.parse_args()

def make_synthetic_chunk(seed, n, n_users, pages=PAGES, events=EVENTS):
    rng = np.random.default_rng(seed)
    user_ids = rng.integers(1, n_users+1, size=n, dtype=np.int64)
    ts_base = int(START_TS.value // 10**9)
    ts = ts_base + rng.integers(0, DAYS*24*3600, size=n, dtype=np.int64)
    page = rng.choice(pages, p=[0.22,0.17,0.28,0.12,0.05,0.08,0.08], size=n)
    event = rng.choice(events, p=[0.45,0.30,0.13,0.02,0.10], size=n)
    price = np.where(event=="purchase", rng.uniform(5, 300, size=n), 0.0)
    df = pd.DataFrame({
        "user_id": user_ids,
        "ts": pd.to_datetime(ts, unit="s"),
        "page": pd.Categorical(page, categories=pages),
        "event": pd.Categorical(event, categories=events),
        "price": price.astype("float32")
    })
    return df

def run_dask(parts, rows_per_part, seed, outdir, show_plot):
    try:
        import dask
        import dask.dataframe as dd
        from dask import delayed
        from dask.distributed import Client, LocalCluster
    except Exception as e:
        raise RuntimeError("Dask not installed. pip install 'dask[complete]'") from e

    N_ROWS = parts * rows_per_part
    N_USERS = max(1_000_000, rows_per_part // 10)


    delayed_parts = [delayed(make_synthetic_chunk)(seed + i, rows_per_part, N_USERS) for i in range(parts)]
    ddf = dd.from_delayed(delayed_parts)
    ddf = ddf.persist()

    
    approx_users = int(ddf["user_id"].nunique_approx().compute())
    events_count = ddf.groupby("event").size().compute().sort_values(ascending=False)
    top_pages = ddf.groupby("page")["user_id"].nunique_approx().compute().sort_values(ascending=False)
    total_rev = float(ddf["price"].sum().compute())
    purchases = ddf[ddf["event"]=="purchase"]
    unique_buyers = int(purchases["user_id"].nunique_approx().compute())
    visitors = int(ddf[ddf["event"]!="purchase"]["user_id"].nunique_approx().compute())
    conv_rate = unique_buyers / visitors if visitors>0 else float("nan")
    hourly = ddf.assign(hour=ddf["ts"].dt.floor("H")).groupby("hour").size().compute().sort_index()

    
    SESSION_GAP = pd.Timedelta(minutes=SESSION_GAP_MINUTES)
    ddf2 = ddf.set_index("ts", shuffle="tasks")
    def sessionize(pdf):
        pdf = pdf.sort_values(["user_id", "ts"])
        gap = (pdf["ts"] - pdf.groupby("user_id")["ts"].shift())
        new_sess = (gap.isna()) | (gap > SESSION_GAP) | (pdf["user_id"] != pdf["user_id"].shift())
        pdf["session_id"] = new_sess.cumsum().astype("int64")
        return pdf
    ddf_sess = ddf2.map_partitions(sessionize, meta=ddf2.assign(session_id=np.int64(0)))
    sess_len = ddf_sess.groupby("session_id").size().compute()
    sess_dur = (ddf_sess.groupby("session_id")["ts"].max() - ddf_sess.groupby("session_id")["ts"].min()).compute().dt.total_seconds()
    avg_events_per_session = float(sess_len.mean())
    median_session_seconds = float(sess_dur.median())

    
    timings = []
    for parts_try in [max(1, parts//2), parts, parts*2]:
        ddf_re = ddf.repartition(npartitions=max(1, parts_try))
        t0 = time.time()
        _ = ddf_re.groupby("page").size().compute()
        dt = time.time() - t0
        timings.append(("dask", parts_try, dt))

    
    os.makedirs(outdir, exist_ok=True)
    events_count.to_csv(os.path.join(outdir, "events_count.csv"))
    top_pages.to_csv(os.path.join(outdir, "top_pages_unique_users.csv"))
    hourly.to_csv(os.path.join(outdir, "hourly_events.csv"), header=True)
    with open(os.path.join(outdir, "scalability_timings.json"), "w") as f:
        json.dump(timings, f, indent=2)

  
    busiest_hour = hourly.idxmax() if len(hourly)>0 else None
    busiest_hour_ct = int(hourly.max()) if len(hourly)>0 else 0
    print(f"Engine: Dask | Rows: {N_ROWS:,} | Partitions: {parts}")
    print(f"Approx unique users: {approx_users:,}")
    print(f"Revenue: ${total_rev:,.2f} | Conversion: {conv_rate:.4f}")
    print(f"Avg events/session: {avg_events_per_session:.2f} | Median session (s): {median_session_seconds:.1f}")
    print(f"Top page: {top_pages.index[0] if len(top_pages)>0 else 'n/a'} ({int(top_pages.iloc[0]) if len(top_pages)>0 else 0})")
    print(f"Busiest hour: {busiest_hour} ({busiest_hour_ct} events)")
    print("Timings:", timings)

    if show_plot:
        try:
            import matplotlib.pyplot as plt
            hourly.plot(title="Hourly Event Volume", figsize=(10,4))
            plt.tight_layout()
            plt.show()
        except Exception:
            print("matplotlib not available or failed to plot.")

def run_pyspark(parts, rows_per_part, seed, outdir, show_plot):
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, expr, rand, when, lit, from_unixtime, unix_timestamp, approx_count_distinct, sum as spark_sum, date_trunc
        from pyspark.sql import Window
    except Exception as e:
        raise RuntimeError("PySpark not installed. pip install pyspark") from e

    N_ROWS = parts * rows_per_part
    N_USERS = max(1_000_000, rows_per_part // 10)

    spark = SparkSession.builder.appName("BigDataAnalysisDemo").getOrCreate()
    sdf = spark.range(0, N_ROWS).repartition(parts)

    sdf = (sdf
           .withColumn("user_id", ((col("id") * 2654435761) % lit(N_USERS)).cast("long") + 1)
           .withColumn("ts_sec", (lit(int(START_TS.value//10**9)) + (col("id") % lit(DAYS*24*3600))).cast("long"))
           .withColumn("ts", from_unixtime(col("ts_sec")).cast("timestamp"))
           .withColumn("page_idx", (col("id") % lit(len(PAGES))))
           .withColumn("event_idx", (col("id") % lit(len(EVENTS))))
    )
    page_expr = "CASE " + " ".join([f"WHEN page_idx={i} THEN '{p}'" for i,p in enumerate(PAGES)]) + " END"
    event_expr = "CASE " + " ".join([f"WHEN event_idx={i} THEN '{e}'" for i,e in enumerate(EVENTS)]) + " END"
    sdf = sdf.withColumn("page", expr(page_expr)).withColumn("event", expr(event_expr))
    sdf = sdf.withColumn("price", when(col("event")=="purchase", (rand(seed+1)*295 + 5)).otherwise(lit(0.0)))
    sdf.cache()

    approx_users = int(sdf.agg(approx_count_distinct("user_id").alias("u")).collect()[0]["u"])
    events_count_pd = sdf.groupBy("event").count().toPandas().set_index("event")["count"].sort_values(ascending=False)
    top_pages_pd = sdf.groupBy("page").agg(approx_count_distinct("user_id").alias("u")).toPandas().set_index("page")["u"].sort_values(ascending=False)
    total_rev = float(sdf.agg(spark_sum("price")).collect()[0][0])
    unique_buyers = int(sdf.filter(col("event")=="purchase").agg(approx_count_distinct("user_id")).collect()[0][0])
    visitors = int(sdf.filter(~(col("event")=="purchase")).agg(approx_count_distinct("user_id")).collect()[0][0])
    conv_rate = unique_buyers / visitors if visitors>0 else float("nan")
    hourly_pd = (sdf.withColumn("hour", date_trunc("hour", col("ts")))
                 .groupBy("hour").count().orderBy("hour").toPandas().set_index("hour")["count"])

    # sessionization (approximate)
    w = Window.partitionBy("user_id").orderBy("ts")
    sdf_sess = sdf.selectExpr("*", "lag(ts) over (partition by user_id order by ts) as prev_ts")
    sdf_sess = sdf_sess.withColumn("session_new", (sdf_sess["prev_ts"].isNull()) | ((unix_timestamp("ts")-unix_timestamp("prev_ts")) > SESSION_GAP_MINUTES*60))
    sdf_sess = sdf_sess.withColumn("session_new_int", when(sdf_sess["session_new"], lit(1)).otherwise(lit(0)))
    sdf_sess = sdf_sess.withColumn("session_id", expr("sum(session_new_int) over (partition by user_id order by ts rows between unbounded preceding and current row)"))
    sess_len_pd = sdf_sess.groupBy("session_id").count().toPandas().set_index("session_id")["count"]
    sess_dur_pd = (sdf_sess.groupBy("session_id")
                       .agg((unix_timestamp(expr("max(ts)")) - unix_timestamp(expr("min(ts)"))).alias("dur"))
                       .toPandas().set_index("session_id")["dur"])
    avg_events_per_session = float(sess_len_pd.mean())
    median_session_seconds = float(sess_dur_pd.median())

    
    timings = []
    for parts_try in [max(1, parts//2), parts, parts*2]:
        sdf_re = sdf.repartition(parts_try)
        t0 = time.time()
        _ = sdf_re.groupBy("page").count().collect()
        dt = time.time() - t0
        timings.append(("pyspark", parts_try, dt))

    
    os.makedirs(outdir, exist_ok=True)
    events_count_pd.to_csv(os.path.join(outdir, "events_count.csv"))
    top_pages_pd.to_csv(os.path.join(outdir, "top_pages_unique_users.csv"))
    hourly_pd.to_csv(os.path.join(outdir, "hourly_events.csv"), header=True)
    with open(os.path.join(outdir, "scalability_timings.json"), "w") as f:
        json.dump(timings, f, indent=2)

    busiest_hour = hourly_pd.idxmax() if len(hourly_pd)>0 else None
    busiest_hour_ct = int(hourly_pd.max()) if len(hourly_pd)>0 else 0
    print(f"Engine: PySpark | Rows: {N_ROWS:,} | Partitions: {parts}")
    print(f"Approx unique users: {approx_users:,}")
    print(f"Revenue: ${total_rev:,.2f} | Conversion: {conv_rate:.4f}")
    print(f"Avg events/session: {avg_events_per_session:.2f} | Median session (s): {median_session_seconds:.1f}")
    print(f"Top page: {top_pages_pd.index[0] if len(top_pages_pd)>0 else 'n/a'} ({int(top_pages_pd.iloc[0]) if len(top_pages_pd)>0 else 0})")
    print(f"Busiest hour: {busiest_hour} ({busiest_hour_ct} events)")
    print("Timings:", timings)

    if show_plot:
        try:
            import matplotlib.pyplot as plt
            hourly_pd.plot(title="Hourly Event Volume", figsize=(10,4))
            plt.tight_layout()
            plt.show()
        except Exception:
            print("matplotlib not available or failed to plot.")

def main():
    args = parse_args()
    outdir = args.outdir
    if args.engine == "dask":
        run_dask(args.parts, args.rows_per_part, args.seed, outdir, args.plot)
    else:
        run_pyspark(args.parts, args.rows_per_part, args.seed, outdir, args.plot)

if __name__ == "__main__":
    main()

