from pyspark.sql import SparkSession
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyspark.sql.functions as func
from pyspark.sql.functions import count, desc, regexp_extract
import seaborn as sns
import pandas as pd

# Initialize SparkSession
spark = SparkSession.builder \
        .master("local[2]") \
        .appName("LogMiningAnalysis") \
        .config("spark.local.dir", "/fastdata/acq22vv") \
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

# Reading the data from NASA access logs
log_file = spark.read.text("../Data/NASA_access_log_Jul95.gz").cache()

# Extracting fields from the log file
data_df = log_file.withColumn('host', func.regexp_extract('value', '^(.*) - -.*', 1)) \
                  .withColumn('timestamp', func.regexp_extract('value', '.* - - \[(.*)\].*', 1)) \
                  .withColumn('request', func.regexp_extract('value', '.*\"(.*)\".*', 1)) \
                  .withColumn('HTTP_reply_code', func.split('value', ' ').getItem(func.size(func.split('value', ' ')) - 2).cast("int")) \
                  .withColumn('bytes_in_reply', func.split('value', ' ').getItem(func.size(func.split('value', ' ')) - 1).cast("int")) \
                  .drop("value").cache()

# Total number of requests from Germany, Canada, and Singapore
data_ger = data_df.filter(data_df.host.endswith(".de"))
data_can = data_df.filter(data_df.host.endswith(".ca"))
data_sin = data_df.filter(data_df.host.endswith(".sg"))

country_data = [data_ger, data_can, data_sin]
country_names = ['Germany', 'Canada', 'Singapore']

for i, country_df in enumerate(country_data):
    print(f"\n\nThere are a total of {country_df.count()} requests from {country_names[i]}\n")

# Total unique requests and top 9 most frequent hosts
unique_requests_df = lambda df: df.select("host").groupBy("host").agg(count("host").alias("count")).orderBy(desc("count"), "host")
data_ger_unq = unique_requests_df(data_ger)
data_can_unq = unique_requests_df(data_can)
data_sin_unq = unique_requests_df(data_sin)

top_hosts_df = lambda df: df.limit(9)
data_ger_top9 = top_hosts_df(data_ger_unq)
data_can_top9 = top_hosts_df(data_can_unq)
data_sin_top9 = top_hosts_df(data_sin_unq)

# Collect top hosts into dictionaries
dict_from_df = lambda df: df.rdd.map(lambda row: (row['host'], row['count'])).collectAsMap()
data_ger_dict = dict_from_df(data_ger_top9)
data_can_dict = dict_from_df(data_can_top9)
data_sin_dict = dict_from_df(data_sin_top9)

# Print results
print(f"\n\nThere are a total of {data_ger_unq.count()} unique requests from Germany")
print(f"\nThere are a total of {data_can_unq.count()} unique requests from Canada")
print(f"\nThere are a total of {data_sin_unq.count()} unique requests from Singapore\n")

print(f"The top 9 most frequent hosts from Germany are:\n")
print('\n'.join(data_ger_dict.keys()))

print(f"\nThe top 9 most frequent hosts from Canada are:\n")
print('\n'.join(data_can_dict.keys()))

print(f"\nThe top 9 most frequent hosts from Singapore are:\n")
print('\n'.join(data_sin_dict.keys()))

# Plotting percentage distribution of requests
calculate_percentage = lambda counts, total: {k: round(v / total, 2) for k, v in counts.items()}
calculate_rest_percentage = lambda top_counts, total: round(total - top_counts / total, 2)

def plot_pie_chart(data_dict, filename, title):
    proportions = list(data_dict.values())
    plt.figure(figsize=(12, 8))
    explode = [0.1 if i == 0 else 0 for i in range(len(proportions))]

    fig, ax = plt.subplots()
    wedges, _, _ = ax.pie(proportions, explode=explode, labels=None, autopct='%1.1f%%')
    legend_labels = [f"{label}: {prop * 100:.1f}%" for label, prop in zip(data_dict.keys(), proportions)]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0.5, 0.5, 1))
    ax.set_title(title)

    plt.savefig(filename, bbox_inches='tight')
    plt.clf()

total_counts = {
    'Germany': data_ger.count(),
    'Canada': data_can.count(),
    'Singapore': data_sin.count()
}

top9_counts = {
    'Germany': data_ger_top9.agg(func.sum('count')).collect()[0][0],
    'Canada': data_can_top9.agg(func.sum('count')).collect()[0][0],
    'Singapore': data_sin_top9.agg(func.sum('count')).collect()[0][0]
}

for country, total_count in total_counts.items():
    top9_count = top9_counts[country]
    rest_count = total_count - top9_count
    data_dict = globals()[f"data_{country.lower()}_dict"]
    data_dict["Rest"] = calculate_rest_percentage(top9_count, total_count)
    data_dict = calculate_percentage(data_dict, total_count)
    plot_pie_chart(data_dict, f"../Output/{country}_Top_Hosts_Percentage.jpg", f"Top Hosts Percentage in {country}")

# Heatmap for most frequent host
def plot_heatmap(df, top_host, filename, title):
    time_host = df.filter(df.host == top_host).select('timestamp', 'host')
    time_host = time_host.withColumn('day', regexp_extract('timestamp', r'^(\d{2})', 1)) \
                         .withColumn('hour', regexp_extract('timestamp', r'^\d{2}/\w{3}/\d{4}:(\d{2})', 1))
    
    heat_matrix = time_host.groupBy("day", "hour").count()
    fin_heat_matrix = heat_matrix.toPandas()
    heat_data = pd.pivot_table(fin_heat_matrix, values='count', index=['hour'], columns=['day'])
    x_axis_range = range(int(heat_data.columns.min()), int(heat_data.columns.max()) + 1)
    
    plt.figure(figsize=(12, 7))
    sns.heatmap(heat_data, cmap="BuPu", xticklabels=x_axis_range, cbar_kws={'label': 'Number of Visits'})
    plt.title(title)
    plt.xlabel("Day of the Month")
    plt.ylabel("Time of Visit")
    plt.gca().invert_yaxis()
    plt.savefig(filename)
    plt.clf()

for i, (df, top_host) in enumerate(zip(country_data, [data_ger_top9, data_can_top9, data_sin_top9])):
    top_host_name = top_host.select("host").collect()[0]['host']
    plot_heatmap(df, top_host_name, f"../Output/{country_names[i]}_Heatmap.jpg", f"Heatmap of Visits from Most Frequent Host in {country_names[i]}")
