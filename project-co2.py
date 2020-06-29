from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col, expr, monotonically_increasing_id, max, min, udf, desc, asc, sum
from pyspark.sql.types import IntegerType, StructField, StructType, StringType

# KMeans
import numpy as np
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator

# Plotting
import matplotlib.pyplot as plt
import pandas as pd
import geopandas
import geoplot
#import mapclassify as mc


# Function for Preprocessing
def preprocessing(df):
    print("PREPROCESSING PART")

    # Remove countries that do not contain any data between 2004 - 2014. As we not have any data about that country, we won't consider it in further processing
    df = df.na.drop("all")
    df = df.na.drop("all", subset=("2004","2014")) #Can be removed if we remove any null value anyways

    # We chose to ignore these countries where 2004 or 2014 is missing (or Country) as we can not compute the change then
    df = df.na.drop("any")

    return df


def clustering(df):
    # Collect Features for Clustering, we added isReduced to get a better seperation of values above or below zero
    FEATURES_COL = ['change', 'isReduced']
    # FEATURES_COL = ['change']
    vecAssembler = VectorAssembler(inputCols=FEATURES_COL, outputCol="features")
    df_kmeans = vecAssembler.transform(df)
    df_kmeans.show()


    # Estimate best choice of k
    maxK = 10
    cost = np.zeros(maxK)
    for k in range(2,maxK):
        kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
        model = kmeans.fit(df_kmeans.sample(False,0.1, seed=42))
        cost[k] = model.computeCost(df_kmeans) # Deprecated in Version 3.0, use ClusteringEvaluator instead then
        
    fig, ax = plt.subplots(1,1, figsize =(8,6))
    ax.plot(range(2,maxK),cost[2:maxK])
    ax.set_xlabel('k')
    ax.set_ylabel('cost')
    plt.savefig("figures/bestK.png")
    plt.show()

    # After analyzing the costs from the plot, we chose k = 5.
    bestK = 5

    # Training model
    kmeans = KMeans().setK(bestK).setFeaturesCol("features").setPredictionCol("cluster")
    
    # TODO: Choose model
    model = kmeans.fit(df_kmeans.sample(False,0.1, seed=42))
    #model = kmeans.fit(df_kmeans)
    
    # Predict Clusters
    predictions = model.transform(df_kmeans)
    predictions.show()
    clustered_df = predictions

    # See Cluster Centroids Values
    centers = model.clusterCenters()

    return clustered_df, bestK, centers


def main():
    sc = SparkContext("local", "co2 emissions")
    sc.setLogLevel("ERROR")
    spark = SparkSession(sc)

    # IMPORTING DATA    

    # load the co2 dataset
    co2_data = spark.read.option("inferSchema", "true").option("header", "true").csv("co2-dataset-edited.csv")
    # only consider the data that is important for our task (Removed Indicator Name, Indicator Code and all other year dates except 2004-2014)
    co2_data = co2_data.select("Country Name", "Country Code", "2004", "2014")
    co2_data.show()
    co2_data.printSchema()


    # PREPROCESSING
    clean_co2_data = preprocessing(co2_data)
    clean_co2_data.show()



    # CALCULATION THE CHANGE
    co2_change = clean_co2_data.withColumn("change", col("2014") - col("2004"))
    # co2_change = co2_change.withColumn("id", monotonically_increasing_id())

    # Did a country reduce is co2 emission during that time (resulting in a negative change)
    co2_change = co2_change.withColumn("isReduced", col("change") <= 0 )
    co2_change.show()




    # APPLY K-MEANS ON CHANGE. Choose a reasonable amount of clusters
    
    # Get clustering
    clustered_df, bestK, centroids = clustering(co2_change)
    co2_change = clustered_df

    
    # CHECK CLUSTERING VIA VISUALIZING
    print("Cluster Centers: ")
    for center in centroids:
        print(center)

    x = co2_change.toPandas()['change']
    y = co2_change.toPandas()['cluster']
    plt.scatter(x,x, c=y, cmap='rainbow')
    plt.savefig("figures/clustering_values.png")
    plt.show()

    # Data about Cluster
    udf_label = udf(lambda minV, maxV: str(round(minV,3)) + " - " + str(round(maxV, 3)), StringType())
    clusterInfo = co2_change.groupBy("cluster").agg(min("change"), max("change")).sort("cluster")
    clusterInfo = clusterInfo.withColumn("label", udf_label(col("min(change)"), col("max(change)")) )
    clusterInfo.show()

    # VISUALIZATION

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    
    # Corrected wrong country codes, see: https://github.com/geopandas/geopandas/issues/1041
    world.loc[world['name'] == 'France', 'iso_a3'] = 'FRA'
    world.loc[world['name'] == 'Norway', 'iso_a3'] = 'NOR'
    world.loc[world['name'] == 'Somaliland', 'iso_a3'] = 'SOM'
    world.loc[world['name'] == 'Kosovo', 'iso_a3'] = 'RKS'

    # #Match Country Code with iso_a3 column from geopandas
    clusterDeclaration = co2_change.withColumnRenamed("Country Code", "iso_a3").select("iso_a3", "change", "cluster", "isReduced").toPandas()
    world_with_cluster = world.join(clusterDeclaration.set_index('iso_a3'), on='iso_a3')

    # k=bestK

    # # VISUALIZATION OF CHANGE VALUE
    ax = world_with_cluster[world_with_cluster.change.notna()].plot(column='change', figsize=(13, 8), legend=True, cmap="Spectral_r")
    world_with_cluster[world_with_cluster.change.isna()].plot(color='lightgrey', hatch='///', ax=ax)
    plt.title("Co2 Emission Change between 2004 and 2014")
    plt.savefig("figures/co2_change.png")
    plt.show()


    # VISUALIZATION OF CLUSTER VALUE
    
    ax = world_with_cluster[world_with_cluster.cluster.notna()].plot(column='cluster', figsize=(13, 8), legend=True, cmap="Spectral_r", categorical=True)
    labels = clusterInfo.toPandas()['label']
    leg = ax.get_legend().get_texts()
    for index in range(bestK):
        leg[index].set_text(labels[index])
    world_with_cluster[world_with_cluster.cluster.isna()].plot(color='lightgrey', hatch='///', ax=ax)
    plt.title("Co2 Emiision Change between 2004 and 2014 displayed by Clusters")
    plt.savefig("figures/co2_clusters.png")
    plt.show()


    # VISUALIZATION OF IMPROVEMENT
    ax = world_with_cluster.plot(color='lightgrey', figsize=(16, 8))
    world_with_cluster[world_with_cluster.change.notna() & world_with_cluster.isReduced].plot(column='change', legend=True, cmap="summer", ax = ax, legend_kwds={'shrink': 0.7})
    world_with_cluster[world_with_cluster.change.notna() & world_with_cluster.isReduced == False].plot(column='change', legend=True, cmap="Reds", ax = ax, legend_kwds={'shrink': 0.7})
    plt.title("Co2 Emission Change between 2004 and 2014")
    plt.savefig("figures/co2_improvement.png")
    plt.show()
    


    # SOME FACTS, Remark: Only countries that have co2 emission values for 2004 AND 2014 are taken into account here
    
    # Which top 3 countries have the biggest co2 emission in year 2004 and year 2014?
    print("Which top 3 countries have the biggest co2 emission in year 2004 and year 2014?")
    print("In 2004:")
    max_in_2004 = co2_change.orderBy(desc("2004")).limit(3).show()
    min_in_2004 = co2_change.orderBy(asc("2004")).limit(3).show()

    print("In 2014:")
    max_in_2014 = co2_change.orderBy(desc("2014")).limit(3).show()
    min_in_2014 = co2_change.orderBy(asc("2014")).limit(3).show()

    # Which country (top 3) reduced co2 emissions the most? 
    print("Which country (top 3) reduced co2 emissions the most? ")
    highest_reduction = co2_change.orderBy(asc("change")).limit(3).show()

    # Which country (top 3) increased his co2 emissions the most?
    print("Which country (top 3) increased his co2 emissions the most?")
    most_increased = co2_change.orderBy(desc("change")).limit(3).show()

    # How many countries increased /reduces their emissions? Also which?
    print("Reduced:")
    countries_that_reduced = co2_change.filter(col("isReduced") == True).sort(asc("change"))
    num_reduced = countries_that_reduced.count()
    print(num_reduced)
    countries_that_reduced.show()

    print("Increased:")
    countries_that_increased = co2_change.filter(col("isReduced") == False).sort(desc("change"))
    num_increased = countries_that_reduced.count()
    print(num_increased)
    countries_that_increased.show()

    # Sum of change:
    print("Sum of change:")
    print("Reduced:")
    co2_change.select("change", "isReduced").filter(col("isReduced") == True).agg(sum("change")).show()
    print("Increased:")
    co2_change.select("change", "isReduced").filter(col("isReduced") == False).agg(sum("change")).show()
    print("Total Change:")
    co2_change.agg(sum("change")).show()

    # For personal interest data of: Germany, USA, France, China, Sweden
    print("Out of interest:")
    co2_change.filter( (col("Country Name") == "Germany") | (col("Country Name") == "United States") | (col("Country Name") == "France") | (col("Country Name") == "China") | (col("Country Name") == "Sweden")).sort(asc("2014")).show()

    sc.stop()

if __name__ == "__main__":
    main()
