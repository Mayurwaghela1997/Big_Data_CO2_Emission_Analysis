from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col, expr, monotonically_increasing_id
from pyspark.sql.types import IntegerType, StructField, StructType

# KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

# Plotting
import geopandas
import geoplot
import matplotlib.pyplot as plt
import mapclassify as mc


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
    changeData = df.select("change").toPandas()

    km = KMeans()

    # Determine best k using elbow method
    visualizer = KElbowVisualizer(km, k=(2,10))

    visualizer.fit(changeData)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure

    bestK = visualizer.elbow_value_ # Result = 4
    
    #TODO: remove this? Visualization results look better with 6 cluster as we have 2 clusters only containing "outsiders".
    bestK = 6

    # Determine clusters
    km = KMeans(n_clusters=bestK)
    km.fit(changeData)
    clusters = km.predict(changeData)

    return clusters, bestK


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
    co2_change.show()



    # APPLY K-MEANS ON CHANGE. Choose a reasonable amount of clusters
    
    # Get clustering
    clusters, bestK = clustering(co2_change)
    clusters = clusters.tolist()

    # Add clustering to spark dataframe
    clustersDf = spark.createDataFrame(clusters, IntegerType())
    clustersDf = clustersDf.withColumnRenamed("value", "cluster")

    co2_change = co2_change.withColumn("id", monotonically_increasing_id())
    clustersDf = clustersDf.withColumn("id", monotonically_increasing_id())

    co2_change = co2_change.join(clustersDf, ["id"])
    co2_change.show()

    
    # CHECK CLUSTERING
    x = co2_change.toPandas()['change']
    y = co2_change.toPandas()['cluster']
    plt.scatter(x,x, c=y, cmap='rainbow')
    plt.show()



    # VISUALIZATION

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    
    # Corrected wrong country codes, see: https://github.com/geopandas/geopandas/issues/1041
    world.loc[world['name'] == 'France', 'iso_a3'] = 'FRA'
    world.loc[world['name'] == 'Norway', 'iso_a3'] = 'NOR'
    world.loc[world['name'] == 'Somaliland', 'iso_a3'] = 'SOM'
    world.loc[world['name'] == 'Kosovo', 'iso_a3'] = 'RKS'

    #Match Country Code with iso_a3 column from geopandas
    clusterDeclaration = co2_change.withColumnRenamed("Country Code", "iso_a3").select("iso_a3", "change", "cluster").toPandas()
    world_with_cluster = world.join(clusterDeclaration.set_index('iso_a3'), on='iso_a3')

    k=bestK

    # VISUALIZATION OF CHANGE VALUE
    #ax = world_with_cluster[world_with_cluster.change.notna()].plot(column='change', figsize=(15, 10), legend=True, cmap="Spectral")
    #world_with_cluster[world_with_cluster.change.isna()].plot(color='lightgrey', hatch='///', ax=ax)
    #plt.show()


    # VISUALIZATION OF CLUSTER VALUE
    ax = world_with_cluster[world_with_cluster.cluster.notna()].plot(column='cluster', figsize=(13, 8), legend=True, cmap="Spectral", categorical=True)
    world_with_cluster[world_with_cluster.cluster.isna()].plot(color='lightgrey', hatch='///', ax=ax)
    plt.show()

    # Colormaps =  https://matplotlib.org/examples/color/colormaps_reference.html 


    sc.stop()

if __name__ == "__main__":
    main()
