import mapclassify
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
import geopandas
import geoplot
import matplotlib.pyplot as plt

from pyspark.sql.functions import when, coalesce

conf = SparkConf().setAppName("Hello World")
sc = SparkContext('local', conf)
spark = SparkSession.builder.appName("Hello World").config("spark.debug.maxToStringFields", 1000).getOrCreate()

print('Hello World')

df = spark.read.option('inferSchema', 'true').option('header', 'true').csv('Co2.csv', escape="@")

df.show()
df = df.select("Country Name","Country Code", "2004", "2014")
df = df.dropna()
df = df.select("Country Name","Country Code", "2004", "2014", (F.col("2004") - F.col("2014")).alias("Change"))
df.show()
print(df.count())
vecAssembler = VectorAssembler(inputCols=["Change"], outputCol="features")
dataset = vecAssembler.transform(df)
dataset.show()

#sil_dist = []
#err = []
#for k in range(8,9):

kmeans = KMeans().setK(5).setSeed(1)
model = kmeans.fit(dataset.select('features'))
# Make predictions
predictions = model.transform(dataset)
predictions.orderBy("Change").show()
# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
#sil_dist.append(silhouette)
#err.append(model.computeCost(dataset))
print("Silhouette with squared euclidean distance = " + str(silhouette))

#print(sil_dist)
#print(err)
# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

#predictions = predictions.toPandas()
#data = predictions.Change.squeeze()
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
world.loc[world['name'] == 'France', 'iso_a3'] = 'FRA'
world.loc[world['name'] == 'Norway', 'iso_a3'] = 'NOR'
world.loc[world['name'] == 'Somaliland', 'iso_a3'] = 'SOM'
world.loc[world['name'] == 'Kosovo', 'iso_a3'] = 'RKS'

clusterDeclaration = predictions.withColumnRenamed("Country Code", "iso_a3").select("iso_a3", "change", "prediction").toPandas()
world_with_cluster = world.join(clusterDeclaration.set_index('iso_a3'), on='iso_a3')

ax = world_with_cluster[world_with_cluster.prediction.notna()].plot(column='prediction', figsize=(13, 8), legend=True, edgecolor='black',
                                                                cmap="Spectral_r", categorical=True)
world_with_cluster[world_with_cluster.change.isna()].plot(color='lightgrey', hatch='///', ax=ax)
plt.title("Co2 Emission Change between 2004 and 2014")
#plt.savefig("figures/co2_change.png")
plt.show()


#scheme = mapclassify.Quantiles(data, k=7)
#geoplot.choropleth(
 #   world, hue=data, scheme=scheme,
  #  figsize=(8, 4)
#)

plt.show()

spark.stop()
