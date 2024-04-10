#driver init

docker exec spark pip install numpy



#hdfs init

docker exec namenode hdfs dfs -mkdir -p input
docker exec namenode hdfs dfs -mkdir -p models/k_means
docker exec namenode hdfs dfs -mkdir -p models/lreg
docker exec namenode hdfs dfs -mkdir -p results
docker exec namenode hdfs dfs -mkdir -p chk
docker cp input namenode:/input
docker exec namenode hdfs dfs -put /input/emission_data_og.csv input
#docker cp models namenode:/models
#docker exec namenode hdfs dfs -put /models models
