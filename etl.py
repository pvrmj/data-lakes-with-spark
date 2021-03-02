import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import TimestampType, DateType
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format


"""
Setup and raed in AWS credentials
"""
config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID'] = config.get('AWS', 'AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = config.get('AWS', 'AWS_SECRET_ACCESS_KEY')


"""
This function creates a Spark session
"""
def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


"""
This function processes the song data files and creates new tables ('songs' and 'artists')
It takes three params:
- spark: the SparkSession
- input_data: path to the data files
- output_data: path to store the outputted results
Finally, the tables are written to S3 in parquet format
"""
def process_song_data(spark, input_data, output_data):
    # get filepath to song data file
    song_data = input_data + "song_data/*/*/*/*.json"
    
    # read song data file
    df = spark.read.json(song_data).dropDuplicates()

    # extract columns to create songs table
    songs_table = df.select(["song_id",
                             "title",
                             "artist_id",
                             "year",
                             "duration"]).distinct()
    songs_table.createOrReplaceTempView("songs")

    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy('year', 'artist_id').parquet(output_data + "songs/songs.parquet", "overwrite")

    # extract columns to create artists table
    artists_table = df.select(["artist_id",
                               "artist_name",
                               "artist_location",
                               "artist_latitude",
                               "artist_longitude"]).distinct()
    artists_table.createOrReplaceTempView("artists")

    # write artists table to parquet files
    artists_table.write.parquet(output_data + "artists/artists.parquet", "overwrite")


"""
This function processes the log data files and creates new tables ('users', 'time' and 'songplays')
It takes three params:
- spark: the SparkSession
- input_data: path to the data files
- output_data: path to store the outputted results
Finally, the tables are written to S3 in parquet format
"""
def process_log_data(spark, input_data, output_data):
    # get filepath to log data file
    log_data = input_data + "log-data/*.json"

    # read log data file
    df = spark.read.json(log_data).dropDuplicates()

    # filter by actions for song plays
    df = df.filter(df.page == "NextSong")

    # extract columns for users table
    users_table = df.select(["userId",
                             "firstName",
                             "lastName",
                             "gender",
                             "level"]).distinct()
    users_table.createOrReplaceTempView("users")

    # write users table to parquet files
    users_table.write.parquet(output_data + "users/users.parquet", "overwrite")

    # create timestamp column from original timestamp column
    get_timestamp = udf(
        lambda x: datetime.fromtimestamp(x / 1000), TimestampType()
    )

    df = df.withColumn("timestamp", get_timestamp("ts"))

    # create datetime column from original timestamp column
    df = df.withColumn("datetime", get_timestamp("ts"))

    # extract columns to create time table
    df = df.withColumn("start_time", get_timestamp("ts"))
    df = df.withColumn("hour", hour("timestamp"))
    df = df.withColumn("day", dayofmonth("timestamp"))
    df = df.withColumn("week", weekofyear("timestamp"))
    df = df.withColumn("month", month("timestamp"))
    df = df.withColumn("year", year("timestamp"))

    time_table = df.select(["start_time",
                            "hour",
                            "day",
                            "week",
                            "month",
                            "year"]).distinct()
    time_table.createOrReplaceTempView("time")

    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy('year', 'month').parquet(output_data + "time/time.parquet", "overwrite")

    # read in song data to use for songplays table
    # And create a log table
    # No need to create a new / separate 'song_df' var here, since we already have the 'songs' table created above
    df.createOrReplaceTempView("log_df")

    # extract columns from joined song and log datasets to create songplays table
    songplays_table = spark.sql("""
        SELECT monotonically_increasing_id() as songplay_id, 
                log_df.start_time as start_time,
                time.year as year,
                time.month as month,
                log_df.userId as user_id,
                log_df.level as level,
                songs.song_id as song_id,
                songs.artist_id as artist_id,
                log_df.sessionId as session_id,
                log_df.location as location,
                log_df.userAgent as user_agent
        FROM log_df
        JOIN songs
            ON log_df.song == songs.title
        JOIN time
            ON log_df.start_time == time.start_time
    """)

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy('year', 'month').parquet(output_data + "songplays/songplays.parquet", "overwrite")


"""
The main functions calls `create_spark_session()`, as well as
`process_song_data` and `process_log_data` (explained above)
`input_data` and `output_data` are also set here
"""
def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://udacity-dend-sparkify-output/"

    process_song_data(spark, input_data, output_data)
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
