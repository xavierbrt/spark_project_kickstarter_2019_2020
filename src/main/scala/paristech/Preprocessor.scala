package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import org.apache.spark.sql.functions._

object Preprocessor {

  def main(args: Array[String]): Unit = {

    /**
      * Initialisation de Spark
      */

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()

    import spark.implicits._  // to use the symbol $


    /**
      * Chargement du dataframe
      */

    // Chargement du fichier train dans un dataframe
    val df: DataFrame = spark
        .read
        .option("header", true)
        .option("inferSchema", "true")
        .option("quote", "\"")
        .option("escape", "\"")
        .csv("./data/train_clean.csv")

    println("\n========================= Chargement du fichier train_clean.csv ===================================")
    println(s"Nombre de lignes: ${df.count}")
    println(s"Nombre de colonnes: ${df.columns.length}")


    /**
      * Cast des colonnes, suppression de colonnes
      */

    // Transformation du type des colonnes
    val dfCasted: DataFrame = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline" , $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))

    dfCasted
      .select("goal", "deadline", "state_changed_at", "created_at", "launched_at", "backers_count", "final_status")
      .describe()
      .show

    println("Structure du dataframe: ")
    dfCasted.printSchema()

    // On enlève la colonne disable_communication, qui contient peu de données et les colonnes backers_count et state_changed_at qui sont des fuites du futur
    val df2: DataFrame = df.drop("disable_communication", "backers_count", "state_changed_at")


    /**
      * Vérification des données
      */

    // Ce n'est plus la peine de cleaner les colonnes currency et country, qui sont bien remplies
    // (elles étaient mal remplies au départ car les virgules n'étaient pas échappées à la lecture du fichier csv)
    println("================= Description des données pour chercher les anomalies =======================")
    df2.select("goal", "country", "currency", "deadline","created_at", "launched_at","final_status")
      .describe()
      .show

    df2.groupBy("final_status").count.show()  // 0: fail, 1: success


    /**
      * Retraitement de colonnes
      */

    // create column "days_campaign" with the (truncated) number of days between launch time and deadline
    val df3: DataFrame = df2.withColumn("days_campaign", datediff(from_unixtime($"deadline"), from_unixtime($"launched_at")))

    // create column "hours_prepa" with the number of hours between creation time and launch time
    val df4: DataFrame = df3.withColumn("hours_prepa", round(($"launched_at" - $"created_at")/3600,3))

    // create column "launched_month" with the month of the launched date
    val df4b: DataFrame = df4.withColumn("launched_month", month(from_unixtime($"launched_at")))

    val df5: DataFrame = df4b.drop("launched_at", "created_at", "deadline")

    val df6: DataFrame = df5.withColumn("name", lower($"name"))
      .withColumn("desc", lower($"desc"))
      .withColumn("keywords", regexp_replace(lower($"keywords"), "-", " "))

    val df7: DataFrame = df6.withColumn("text", concat_ws(" ",$"name", $"desc",$"keywords"))
    val df8: DataFrame = df7.drop("name", "desc", "keywords")


    /**
      * Vérification des valeurs nulles
      */

    println("========================= Vérification des valeurs nulles ===================================")
    // There is no null values. To verify :
    println("project_id:", df8.filter("project_id is null").count())
    println("goal:", df8.filter("goal is null").count())
    println("country:", df8.filter("country is null").count())
    println("currency:", df8.filter("currency is null").count())
    println("final_status:", df8.filter("final_status is null").count())
    println("days_campaign:", df8.filter("days_campaign is null").count())
    println("hours_prepa:", df8.filter("hours_prepa is null").count())
    println("text:", df8.filter("text is null").count())
    println("launched_month:", df8.filter("launched_month is null").count())


    /**
      * Export du dataframe
      */

    println("\n========================= Export du dataframe ===================================")
    println(s"Nombre de lignes: ${df8.count}")
    println(s"Nombre de colonnes: ${df8.columns.length}")
    df8.show()

    df8.write.mode("overwrite").parquet("./data/dataframe")
    println("Dataframe exporté dans le dossier data/dataframe")

  }
}
