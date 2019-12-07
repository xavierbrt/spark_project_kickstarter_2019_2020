package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, Dataset, Row, SQLContext, SparkSession}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, IDF, OneHotEncoderEstimator, RegexTokenizer, StopWordsRemover, StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.param.ParamMap


object Trainer {

  def main(args: Array[String]): Unit = {

    /**
      * Initialisation de Spark
      */

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
      //"spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()

    import spark.implicits._  // to use the symbol $


    /**
      * Modèle 1 : Modèle de base, données prepared_trainingset
      */
    println("\n========================== Chargement du dataframe =====================================")
    val df = loadDataFrame(spark, "src/main/resources/prepared_trainingset/")
    val Array(training, test) = df.randomSplit(Array(0.9, 0.1), seed=261)

    println("\n========================= Implémentation du pipeline ===================================")
    val (pipeline, cvModel, tokenizer, stopWordsRemover, idf, encoder, assembler, lr) = createPipeline(spark)
    val model = pipeline.fit(training)
    val evaluator = createEvaluator(spark)
    // Sauvegarde du modèle entraîné
    model.write.overwrite().save("src/main/resources/model/spark-logistic-regression-model")

    println("\n=============== Test du modèle (sur les données prepared_trainingset) ==================")
    println("Calcul des prédictions sur les données de test")
    val dfWithSimplePredictions = model.transform(test)
    displayModelResult(1, dfWithSimplePredictions, evaluator)


    /**
      * Modèle 2: Modèle issu de l'optimisation de paramètres, même données
      */
    println("\n========================= Réglage des hyper-paramètres du modèle ===================================")
    println("Recherche des meilleurs paramètres (pour les paramètres minDF et elasticNetParam)")
    // Doc : https://spark.apache.org/docs/2.2.0/ml-tuning.html
    val (validation, paramGrid) = createValidation(spark, model, cvModel, lr, pipeline, evaluator)
    val model_improved = validation.fit(training)
    // Sauvegarde du modèle entraîné
    model_improved.write.overwrite().save("src/main/resources/model/spark-logistic-regression-model-improved")

    println("Calcul des prédictions sur les données de test, avec les paramètres sélectionnés")
    val dfWithPredictionsFirstData = model_improved.transform(test)
    displayModelResult(2, dfWithPredictionsFirstData, evaluator)


    /**
      * Modèle 3: Même modèle, données cleanées par le Preprocessor
      */

    println("\n============= Test du modèle sur les données cleanées par le Preprocessor ===========================")
    val df_cleaned = loadDataFrame(spark, "src/main/resources/dataframe/")
    val Array(training_cleaned, test_cleaned) = df_cleaned.randomSplit(Array(0.9, 0.1), seed=261)
    val validation_cleaned = createValidationCleaned(spark, tokenizer, stopWordsRemover, cvModel, idf, encoder, assembler, lr, paramGrid, evaluator)
    val model_improved_training_cleaned = validation_cleaned.fit(training_cleaned)
    // Sauvegarde du modèle entraîné
    model_improved_training_cleaned.write.overwrite().save("src/main/resources/model/spark-logistic-regression-model-final")

    println("Résultat des prédictions sur les données cleanées précédemment:")
    val dfWithPredictions = model_improved_training_cleaned.transform(test_cleaned)
    displayModelResult(3, dfWithPredictions, evaluator)




    /**
      * Explication du modèle
      */
    println("\n\n========================= Explication du modèle ===================================")
    explainModel(spark, model_improved_training_cleaned, test_cleaned)

    /**
      * Test d'un modèle random forest
      */
    println("\n\n====================== Test d'un modèle randomForest ===============================")
    randomForest(spark, tokenizer, stopWordsRemover, cvModel, idf, encoder, assembler, evaluator, training_cleaned, test_cleaned)

  }




  def loadDataFrame(spark: SparkSession, path: String): DataFrame = {
    val df: DataFrame = spark
      .read
      .option("header", true)
      .option("inferSchema", "true")
      .parquet(path)

    println(s"Nombre de lignes: ${df.count}")
    println(s"Nombre de colonnes: ${df.columns.length}")
    df.show(5)

    return df
  }


  def createPipeline(spark: SparkSession): (Pipeline, CountVectorizer, RegexTokenizer, StopWordsRemover, IDF, OneHotEncoderEstimator, VectorAssembler, LogisticRegression) = {
    println("Création du pipeline et de ses différents stages")
    /**
      * Retraitement des données textuelles
      */
    // Stage 1 : récupérer les mots des textes
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    // Stage 2 : retirer les stop words (liste : StopWordsRemover.loadDefaultStopWords("english"))
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("text_filtered")

    // Stage 3 : computer la partie TF
    val cvModel: CountVectorizer = new CountVectorizer()
      .setInputCol("text_filtered")
      .setOutputCol("cv_features")

    // Stage 4 : computer la partie IDF
    val idf = new IDF()
      .setInputCol("cv_features")
      .setOutputCol("tfidf")

    /**
      * Conversion des variables catégorielles en variables numériques
      */

    //Stage 5 : convertir country2 en quantités numériques
    val indexer_country = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("keep")

    //Stage 6 : convertir currency2 en quantités numériques
    val indexer_currency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    // Stages 7 et 8: One-Hot encoder ces deux catégories
    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed", "currency_indexed"))
      .setOutputCols(Array("country_onehot", "currency_onehot"))

    /**
      * Mise en forme des données sous une forme utilisable par Spark.ML
      */

    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

    /**
      * Création du modèle de classification
      */

    // Stage 10 : créer/instancier le modèle de classification
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(20)

    /**
      * Création du pipeline
      */
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover,
        cvModel, idf, indexer_country, indexer_currency,
        encoder, assembler, lr))

    return (pipeline, cvModel, tokenizer, stopWordsRemover, idf, encoder, assembler, lr)
  }

  def createEvaluator(spark: SparkSession): MulticlassClassificationEvaluator = {
    return new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")
  }


  def createValidation(spark: SparkSession, model: PipelineModel, cvModel: CountVectorizer, lr: LogisticRegression,
                       pipeline: Pipeline, evaluator: MulticlassClassificationEvaluator): (TrainValidationSplit, Array[ParamMap]) = {
    /**
      * Réglage des hyper-paramètres du modèle
      */
    val paramGrid = new ParamGridBuilder()
      .addGrid(cvModel.minDF, Array(5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 75.0, 95.0))
      .addGrid(lr.elasticNetParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .build()

    val validation = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    return (validation, paramGrid)
  }

  def createValidationCleaned(spark: SparkSession, tokenizer: RegexTokenizer, stopWordsRemover: StopWordsRemover,
                              cvModel: CountVectorizer, idf: IDF, encoder: OneHotEncoderEstimator, assembler: VectorAssembler,
                              lr: LogisticRegression, paramGrid: Array[ParamMap], evaluator: MulticlassClassificationEvaluator): TrainValidationSplit = {
    //Stage 5 : convertir country en quantités numériques
    val indexer_country_cleaned = new StringIndexer()
      .setInputCol("country")
      .setOutputCol("country_indexed")
      .setHandleInvalid("keep")

    //Stage 6 : convertir currency en quantités numériques
    val indexer_currency_cleaned = new StringIndexer()
      .setInputCol("currency")
      .setOutputCol("currency_indexed")

    val pipeline_cleaned = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover,
        cvModel, idf, indexer_country_cleaned, indexer_currency_cleaned,
        encoder, assembler, lr))

    val validation_cleaned = new TrainValidationSplit()
      .setEstimator(pipeline_cleaned)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    return validation_cleaned
  }


  def explainModel(spark: SparkSession, model_improved_training_cleaned: TrainValidationSplitModel, test_cleaned: DataFrame){
    val model_lr = model_improved_training_cleaned.bestModel.asInstanceOf[PipelineModel].stages.last.asInstanceOf[LogisticRegressionModel]
    // Extract the attributes of the input (features)
    val schema = model_improved_training_cleaned.transform(test_cleaned).schema
    val featureAttrs = AttributeGroup.fromStructField(schema(model_lr.getFeaturesCol)).attributes.get
    val features = featureAttrs.map(_.name.get)

    // Add "(Intercept)" to list of feature names if the model was fit with an intercept
    val featureNames: Array[String] = if (model_lr.getFitIntercept) {
      Array("(Intercept)") ++ features
    } else {
      features
    }

    // Get array of coefficients
    val lrModelCoeffs = model_lr.coefficients.toArray
    val coeffs = if (model_lr.getFitIntercept) {
      lrModelCoeffs ++ Array(model_lr.intercept)
    } else {
      lrModelCoeffs
    }
    val coeffs_abs = coeffs.map(num => Math.abs(num))

    // Print feature names & coefficients together
    println("Liste des variables impactant le plus le modèle (20 premières):")
    println("Variables\t\t\t|Coefficients|")
    featureNames.zip(coeffs_abs).sortBy(_._2)(Ordering[Double].reverse).take(20).foreach { case (feature, coeff) =>
      println(s"$feature\t$coeff")
    }

    //println("Affichage de l'ensemble des paramètres du modèle")
    //model_improved_training_cleaned.bestModel.asInstanceOf[PipelineModel].stages.foreach(stage => println(stage.extractParamMap))
  }


  def displayModelResult(modelNumber: Int, df: DataFrame, evaluator: MulticlassClassificationEvaluator) {
    df.groupBy("final_status", "predictions").count.show()
    println(s"Modèle ${modelNumber} -- F1 score: ${evaluator.evaluate(df)}")
  }

  def randomForest(spark: SparkSession, tokenizer: RegexTokenizer, stopWordsRemover: StopWordsRemover,
  cvModel: CountVectorizer, idf: IDF, encoder: OneHotEncoderEstimator, assembler: VectorAssembler,
  evaluator: MulticlassClassificationEvaluator, training: DataFrame, test: DataFrame){
    val indexer_country = new StringIndexer()
      .setInputCol("country")
      .setOutputCol("country_indexed")
      .setHandleInvalid("skip")

    val indexer_currency = new StringIndexer()
      .setInputCol("currency")
      .setOutputCol("currency_indexed")
      .setHandleInvalid("skip")

    val rf = new RandomForestClassifier()
      .setLabelCol("final_status")
      .setFeaturesCol("features")
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setNumTrees(10)

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover,
        cvModel, idf, indexer_country, indexer_currency,
        encoder, assembler, rf))

    val model = pipeline.fit(training)
    val df = model.transform(test)

    df.groupBy("final_status", "predictions").count.show()
    print(s"F1 score du modèle random forest: ${evaluator.evaluate(df)}")
  }


}
