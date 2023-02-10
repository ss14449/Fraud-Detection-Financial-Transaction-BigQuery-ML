SELECT type, isFraud, count(*) as cnt
FROM `finance.fraud_data`
GROUP BY isFraud, type
ORDER BY type;

SELECT isFraud, count(*) as cnt
FROM `finance.fraud_data`
WHERE type in ("CASH_OUT", "TRANSFER")
GROUP BY isFraud;

-- top10 maximum amount of transactions
SELECT *
FROM `finance.fraud_data`
ORDER BY amount desc
LIMIT 10;

-- Creating a new table for machine learning
CREATE OR REPLACE TABLE finance.fraud_data_sample AS
SELECT
      type,
      amount,
      nameOrig,
      nameDest,
      oldbalanceOrg as oldbalanceOrig,  #standardize the naming.
      newbalanceOrig,
      oldbalanceDest,
      newbalanceDest,
# add new features:
      if(oldbalanceOrg = 0.0, 1, 0) as origzeroFlag,
      if(newbalanceDest = 0.0, 1, 0) as destzeroFlag,
      round((newbalanceDest-oldbalanceDest-amount)) as amountError,
      generate_uuid() as id,        #create a unique id for each transaction.
      isFraud
FROM finance.fraud_data
WHERE
# filter unnecessary transaction types:
      type in("CASH_OUT","TRANSFER") AND
# undersample:
      (isFraud = 1 or (RAND()< 10/100));  # select 10% of the non-fraud cases

-- creating test dataset
CREATE OR REPLACE TABLE finance.fraud_data_test AS
SELECT *
FROM finance.fraud_data_sample
where RAND() < 20/100;

-- creating validation and training dataset
CREATE OR REPLACE TABLE finance.fraud_data_model AS
SELECT
*
FROM finance.fraud_data_sample  
EXCEPT distinct select * from finance.fraud_data_test;

-- creating a model
CREATE OR REPLACE MODEL
  finance.model_unsupervised OPTIONS(model_type='kmeans', num_clusters=5) AS
SELECT
  amount, oldbalanceOrig, newbalanceOrig, oldbalanceDest, newbalanceDest, type, origzeroFlag, destzeroFlag, amountError
  FROM
  `finance.fraud_data_model`;

-- scoring the test data

SELECT
  centroid_id, sum(isfraud) as fraud_cnt,  count(*) total_cnt
FROM
  ML.PREDICT(MODEL `finance.model_unsupervised`,
    (
    SELECT *
    FROM  `finance.fraud_data_test`))
group by centroid_id
order by centroid_id;

-- create a logistic regression supervised learning model
CREATE OR REPLACE MODEL
  finance.model_supervised_initial
  OPTIONS(model_type='LOGISTIC_REG', INPUT_LABEL_COLS = ["isfraud"]
  )
AS
SELECT
type, amount, oldbalanceOrig, newbalanceOrig, oldbalanceDest, newbalanceDest, isFraud
FROM finance.fraud_data_model;

-- checking the feature importance of the supervised learning model
SELECT
  *
FROM
  ML.WEIGHTS(MODEL `finance.model_supervised_initial`,
    STRUCT(true AS standardize));

-- creating a new model gradient boost
CREATE OR REPLACE MODEL
finance.model_supervised_initial
OPTIONS(model_type='BOOSTED_TREE_CLASSIFIER', INPUT_LABEL_COLS = ["isfraud"]
)
AS
SELECT
type, amount, oldbalanceOrig, newbalanceOrig, oldbalanceDest, newbalanceDest, isFraud
FROM finance.fraud_data_model;

-- evaluating the supervised learning models
CREATE OR REPLACE TABLE finance.table_perf AS
SELECT "Initial_reg" as model_name, *
FROM ML.EVALUATE(MODEL `finance.model_supervised_initial`, (
SELECT *
FROM `finance.fraud_data_model` ));

insert finance.table_perf
SELECT "improved_reg" as model_name, *
FROM  ML.EVALUATE(MODEL `finance.model_supervised_initial`, (
SELECT *
FROM  `finance.fraud_data_model` ));

insert finance.table_perf
SELECT "boosted_tree" as model_name, *
FROM  ML.EVALUATE(MODEL `finance.model_supervised_initial`, (
SELECT *
FROM  `finance.fraud_data_model` ));

-- Predicting fraudulent transactions on test data
SELECT id, label as predicted, isFraud as actual
FROM
  ML.PREDICT(MODEL `finance.model_supervised_initial`,
   (
    SELECT  *
    FROM  `finance.fraud_data_test`
   )
  ), unnest(predicted_isfraud_probs) as p
where p.label = 1 and p.prob > 0.5;
