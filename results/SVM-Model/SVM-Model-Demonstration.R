# load data to be used
load('~/module-2-group3/data/claims-clean-example.RData')

# select only cleaned text
claims_text <- claims_clean$text_clean

# define preprocessing step
preprocess_layer <- layer_text_vectorization(
  standardize = NULL,
  split = 'whitespace',
  ngrams = NULL,
  max_tokens = 7000,
  output_mode = 'tf_idf'
)

preprocess_layer %>% adapt(claims_text)

# create tfidf matrix
claims_tfidf <- as.array(preprocess_layer(claims_text))



svmfit_bclass <- readRDS("~/module-2-group3/results/SVM-Model/svm_bclass.rds")
svmfit_mclass <- readRDS("~/module-2-group3/results/SVM-Model/svm_mclass.rds")

pred.bclass <- predict(svmfit_bclass, newdata = claims_tfidf)
pred.mclass <- predict(svmfit_mclass, newdata = claims_tfidf)

pred_df <- data.frame(
  .id = claims_clean %>% pull(.id),
  bclass.pred = pred.bclass,
  mclass.pred = pred.mclass
)
