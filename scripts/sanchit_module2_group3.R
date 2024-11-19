# Required libraries
library(rvest)
library(stringr)
library(qdapRegex)
library(tidyr)
library(dplyr)
library(tidymodels)
library(textrecipes)
library(tidytext)
library(yardstick)

# Set working directory
setwd("~/GitHub/module-2-group3/data")

# Load datasets
load('claims-raw.RData')  # Load raw data
load('claims-test.RData')  # Load test data

# Parsing function for paragraphs and headers
parse_fn_para_header <- function(html) {
  text <- read_html(html) %>%
    {c(
      html_elements(., 'p') %>% html_text2(),
      html_elements(., 'h1, h2, h3') %>% html_text2()
    )} %>%
    str_c(collapse = ' ') %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', 
                            '[[:punct:]]', 
                            'nbsp', 
                            '[[:digit:]]', 
                            '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
  return(text)
}

# Preprocessing function
parse_data <- function(claims_raw, parse_fn) {
  claims_raw %>%
    filter(str_detect(text_tmp, '<!')) %>%  # Ensure valid HTML
    rowwise() %>%
    mutate(text_clean = parse_fn(text_tmp)) %>%
    ungroup() %>%
    unnest(cols = c(text_clean))  # Flatten nested structure if necessary
}

# Clean and process training data
claims_clean_para_header <- parse_data(claims_raw, parse_fn_para_header)
save(claims_clean_para_header, file = "claims-clean.RData")

# NLP feature extraction
nlp_fn <- function(claims_clean) {
  if ("bclass" %in% colnames(claims_clean)) {
    # For training data with 'bclass'
    claims_clean %>%
      unnest_tokens(token, text_clean) %>%
      count(.id, bclass, token, sort = TRUE) %>%
      bind_tf_idf(token, .id, n)
  } else {
    # For test data without 'bclass'
    claims_clean %>%
      unnest_tokens(token, text_clean) %>%
      count(.id, token, sort = TRUE) %>%
      bind_tf_idf(token, .id, n)
  }
}

#  TF-IDF for training data
claims_tfidf_para_header <- nlp_fn(claims_clean_para_header)

# Logistic PCA function
pca_logistic_fn <- function(data) {
  pca_rec <- recipe(bclass ~ tf_idf, data = data) %>%
    step_normalize(all_numeric_predictors()) %>%
    step_pca(all_numeric_predictors(), num_comp = 10)
  
  log_model <- logistic_reg() %>%
    set_engine("glm")
  
  workflow() %>%
    add_recipe(pca_rec) %>%
    add_model(log_model) %>%
    fit(data = data)
}

# Train Logistic PCA model
log_model_para_header <- pca_logistic_fn(claims_tfidf_para_header)

# Clean and process test data
claims_test_clean_para_header <- parse_data(claims_test, parse_fn_para_header)
claims_test_tfidf_para_header <- nlp_fn(claims_test_clean_para_header)

# Generate predictions for the test set
predictions_para_header <- predict(log_model_para_header, claims_test_tfidf_para_header)
save(predictions_para_header, file = "predictions-para-header.RData")


# Split data into training and validation sets
set.seed(123)  
data_split <- initial_split(claims_tfidf_para_header, prop = 0.7, strata = bclass)
train_data <- training(data_split)
val_data <- testing(data_split)

# Training model
log_model_para_header <- pca_logistic_fn(train_data)

# Make predictions 
val_predictions <- predict(log_model_para_header, val_data, type = "class")

# metrics
val_metrics <- val_data %>%
  bind_cols(predictions = val_predictions) %>%
  metrics(truth = bclass, estimate = .pred_class)

print(val_metrics)

































