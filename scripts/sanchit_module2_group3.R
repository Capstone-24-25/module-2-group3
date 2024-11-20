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

# PART 1: Word Tokenization and Logistic PCA Regression

# Clean and process training data
claims_clean_para_header <- parse_data(claims_raw, parse_fn_para_header)
#save(claims_clean_para_header, file = "claims-clean.RData")

# NLP feature extraction
nlp_fn <- function(claims_clean) {
  if ("bclass" %in% colnames(claims_clean)) {
    claims_clean %>%
      unnest_tokens(token, text_clean) %>%
      count(.id, bclass, token, sort = TRUE) %>%
      bind_tf_idf(token, .id, n)
  } else {
    claims_clean %>%
      unnest_tokens(token, text_clean) %>%
      count(.id, token, sort = TRUE) %>%
      bind_tf_idf(token, .id, n)
  }
}

# Generate TF-IDF for word tokens
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

# Generate predictions
predictions_para_header <- predict(log_model_para_header, claims_test_tfidf_para_header)
#save(predictions_para_header, file = "predictions-para-header.RData")

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



## part 2 

# Bigram Tokenization and Combined Logistic Regression

# Bigram tokenization and NLP feature extraction function
nlp_fn_bigrams <- function(claims_clean) {
  if ("bclass" %in% colnames(claims_clean)) {
    claims_clean %>%
      unnest_tokens(token, text_clean, token = "ngrams", n = 2) %>%
      count(.id, bclass, token, sort = TRUE) %>%
      bind_tf_idf(token, .id, n)
  } else {
    claims_clean %>%
      unnest_tokens(token, text_clean, token = "ngrams", n = 2) %>%
      count(.id, token, sort = TRUE) %>%
      bind_tf_idf(token, .id, n)
  }
}

# Generate TF-IDF for bigrams
claims_tfidf_bigrams <- nlp_fn_bigrams(claims_clean_para_header)

# Handle missing TF-IDF values
claims_tfidf_bigrams <- claims_tfidf_bigrams %>%
  mutate(tf_idf = ifelse(is.na(tf_idf), 0, tf_idf))

# Train Logistic PCA model for bigram tokens
log_model_bigrams <- pca_logistic_fn(claims_tfidf_bigrams)

# Generate log-odds from word-based model
# Step 1: Predict probabilities
log_odds_word_probs <- predict(log_model_para_header, claims_tfidf_bigrams, type = "prob")

# Step 2: Check column names of probabilities
print(colnames(log_odds_word_probs))

# Step 3: Calculate log-odds using correct column names
log_odds_word <- log_odds_word_probs %>%
  mutate(log_odds = log(`.pred_N/A: No relevant content.` / `.pred_Relevant claim content`))  # Replace with your column names

# PCA for bigram TF-IDF data
pca_rec_bigrams <- recipe(bclass ~ tf_idf, data = claims_tfidf_bigrams) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_numeric_predictors(), num_comp = 10)

bigrams_pca <- pca_rec_bigrams %>%
  prep(training = claims_tfidf_bigrams) %>%
  bake(new_data = NULL)


# Clean bigrams PCA data to exclude redundant `bclass`
bigrams_pca_clean <- bigrams_pca %>%
  select(-bclass)  # Exclude `bclass` if it exists

# Select only log-odds column
log_odds_clean <- log_odds_word %>%
  select(log_odds)

# Combine features without duplicate `bclass`
combined_data <- bigrams_pca_clean %>%
  bind_cols(log_odds_clean) %>%
  bind_cols(claims_tfidf_bigrams %>% select(bclass))  # Add `bclass` from one source


# Combine features for the final model
combined_data <- bigrams_pca %>%
  bind_cols(log_odds_word %>% select(log_odds))
##  %>% bind_cols(claims_tfidf_bigrams %>% select(bclass))
colnames(combined_data)

# Split combined data into train and validation sets
set.seed(123)
data_split_combined <- initial_split(combined_data, prop = 0.7, strata = bclass)
train_combined <- training(data_split_combined)
val_combined <- testing(data_split_combined)

# Create the recipe for the logistic regression model
final_recipe <- recipe(bclass ~ ., data = train_combined) %>%
  step_normalize(all_numeric_predictors())

# Create the workflow
final_workflow <- workflow() %>%
  add_recipe(final_recipe) %>%
  add_model(logistic_reg() %>% set_engine("glm"))

# Fit the model
final_model <- final_workflow %>%
  fit(data = train_combined)


# Validation predictions for the combined model
val_predictions_combined <- predict(final_model, val_combined, type = "class")

# Metrics for the combined model
val_metrics_combined <- val_combined %>%
  bind_cols(predictions = val_predictions_combined) %>%
  metrics(truth = bclass, estimate = .pred_class)

# Print validation metrics for the combined model
print(val_metrics_combined)
