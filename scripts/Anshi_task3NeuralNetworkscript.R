library(tidyverse)
library(tidymodels)
library(tidytext)
library(keras)
library(tensorflow)

setwd("~/Downloads/module-2-group3")

load("~/Downloads/module-2-group3/data/claims-clean-example.RData")

# Model Attempt 1: Simple RNN with one hot encoding for the dataset and an embedding layer
claims_clean2 <- claims_clean %>% select(bclass, text_clean)

library(tokenizers)
claims_clean3 <- claims_clean2 %>% mutate(tokens = (claims_clean2$text_clean %>% tokenize_words())) %>% select(bclass, tokens)

vocab_size <- claims_clean3 %>% rowwise() %>% mutate(n = length(tokens))
max_size <- max(vocab_size$n)
vocab_size <- sum(vocab_size$n)

# encoding variables
text <- claims_clean2 %>% pull(text_clean)
encod_corp <- list()

# Loop through each document and encode using text_one_hot
for (i in 1:length(text)) {
  # Apply text_one_hot to encode the document (50 is the size of the vocabulary)
  encoded_doc <- text_one_hot(text[i], 833732)
  
  # Append the encoded document to the list
  encod_corp[[i]] <- encoded_doc
}
claims_clean2$encoded <- encod_corp

x_train <- pad_sequences(claims_clean2$encoded, maxlen = 11299, padding="post", value = 0)
claims_clean2 <- claims_clean2 %>% mutate(bclass = ifelse(bclass == "Relevant claim content", 1, 0))
y_train <- claims_clean2$bclass

rnn_model <- keras_model_sequential()
rnn_model %>%
  layer_embedding(input_dim = vocab_size, output_dim = 128) %>% 
  layer_simple_rnn(units = 64, dropout = 0.2, 
                   recurrent_dropout = 0.2) %>% 
  layer_dense(units = 1, activation = 'sigmoid')

rnn_model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

history <- rnn_model %>% fit(
  x_train, y_train,
  epochs = 10,
  validation_split = 0.2
)


# Accuracy was hovering around .51

# Model Attempt 2: TF-IDF data into a LSTM model with Overfitting restraints

clean <- claims_clean %>% select(.id, bclass, text_clean)


set.seed(10435467)
partitions <- clean %>%
  mutate(text_clean = str_trim(text_clean)) %>%
  filter(str_length(text_clean) > 5) %>%
  initial_split(prop = 0.8)

train_dtm <- training(partitions) %>%
  unnest_tokens(output = 'token', 
                input = text_clean) %>%
  group_by(.id, bclass) %>%
  count(token) %>%
  bind_tf_idf(term = token, 
              document = .id, 
              n = n) %>%
  pivot_wider(id_cols = c(.id, bclass), 
              names_from = token, 
              values_from = tf_idf,
              values_fill = 0) %>%
  ungroup()

x_train2 <- train_dtm %>%
  select(-bclass, -.id) %>%
  as.matrix()

y_train2 <- train_dtm %>% 
  pull(bclass) %>%
  factor() %>%
  as.numeric() - 1

X_train_reshaped <- array(x_train2, dim = c(nrow(x_train2), 1, ncol(x_train2)))

lstm_model <- keras_model_sequential() %>%
  layer_lstm(units = 6, input_shape = c(1, ncol(x_train2)), dropout = 0.2) %>%
  layer_batch_normalization() %>%
  layer_dense(units = 1, activation = 'sigmoid')  # Output layer (binary classification)

summary(lstm_model)

lstm_model %>%
  compile(
    loss = 'binary_crossentropy',
    optimizer = optimizer_adam(lr = 0.0001),
    metrics = list('accuracy'))

early_stopping <- callback_early_stopping(monitor = 'val_loss', patience = 3, restore_best_weights = TRUE)

history2 <- lstm_model %>%
  fit(x = X_train_reshaped,
      y = y_train2,
      epochs = 15,
      validation_split = 0.2, 
      callbacks = list(early_stopping))


# Finding Specificity, Sensitivity and Accuracy on test set
test_dtm <- testing(partitions) %>%
  unnest_tokens(output = 'token', 
                input = text_clean) %>%
  group_by(.id, bclass) %>%
  count(token) %>%
  bind_tf_idf(term = token, 
              document = .id, 
              n = n) %>%
  pivot_wider(id_cols = c(.id, bclass), 
              names_from = token, 
              values_from = tf_idf,
              values_fill = 0) %>%
  ungroup()

x_test <- test_dtm %>%
  select(-bclass, -.id) %>%
  as.matrix()

zeros_needed <- 34170 - ncol(x_test)
zeros_matrix <- matrix(0, nrow = nrow(x_test), ncol = zeros_needed)
new_matrix <- cbind(x_test, zeros_matrix)

X_test_reshaped <- array(new_matrix, dim = c(nrow(new_matrix), 1, ncol(new_matrix)))

predictions <- predict(lstm_model, X_test_reshaped) %>%
  as.numeric()

pred_classes <- as.numeric(ifelse(predictions > 0.5, 1, 0))

y_test <- test_dtm %>% 
  pull(bclass) %>%
  factor() %>%
  as.numeric() - 1

df <- data.frame(y_test, pred_classes) %>% mutate(
                            group = factor(y_test),
                        pred.group = factor(pred_classes)) %>% select(group, pred.group)

panel_fn <- metric_set(sensitivity, specificity, accuracy)

df %>%
  panel_fn(truth = group,
           estimate = pred.group,
           event_level = 'second')

save_model_tf(lstm_model, "results/task3-NN-model")