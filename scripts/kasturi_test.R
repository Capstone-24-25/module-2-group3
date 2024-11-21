#libraries
library(tidyverse)
library(tidytext)
library(gutenbergr)
library(rvest)


load("~/Group 3/data/claims-clean-example.RData")

# claims_clean data
claims_clean_short <- claims_clean %>% head(10)

# secondary tokenization to obtain bigrams
claims_bigrams <- claims_clean_short %>%
  unnest_tokens(output = bigram, input = text_clean, token = "ngrams", n = 2) 

claims_bigrams$bigram

# tf-idf on the data (term frequency - inverse document frequency)
claims_tfidf <- claims_bigrams %>%
  count(.id, bigram) %>%
  bind_tf_idf(term = bigram,
              document = .id,
              n = n)
claims_tfidf

# pivot_wider
claims_df <- claims_tfidf %>%
  pivot_wider(id_cols = .id, 
              names_from = bigram,
              values_from = tf_idf,
              values_fill = 0)
claims_df
# making matrix numeric 
claims_df$.id<-gsub("url","",as.character(claims_df$.id))

claims_df$.id <- as.numeric(claims_df$.id)
claims_df

# principal component regression on word tokenized data
claims_pcr <- prcomp(claims_df, center = T)
summary(claims_pcr)
