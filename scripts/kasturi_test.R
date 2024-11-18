library(tidyverse)
library(tidytext)
library(gutenbergr)
library(rvest)

#test



load("~/Documents/Fall 2024/PSTAT 197A/module-2-group3/data/claims-raw.RData")
claims_raw %>% head()
claims_clean %>% head()
claims_clean$text_clean

claims_bigrams <- claims_clean %>%
  unnest_tokens(bigram, text_clean, token = "ngrams", n = 2) 

claims_bigrams$bigram
