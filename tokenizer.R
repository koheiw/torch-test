source("settings.R")
library(tok)
library(quanteda)

corp <- readRDS(file.path(DIR_DATA, "corpus_imdb.RDS"))
txt <- corp[[1]] %>% 
  stringr::str_to_lower() %>% 
  stringr::str_replace_all("<br />", " ") %>% 
  stringr::str_remove_all("[:punct:]")

#(tok <- tokenizer$new(model_bpe$new()))
(tok <- tokenizer$new(model_unigram$new()))
#(tok <- tokenizer$from_pretrained("gpt2"))

tok <- tokenizer$from_file("imdb/tokenizer_unigram-20000.json")
enc <- tok$encode(txt)
sapply(enc$ids, tok$decode)
