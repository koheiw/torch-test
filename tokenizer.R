source("settings.R")
library(tok)
library(quanteda)

corp <- readRDS(file.path(DIR_DATA, "corpus_imdb.RDS"))
txt <- corp %>% 
  stringr::str_to_lower() %>% 
  stringr::str_replace_all("<br />", " ") %>% 
  stringr::str_remove_all("[:punct:]")

tok <- tokenizer$from_file(file.path(DIR_RAW, "/tokenizer-unigram-20000.json"))
lis <- lapply(txt, function(x) tok$encode(x)$ids)

#(tok <- tokenizer$new(model_bpe$new()))
(tok <- tokenizer$new(model_unigram$new()))
#(tok <- tokenizer$from_pretrained("gpt2"))

enc <- tok$encode(txt[1])
sapply(enc$ids, tok$decode)
