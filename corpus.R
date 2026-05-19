source("settings.R")
library(quanteda)

file <- list.files(DIR_RAW, pattern = "[0-9_]+\\.txt", recursive = TRUE)
txt <- sapply(file, function(f) { 
  paste0(readLines(file.path(DIR_RAW, f)), collapse = "\n")
})
dat <- data.frame(text = txt, file = file)
match <- stringi::stri_match_first_regex(file, "([a-z]+)\\/([a-z]+)")
dat$split <- match[,2]
dat$sentiment <- match[,3]
dat$doc_id <- file
corp <- corpus(dat)

saveRDS(corp, file.path(DIR_DATA, "corpus_imdb.RDS"))