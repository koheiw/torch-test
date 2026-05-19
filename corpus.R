source("settings.R")
library(quanteda)
library(stringi)

file <- list.files(DIR_RAW, pattern = "[0-9_]+\\.txt", recursive = TRUE)
txt <- sapply(file, function(f) { 
  paste0(readLines(file.path(DIR_RAW, f)), collapse = "\n")
})
txt <- stri_replace_all_fixed(txt, "<br />", " ")
dat <- data.frame(text = txt, file = file)
match <- stri_match_first_regex(file, "([a-z]+)\\/([a-z]+)")
dat$split <- match[,2]
dat$sentiment <- factor(match[,3], levels = c("neg", "pos"))
dat <- subset(dat, !is.na(sentiment))

corp <- corpus(dat, docid_field = "file")
saveRDS(corp, file.path(DIR_DATA, "corpus_imdb.RDS"))
