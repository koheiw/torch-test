# https://mlverse.github.io/luz/articles/examples/text-classification.html
# https://keras.io/examples/nlp/text_classification_from_scratch/
source("settings.R")
library(torch)
library(luz)
# devtools::install_github("quanteda/quanteda", ref = "dev-as.Matrix")
library(quanteda)

corp <- readRDS(file.path(DIR_DATA, "corpus_imdb.RDS"))

toks <- tokens(corp, remove_punct = TRUE) %>% 
  tokens_tolower() %>% 
  #tokens_remove(stopwords("en"), min_nchar = 2) %>% 
  tokens_trim(min_termfreq = 20000, termfreq_type = "rank")

vocab_size <- length(types(toks)) + 1 # maximum number of items in the vocabulary
output_length <- 500 # padding and truncation length.
embedding_dim <- 128 # size of the embedding vectors

movie_dataset <- dataset(
  
  name = "movie_dataset",

  initialize = function(data, output_length) {
    self$x <- as.Matrix(data, output_length)
    self$y <- as.numeric(docvars(data, "sentiment")) - 1
  },
  .getitem = function(i) {
    list(x = as.integer(self$x[i,]) + 1L, 
         y = self$y[i])
  },
  .length = function() {
    length(self$y)
  }
  
)

xtoks <- as.tokens_xptr(toks)
train_ds <- movie_dataset(tokens_subset(xtoks, split == "train"), output_length)
test_ds <- movie_dataset(tokens_subset(xtoks, split == "test"), output_length)

# ----------------------

model <- nn_module(
    initialize = function(vocab_size, embedding_dim) {
        self$embedding <- nn_sequential(
            nn_embedding(num_embeddings = vocab_size, embedding_dim = embedding_dim),
            nn_dropout(0.5)
        )
        
        self$convs <- nn_sequential(
            nn_conv1d(embedding_dim, 128, kernel_size = 7, stride = 3, padding = "valid"),
            nn_relu(),
            nn_conv1d(128, 128, kernel_size = 7, stride = 3, padding = "valid"),
            nn_relu(),
            #nn_adaptive_max_pool2d(c(128, 1)) # reduces the length dimension
            nn_adaptive_max_pool1d(1) # reduces the length dimension
        )
        
        self$classifier <- nn_sequential(
            nn_flatten(),
            nn_linear(128, 128),
            nn_relu(),
            nn_dropout(0.5),
            nn_linear(128, 1)
        )
    },
    forward = function(x) {
        emb <- self$embedding(x)
        out <- emb$transpose(2, 3) %>% 
            self$convs() %>% 
            self$classifier()
        # we drop the last so we get (B) instead of (B, 1)
        #out$squeeze(2)
        out$squeeze(-1)
    }
)

# test the model for a single example batch
# m <- model(vocab_size, embedding_dim)
# x <- torch_randint(1, 20000, size = c(32, 500), dtype = "int")
# m(x)

# ----------------------

fitted_model <- model %>% 
    setup(
        loss = nnf_binary_cross_entropy_with_logits,
        optimizer = optim_adam,
        metrics = luz_metric_binary_accuracy_with_logits()
    ) %>% 
    set_hparams(vocab_size = vocab_size, embedding_dim = embedding_dim) %>% 
    fit(train_ds, epochs = 3)

# ----------------------

fitted_model %>% evaluate(test_ds)
