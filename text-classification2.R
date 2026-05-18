# https://mlverse.github.io/luz/articles/examples/text-classification.html
library(torch)
library(luz)
library(quanteda)


toks <- tokens(quanteda.textmodels::data_corpus_moviereviews) %>% 
  tokens_remove(stopwords("en")) %>% 
  tokens_trim(min_termfreq = 5)

vocab_size <- length(types(toks)) # maximum number of items in the vocabulary
output_length <- 500 # padding and truncation length.
embedding_dim <- 128 # size of the embedding vectors

movie_dataset <- dataset(
  
  name = "movie_dataset",
  
  initialize = function(data) {
    self$x <- as.tensor(data, output_length)
    self$y <- torch_tensor(data$sentiment)
  },
  .getitem = function(i) {
    list(x = self$x[i], y = self$y[i])
  },
  .length = function() {
    self$y$size()[[1]]
  }
  
)

train_ds <- movie_dataset(head(toks, 1500))
test_ds <- movie_dataset(tail(toks, 500))

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
            nn_adaptive_max_pool2d(c(128, 1)) # reduces the length dimension
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
        out$squeeze(2)
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

