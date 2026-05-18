library(quanteda)
library(torch)
library(quanteda.textmodels)

toks <- tokens(data_corpus_moviereviews)
toks <- tokens_remove(toks, stopwords(), padding = TRUE)

toks

ten <- as.tensor(toks)
ten


dataloader(ten)

movie_dataset <- dataset(
  
  name = "movie_dataset",
  
  initialize = function(toks) {
    self$x <- as.tensor(toks)
    self$y <- torch_tensor(toks$sentiment)
  },
  .getitem = function(i) {
    list(x = self$x[i, ], y = self$y[i])
  },
  .length = function() {
    self$y$size()[[1]]
  }
  
)

train_ds <- movie_dataset(toks)
length(train_ds)
train_dl <- train_ds %>% dataloader(batch_size = 16, shuffle = TRUE)



library(torch)

# Define dimensions
seq_len <- 10  # Sequence length
batch_size <- 2 # Batch size
embed_dim <- 16 # Embedding dimension
num_heads <- 4  # Number of attention heads

# Create tensors for Query, Key, and Value
# Format: (Sequence Length, Batch Size, Embedding Dimension)
query <- torch_randn(seq_len, batch_size, embed_dim)
key <- torch_randn(seq_len, batch_size, embed_dim)
value <- torch_randn(seq_len, batch_size, embed_dim)

# Create input projection weights and biases
# In a real model, these are initialized as learnable parameters
in_proj_weight <- torch_randn(3 * embed_dim, embed_dim)
in_proj_bias <- torch_randn(3 * embed_dim)
out_proj_weight <- torch_randn(embed_dim, embed_dim)
out_proj_bias <- torch_randn(embed_dim)

# Run multi-head attention forward pass
output <- nnf_multi_head_attention_forward(
  query = query,
  key = key,
  value = value,
  embed_dim_to_check = embed_dim,
  num_heads = num_heads,
  #bias_k = 0,
  #bias_v = 0,
  in_proj_weight = in_proj_weight,
  in_proj_bias = in_proj_bias,
  out_proj_weight = out_proj_weight,
  out_proj_bias = out_proj_bias,
  training = FALSE
)

# Output is a list containing $attn_output and $attn_output_weights
cat("Output Tensor Shape:", output$attn_output$shape, "\n")

