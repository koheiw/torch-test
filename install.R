# https://torch.mlverse.org/docs/articles/installation.html?q=debug#pre-built

options(timeout = 600) # increasing timeout is recommended since we will be downloading a 2GB file.
# For Windows and Linux: "cpu", "cu128" are the only currently supported
# For MacOS the supported are: "cpu-intel" or "cpu-m1"
#kind <- "cu128"
kind <- "cpu"
version <- available.packages()["torch","Version"]
options(repos = c(
  torch = sprintf("https://torch-cdn.mlverse.org/packages/%s/%s/", kind, version),
  CRAN = "https://cloud.r-project.org" # or any other from which you want to install the other R dependencies.
))
install.packages("torch")
