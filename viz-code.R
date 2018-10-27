library(ggplot2)

setwd('~/repos/pliny-data/')

combined <- data.frame(word = character(0), count = numeric(0), book = numeric(0))

for (i in 1:9) {
  df <- read.csv(paste("book_", i, ".csv", sep=''))
  df$book <- as.factor(i)
  df$frequency <- df$frequency / sum(df$frequency) * 100
  df <- df[1:15,]
  combined <- rbind(combined, df)
}

ggplot(combined, aes(word, frequency)) +
  geom_bar(aes(fill=book), position="dodge", stat="identity")


