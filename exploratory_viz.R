library(DBI)
library(ggplot2)

con <- dbConnect(RSQLite::SQLite(), "~/repos/pliny-data/groupings.db")
dbListTables(con)

df <- dbReadTable(con, "letters")


ggplot(df) + 
  geom_bar(aes(as.factor(grouping), fill = as.factor(book))) + 
  scale_fill_brewer(palette = "Set1", name="book") +
  xlab('cluster') + ylab('count') + ggtitle('k-means clusters in Pliny, Letters 1-9')

res <- dbSendQuery(con, "SELECT printf('%d.%d', book, letter) as letter FROM letters WHERE grouping != 13 AND book = 9")
not_group_13 <- dbFetch(res)
