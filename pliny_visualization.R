library(DBI)
library(ggplot2)
library(reshape2)
library(RColorBrewer)
con <- dbConnect(RSQLite::SQLite(), '~/repos/pliny-data/groupings.db')
groupings <- dbReadTable(con, "letters")

mysql_con <- dbConnect(RMySQL::MySQL(), "pliny", unix.socket="/opt/local/var/run/mariadb/mysqld.sock")

kMeansClusters <- dbReadTable(mysql_con, "kmeans_clusters")

ggplot(kMeansClusters, aes(x=as.factor(cluster), fill=as.factor(class))) + geom_bar()

res <- dbSendQuery(mysql_con, "SELECT
                   book,
                   cluster,
                   COUNT(cluster) / (SELECT COUNT(cluster) from kmeans_clusters WHERE k1.book = book) * 100
                   AS percentage
                   FROM kmeans_clusters k1 GROUP BY book, cluster ORDER BY book, cluster")

percentages <- dbFetch(res)

percentages$book <- as.factor(percentages$book)
percentages$cluster <- as.factor(percentages$cluster)

for (i in 1:9) {
  for (j in 1:13) {
    if (nrow(percentages[percentages$book == i & percentages$cluster == j,]) == 0) {
      percentages <- rbind(percentages, data.frame(book=i, cluster=j, percentage=0))
    }
  }
}



ggplot() + 
  geom_line(data=percentages[percentages$cluster == 1,], aes(x=book, y=percentage, color=cluster, group=1)) +
  geom_line(data=percentages[percentages$cluster == 2,], aes(x=book, y=percentage, color=cluster, group=1)) +
  geom_line(data=percentages[percentages$cluster == 4,], aes(x=book, y=percentage, color=cluster, group=1)) +
  geom_line(data=percentages[percentages$cluster == 9,], aes(x=book, y=percentage, color=cluster, group=1)) +
  geom_line(data=percentages[percentages$cluster == 13,], aes(x=book, y=percentage, color=cluster, group=1)) +
  scale_color_brewer(palette='Set1')
  
  