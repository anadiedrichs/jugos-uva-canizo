library(readr)
dataset <- read_csv("resultados_muestras.csv")


#' Clustering approach 
#' 
data <- scale(dataset[-1])

# K-Means Cluster Analysis
fit <- kmeans(data, 3) # 3 cluster solution
# get cluster means
aggregate(data,by=list(fit$cluster),FUN=mean)
# append cluster assignment
mydata <- data.frame(data, fit$cluster) 

cbind(mydata$fit.cluster,dataset$Origen)

# Ward Hierarchical Clustering
d <- dist(data, method = "euclidean") # distance matrix
fit <- hclust(d, method="ward")
plot(fit) # display dendogram
groups <- cutree(fit, k=3) # cut tree into 5 clusters
# draw dendogram with red borders around the 5 clusters
rect.hclust(fit, k=3, border="red") 

# Ward Hierarchical Clustering with Bootstrapped p values
library(pvclust)
fit <- pvclust(data, method.hclust="ward",
               method.dist="euclidean")
plot(fit) # dendogram with p values
# add rectangles around groups highly supported by the data
pvrect(fit, alpha=.95)

# Model Based Clustering  <-- interesante, chequear después
library(mclust)
fit <- Mclust(data)
plot(fit,what = c("classification")) # plot results

plot(fit,what = c("BIC"))
summary(fit) # display the best model 

mod2 <- MclustDA(data, dataset$Origen, modelType = "EDDA")
summary(mod2) # analizar
plot(mod2, what = "scatterplot")
plot(mod2, what = "classification")
cv <- cvMclustDA(mod2, nfold = 10)

# K-Means Clustering with 3 clusters
fit <- kmeans(data, 3)

# Cluster Plot against 1st 2 principal components

# vary parameters for most readable graph
library(cluster)
clusplot(data, fit$cluster, color=TRUE, shade=TRUE,
         labels=2, lines=0)

# Centroid Plot against 1st 2 discriminant functions
library(fpc)
plotcluster(data, fit$cluster) 

fit.k2 <- kmeans(data, 2)

# Cluster Plot against 1st 2 principal components

# vary parameters for most readable graph
library(cluster)
clusplot(data, fit.k2$cluster, color=TRUE, shade=TRUE,
         labels=2, lines=0)

# Centroid Plot against 1st 2 discriminant functions
library(fpc)
plotcluster(data, fit.k2$cluster) 

# comparamos los dos clusters
# TODO: evaluar las metricas que alli presenta.
cluster.stats(d, fit$cluster, fit.k2$cluster) 