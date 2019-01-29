library(readr)
dataset <- read_csv("resultados_muestras.csv")
#View(resultados_muestras)

library(DataExplorer)
plot_boxplot(dataset,by="Origen")
plot_correlation(dataset[-1])

#create_report(dataset)
library(caret)
featurePlot(x=dataset[,c(2,3,4,5,6)], y=as.factor(dataset$Origen),  "pairs", auto.key=list(columns=2))

featurePlot(x=dataset[,c(7,8,9,10,11)], y=as.factor(dataset$Origen),  "pairs", auto.key=list(columns=2))

library(ggfortify)
library(factoextra)

pca.model <- prcomp(dataset[-1],scale. = TRUE)
#' Valores regresados por la funcion prcomp()

names(pca.model)
#' sdev : es la desviación estándar de los componenest principales
#'  (las raíces cuadradas de los eigenvalues (valores propios))

head(pca.model$sdev)

#' rotación: la matriz de variables "loadings", columnas que son vectores propios (eigenvectors)

head(unclass(pca.model$rotation)[, 1:4])
#' en center y scale se guardan los valores de media y desviación que normalizaron los datos
#' 
print(pca.model$center)
print(pca.model$scale)
#' Para ver con detalle coordenadas, coseno cuadrado y contribuciones
get_pca_ind(pca.model)
#' Por ejemplo para acceder al cos2 de los individuos usar ´get_pca_ind(pca.model)$cos2´
#' Eigenvalues y varianza
summary(pca.model)


fviz_screeplot(pca.model, ncp=6)


fviz_pca_var(pca.model)

fviz_pca_var(pca.model, col.var = "contrib", 
             gradient.cols = c("white", "blue", "red"),
             ggtheme = theme_minimal())
fviz_pca_ind(pca.model, label="none", habillage=dataset$Origen,
             addEllipses=TRUE, ellipse.level=0.95, palette = "Dark2")

autoplot(pca.model,data=dataset,colour = 'Origen')

#' ## Biplot de individuos y variables
#' 
fviz_pca_biplot(pca.model, label = "var", habillage=dataset$Origen,
                addEllipses=TRUE, ellipse.level=0.85,
                ggtheme = theme_minimal())

plot_prcomp(dataset[-1])

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
cluster.stats(d, fit$cluster, fit.k2$cluster) 