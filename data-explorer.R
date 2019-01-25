library(readr)
dataset <- read_csv("resultados_muestras.csv")
#View(resultados_muestras)

library(DataExplorer)
plot_boxplot(dataset,by="Origen")

plot_correlation(dataset[-1])

#create_report(dataset)
library(caret)
featurePlot(x=dataset[-1], y=as.factor(dataset$Origen),  "pairs", auto.key=list(columns=2))



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
