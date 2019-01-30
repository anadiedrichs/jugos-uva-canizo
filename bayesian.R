# status, no terminado, WIP

library(readr)
dataset <- read_csv("resultados_muestras.csv")
dataset$Origen <- as.factor(dataset$Origen)

library(bnlearn)
res = hc(dataset)
# plot the network structure.
plot(res)


res2 = iamb(dataset)
# plot the new network structure.
plot(res2)
