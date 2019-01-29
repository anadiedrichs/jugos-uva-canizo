# load data
library(readr)
data <- read_csv("resultados_muestras.csv")
#set seed
library(caret)
SEED <- 1234 # seed semilla para números aleatorios
set.seed(SEED)
x = data[,-1]
y = data$Origen
#index <- sample(1:nrow(data), round(nrow(data) * 0.7))
#train <- data[index,]
#test <- data[-index,]

set.seed(SEED)
mySeeds <- sapply(simplify = FALSE, 1:11, function(u) sample(10^4, 3))

METRIC <- "Accuracy" #Accuracy
train_control <- trainControl(method="cv", number=5,seeds = mySeeds)

#' ## NAIVE BAYES CLASSIFIER
#' ### 
set.seed(SEED)
model.nb <- train(as.factor(Origen)~., data=data, 
                  trControl=train_control, method="nb",metric=METRIC)


table(predict(model.nb$finalModel,x)$class,y)

plot(model.nb)

#' ## NEURAL NETWORKS
#' ### 
#' 
set.seed(SEED)
model.nb <- train(as.factor(Origen)~., data=data, 
                  trControl=train_control, method="nb",metric=METRIC,
                  tuneGrid=data.frame(.fL=1, .usekernel=FALSE)) # laplace correction


table(predict(model.nb$finalModel,x)$class,y)

plot(model.nb)