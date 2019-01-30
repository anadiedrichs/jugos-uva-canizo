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
train_control <- trainControl(method="cv", number=10,seeds = mySeeds)

#' ## NAIVE BAYES CLASSIFIER
#' ### 
grid <- as.data.frame(expand.grid(usekernel = c(TRUE, FALSE), fL = 1, adjust = 1))
set.seed(SEED)
model.nb <- train(as.factor(Origen)~., data=data, 
                  trControl=train_control, method="nb",metric=METRIC,
                  tuneGrid=grid) # laplace correction


print(table(predict(model.nb$finalModel,x)$class,y))

plot(model.nb)

#' ## NEURAL NETWORKS
#' ### 
my.grid <- expand.grid(.decay = c(0.5, 0.1), .size = c(5, 6, 7))

set.seed(SEED)
mySeeds <- sapply(simplify = FALSE, 1:11, function(u) sample(10^4, 6))
train_control <- trainControl(method="cv", number=10,seeds = mySeeds)
model.nnet <- train(as.factor(Origen)~., data=data, 
                  trControl=train_control, method="nnet", tuneGrid=my.grid,
                  maxit = 1000, trace = F,metric=METRIC,classPro)

p <- predict(model.nnet$finalModel,x,type="class")

print(table(p,y))

plot(model.nnet)

#' ## KNN o K nearest neighboors

set.seed(SEED)
mySeeds <- sapply(simplify = FALSE, 1:11, function(u) sample(10^4, 12))
train_control <- trainControl(method="cv", number=10,seeds = mySeeds)

model.knn <- train(as.factor(Origen)~., data=data, 
                  trControl=train_control, method="knn",metric=METRIC,
                  preProcess = c("center","scale"), tuneLength = 12) # laplace correction


print(table(predict(model.knn$finalModel,x,type="class"),y))

plot(model.knn)

#' ## Random Forest
#' 
set.seed(SEED)
model.rf <- train(as.factor(Origen)~., data=data, 
                  trControl=train_control, method="rf",metric=METRIC, importance=T)

#'  ### Results of random forest model
print(model.rf)
plot(model.rf)
print(model.rf$finalModel)
#' ### Variable importance
varImp(model.rf)
plot(varImp(model.rf))
#' Predicción en conjunto de testeo test-set
pred <- predict(model.rf,data)
c <- confusionMatrix(as.factor(pred), as.factor(data$Origen),mode = "prec_recall")
print(c)


results <- resamples(list(RF=model.rf,NBayes=model.nb,knn=model.knn,nnet=model.nnet))
# summarize the distributions
summary(results)
# boxplots of results
bwplot(results)
# dot plots of results
dotplot(results)

