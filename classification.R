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