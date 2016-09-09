require("foreign")
require("e1071")
require("FSelector")
require("RWeka")
require("nnet")
require("DMwR")
require("sets")
require("rJava")
require("adabag")
rm(list = ls(all = TRUE))
setwd("C:\\Users\\Mingyao\\Dropbox\\Xiaocong\\Practicum\\BreastCancer")

csTrain <- function(train, numOfClusters) {
  # Kmeans
  fit <- kmeans(train[, -ncol(train)], numOfClusters)
  assignments <- fit$cluster
  centers <- fit$centers
  print(centers)
  train <- cbind(train, assignments)
  for (i in 1:numOfClusters) {
    tmp <- subset(train, assignments==i)
    tmp$assignments <- NULL
    print(nrow(tmp))
    print(table(tmp$V39))
    file <- paste("cluster", i, ".arff", sep="")
    write.arff(tmp, file)
  }
  train$assignments <- NULL
  
  # Train each part
  for (i in 1:numOfClusters) {
    cluster <- read.arff(paste("cluster", i, ".arff", sep="")) 
    c <- table(cluster[,ncol(cluster)])
    print(paste("test set:", c[1], c[2]))
    ratio <- c[1] / c[2]
    
    resulttwo <- 0
    resultone <- 0
  
    if (ratio >= 4.0 && c[1] != 0 && c[2] != 0) { # Imbalanced --------------------
      print("----------------------------------->Imbalanced")
      # Smote
      k <- 4
      smoted <- SMOTE(V39~., train, k=k, perc.over=200, perc.under=0)
      smoted <- rbind(train, smoted)
      smoted <- smoted[sample(nrow(smoted)),]
      smoted <- na.omit(smoted)
      print(paste("Smoted: ", table(smoted[,ncol(smoted)])[1], table(smoted[,ncol(smoted)])[2]))
      
      # C45
      modelone <- J48(V39~., data=smoted)
      pred <- predict(modelone, cluster[,-ncol(cluster)])
      confusion <- table(pred, cluster[,ncol(cluster)])
      TP <- confusion[2,2]
      TN <- confusion[1,1]
      FP <- confusion[2,1]
      FN <- confusion[1,2]
      precision <- TP / (FP + TP)
      recall <- TP / (TP + FN)
      resultone <- 2 * precision * recall / ( precision + recall) 
      print(confusion)
      print(resultone)
      
      # SVM
      obj <- tune.svm(V39~., data=smoted, gamma = 2^(-4:4), cost = 2^(1:10))
      print(paste(obj$best.parameters$gamma, obj$best.parameters$cost))
      modeltwo <- svm(V39~., data=smoted, kernel="radial", gamma = obj$best.parameters$gamma, cost = obj$best.parameters$cost)
      pred <- predict(modeltwo, cluster[,-ncol(cluster)])
      confusion <- table(pred, cluster[,ncol(cluster)])
      TP <- confusion[2,2]
      TN <- confusion[1,1]
      FP <- confusion[2,1]
      FN <- confusion[1,2]
      precision <- TP / (FP + TP)
      recall <- TP / (TP + FN)
      resulttwo <- 2 * precision * recall / ( precision + recall) 
      print(confusion)
      print(resulttwo)
      
    } else { # Balanced -----------------------------------------------
      print("----------------------------------->Balanced")
      # C45
      modelone <- J48(V39~., data=train)
      pred <- predict(modelone, cluster[,-ncol(cluster)])
      confusion <- table(pred, cluster[,ncol(cluster)])
      TP <- confusion[2,2]
      TN <- confusion[1,1]
      FP <- confusion[2,1]
      FN <- confusion[1,2]
      precision <- TP / (FP + TP)
      recall <- TP / (TP + FN)
      resultone <- (TP + TN) / (TP + TN + FP + FN)
      #resultone <- 2 * precision * recall / ( precision + recall) 
      print(confusion)
      print(resultone)
      
      # svm
      obj <- tune.svm(V39~., data=train, gamma = 2^(-4:4), cost = 2^(1:10))
      print(paste(obj$best.parameters$gamma, obj$best.parameters$cost))
      modeltwo <- svm(V39~., data=train, kernel="radial", gamma = obj$best.parameters$gamma, cost = obj$best.parameters$cost)
      pred <- predict(modeltwo, cluster[,-ncol(cluster)])
      confusion <- table(pred, cluster[,ncol(cluster)])
      TP <- confusion[2,2]
      TN <- confusion[1,1]
      FP <- confusion[2,1]
      FN <- confusion[1,2]
      precision <- TP / (FP + TP)
      recall <- TP / (TP + FN)
      resulttwo <- (TP + TN) / (TP + TN + FP + FN)
      #resulttwo <- 2 * precision * recall / ( precision + recall)
      print(confusion)
      print(resulttwo)
    }
  
    if (is.nan(resultone) || is.nan(resulttwo)) {
      model <- modelone
      .jcache(modelone$classifier)
      print("Model 1 selected")
    }
    else if (resulttwo > resultone) {
      model <- modeltwo
      print("Model 2 selected")
    } else {
      model <- modelone
      .jcache(modelone$classifier)
      print("Model 1 selected")
    }
    save(model, file=paste("my_model", i, ".rda", sep = ""))
  }
  
  return(centers)
}

eucdist <- function(x1, x2) { 
  sqrt(sum((x1 - x2) ^ 2))
}

splitTestset <- function(test, centers) {
  for (i in 1:nrow(centers)) {
    path <- paste("splitedtest", i, ".csv", sep="")
    if (file.exists(path))
      file.remove(path)
  }
  
  assignment <- list()
  for (i in 1:nrow(test)) {
    record <- test[i, -ncol(test)]
    min <- 100.0
    for (j in 1:nrow(centers)) {
      dist <- eucdist(record, centers[j,])
      if (dist < min) {
        min <- dist
        assignment[i] <- j
      }
    }
  }
  
  assignment <- unlist(assignment)
  newtest <- cbind(test, assignment)
  for (i in 1:nrow(centers)) {
    tmp <- subset(newtest, assignment==i)
    if (nrow(tmp) == 0) next
    tmp$assignment <- NULL
    print(nrow(tmp))
    print(table(tmp$V39))
    file <- paste("splitedtest", i, ".csv", sep="")
    write.csv(tmp, file, row.names = FALSE, col.names = FALSE )
  }
}

csTest <- function(numOfClusters) {
  combine <- as.table(rbind(c(0,0), c(0,0)))
  for (i in 1:numOfClusters) {
    path <- paste("splitedtest", i, ".csv", sep="")
    if (file.exists(path)) {
      load(paste("my_model", i, ".rda", sep = ""))
      tmp <- read.csv(path, header = TRUE)
      pred <- predict(model, tmp[,1:(ncol(tmp) - 1)])
      confusion <- table(pred, tmp[,ncol(tmp)])
      if(nrow(confusion) == 1) {
        confusion <- rbind(confusion, "2" = c(0,0))
      }
      if(ncol(confusion) == 1) {
        confusion <- cbind(confusion, "2" = c(0,0))
      }
      print(confusion)
      combine <- combine + confusion
    }
  }
  
  return(combine)
}

# Main
kdataset <- read.csv("bcancer.csv", header = FALSE)
kdataset$V39 <- as.factor(kdataset$V39)
y <- kdataset[, 39]

# 5X2 CV
confusion <- as.table(rbind(c(0,0), c(0,0)))
for (i in 1:5) {
  p <- kdataset[kdataset$V39 == 2, ]
  n <- kdataset[kdataset$V39 == 1, ]
  psub <- sample(1:nrow(p), floor(nrow(p) / 2))
  nsub <- sample(1:nrow(n), floor(nrow(n) / 2))
  tr <- rbind(p[psub, ], n[nsub, ])
  te <- rbind(p[-psub, ], n[-nsub, ])
  
  # mydata <-kdataset
  # wss <- (nrow(mydata)-1)*sum(apply(mydata,2,var))
  # for (i in 2:15) 
  #   wss[i] <- sum(kmeans(mydata,centers=i)$withinss)
  # 
  # plot(1:15, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")
  # 
  numOfClusters <- 8
  centers <- csTrain(tr, numOfClusters)
  splitTestset(te, centers)
  result <- csTest(numOfClusters)
  print(result)
  confusion <- confusion + result
  
  numOfClusters <- 8
  centers <- csTrain(te, numOfClusters)
  splitTestset(tr, centers)
  result <- csTest(numOfClusters)
  print(result)
  confusion <- confusion + result
}

confusion <- confusion / 10
TP <- confusion[2,2]
TN <- confusion[1,1]
FP <- confusion[2,1]
FN <- confusion[1,2]
specificity <- TN / (TN + FP)
sensitivity <- TP / (TP + FN)

print(confusion)
print(paste("Specificity:", specificity))
print(paste("Sensitivity:", sensitivity))
print(paste("Accuracy:", (TP + TN) / (TP + TN + FP + FN)))
