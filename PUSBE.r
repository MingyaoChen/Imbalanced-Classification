require("foreign")
require("e1071")
require("FSelector")
require("RWeka")
require("nnet")
require("DMwR")
rm(list = ls(all = TRUE))
setwd("C:\\Users\\Mingyao\\Dropbox\\Xiaocong\\Practicum\\PUSBE")


usbeTrain <- function(tr) {
  nonCancer <- tr[tr$V39 == 1,]
  yesCancer <- tr[tr$V39 == 2,]
  
  # Under Sampling
  i <- 1
  isSampled <- rep(FALSE, nrow(nonCancer))
  while (TRUE) {
    sub <- sample(1:nrow(nonCancer), floor(nrow(yesCancer)))  
    tmp <- nonCancer[sub, ]
    sam <- rbind(tmp, yesCancer)
    sam <- sam[sample(nrow(sam)), ]
    write.arff(sam, paste("sample" , i, ".arff", sep = ""))
    
    isSampled[sub] <- TRUE
    for ( j in 1:length(sub)) {
      isSampled[sub[j]] <- TRUE
    }
    print(isSampled)
    if ( !(FALSE %in% isSampled) || i >= 10)
      break
    else 
      i <- i + 1
  }
  
  numofSamples <- i
  
  # Train each cluster
  for (i in 1:numofSamples) {
    sam <- read.arff(paste("sample" , i, ".arff", sep = ""))
    print(table(sam$V39))
    # Feature selection : weights <- information.gain(V39~., sam)
    # subset <- cfs(V39~., sam)
    weights <- chi.squared(V39~., sam)
    subset <- cutoff.k(weights, 15)
    save(subset, file=paste("subset",i,".rda", sep = ""))
    sam <- data.frame(cbind(sam[,subset], class = sam[,ncol(sam)]))
    obj <- tune.svm(class~., data=sam, gamma = 2^(-5:5), cost = 2^(1:10),  tunecontrol = tune.control(sampling = "fix"))
    #obj <- tune.svm(class~., data=sam, gamma = 2^(-5:5), cost = 2^(1:10))
    print(paste("Gamma", obj$best.parameters$gamma))
    print(paste("Cost:", obj$best.parameters$cost))
    svmModel <- svm(class~., data=sam, kernel="radial", gamma = obj$best.parameters$gamma, cost = obj$best.parameters$cost)
    save(svmModel, file=paste("my_model", i, ".rda", sep = ""))
    pred <- predict(svmModel, sam[,-ncol(sam)])
    print("single classifier performance")
    print(table(pred, sam[,ncol(sam)]))
  }
  
  resultset <- list()
  for (i in 1:numofSamples) {
  # Use train set to predict
    load(paste("my_model", i, ".rda", sep = ""))
    load(paste("subset",i,".rda", sep = ""))
    exp <- data.frame(cbind(tr[,subset], class = tr[,ncol(tr)]))
    pred <- predict(svmModel,exp)
    pred <- as.integer(pred)
    start <- (i-1) * length(pred) + 1
    end <- i * length(pred)
    resultset[start: end] <- pred
  }
  
  resultset <- data.frame(matrix(unlist(resultset), ncol = numofSamples, byrow = FALSE))
  class <- tr[,ncol(tr)]
  resultset <- cbind(resultset, class)
  
  #Purnning
  prune <- resultset
  prune$class <- lapply(prune$class, as.integer)
  for (i in 1:(ncol(prune) - 1)) {
    for( j in 1:nrow(prune)) {
      if(prune[j, i] == prune[j, ncol(prune)]) {
        prune[j, i] <- 1
      } else {
        prune[j, i] <- 0
      }
    }
  }
  prune$class <- NULL
  
  # Get combination of each classifiers
  maxNum <- ncol(prune)
  xx <- c(1:maxNum)
  combo <- list()
  for ( i in 2:maxNum) {
    tmp <- combn(xx, i)
    for (j in 1:ncol(tmp)) {
      combo[[length(combo) + 1]] <- tmp[,j]
    }
  }
  
  # Calculate diversity
  divers <- list()
  for(i in 1:length(combo)) {
    v <- combo[[i]]
    L <- length(v)
    son <- 0.0
    mom <- 0.0
    c <- combn(v, 2)
    
    for (j in 1:ncol(c)) {
      tmp <- c[,j]
      bp <- 0
      bn <- 0
      pn <- 0
      np <- 0
      for ( k in 1:nrow(prune)) {
        if (prune[k,tmp[1]] == prune[k,tmp[2]]) {
          if (prune[k,tmp[1]] == 1) {
            bp <- bp + 1
          } else {
            bn <- bn + 1
          }
        } else {
          if (prune[k,tmp[1]] == 1) {
            pn <- pn + 1
          } else {
            np <- np + 1
          }
        }
      }
      son <- son + bn
      mom <- mom + bp + bn + np + pn
    }
    div <- 2 * son / (mom * L *(L - 1))
    #print(v)
    #print(div)
    divers[[length(divers) + 1]] <- div
  }
  #print(unlist(combo))
  
  indexofcomb <- which.min(unlist(divers))
  haha <- combo[[indexofcomb]]
  haha[length(haha) + 1] <- ncol(resultset)
  total <- ncol(resultset)
  xx <- c(1:total)
  
  pruned <- list()
  for ( i in 1:length(xx)) {
    if ((xx[i] %in% haha)) {
      pruned[[length(pruned) + 1]] <- resultset[,i]
    }
  }
  pruned <- unlist(pruned)
  mytrain <- data.frame(matrix(pruned, nrow = nrow(resultset)))
  mytrain[length(mytrain)] <- as.factor(mytrain[,length(mytrain)])
  
  #fusion
  n <- names(mytrain)
  f <- as.formula(paste(names(mytrain)[length(names(mytrain))],"~", paste(n[!n %in% names(mytrain)[length(names(mytrain))]], collapse = " + ")))
  obj <- tune.nnet(f, data=mytrain, size = 1:5, decay = 0.1:0.5)
  fusionmodel <-  nnet(f, data = mytrain, size = obj$best.parameters$size, decay = obj$best.parameters$decay, maxit = 1000)
  save(fusionmodel, file="fusionModel.rda")
  
  # fusion using naive bayes
  # fusionmodel <- naiveBayes(f, data = resultset, laplace = 3)
  # save(fusionmodel, file="fusionModel.rda")
  result <- list("numofSamples" = numofSamples, "prune" = haha)
  return(result)
}

usbeTest <- function(te, numofSamples, prune) {
  testset <- list()
  for (i in 1:numofSamples) {
    load(paste("my_model", i, ".rda", sep = ""))
    load(paste("subset",i,".rda", sep = ""))
    exp <- data.frame(cbind(te[,subset], class = te[,ncol(te)]))
    
    pred <- predict(svmModel, exp)
    pred <- as.integer(pred)
    start <- (i-1) * length(pred) + 1
    end <- i * length(pred)
    testset[start: end] <- pred
  }
  testset <- data.frame(matrix(unlist(testset), ncol = numofSamples, byrow = FALSE))
  
  prune <-prune[1:length(prune)-1]
  total <- ncol(testset)
  xx <- c(1:total)
  
  pruned <- list()
  for ( i in 1:length(xx)) {
    if ((xx[i] %in% prune)) {
      pruned[[length(pruned) + 1]] <- testset[,i]
    }
  }
  pruned <- unlist(pruned)
  mytest <- data.frame(matrix(pruned, nrow = nrow(testset)))
  
  # load("fusionModel.rda")
  # pred <- predict(fusionmodel, testset, type = "class")
  # t <- table(pred, te[,ncol(te)])
  # if(nrow(t) == 1) {
  #   t <- rbind(t, "2" = c(0,0))
  # }
  
  pred <- list()
  for (i in 1:nrow(mytest)) {
    tmp <- as.list(mytest[i, ])
    tt <- table(unlist(tmp))
    cate <- names(tt)
    if (length(tt) == 1)
      pred[length(pred) + 1] <- as.integer(cate[1])
    else if (tt[1] >= tt[2])
      pred[length(pred) + 1] <- as.integer(cate[1])
    else
      pred[length(pred) + 1] <- as.integer(cate[2])
  }
  t <- table(unlist(pred), te[,ncol(te)])
  if(nrow(t) == 1) {
    t <- rbind(t, "2" = c(0,0))
  }
  else if(ncol(t) == 1)
    t <- cbind(t, "2" = c(0,0))
  
  print(t)
  return(t)
  # return(pred)
} 

myCV <- function(bcancer) {
  for (i in 1:1) {
    p <- bcancer[bcancer$V39 == 2, ]
    n <- bcancer[bcancer$V39 == 1, ]
    psub <- sample(1:nrow(p), floor(nrow(p) / 2))
    nsub <- sample(1:nrow(n), floor(nrow(n) / 2))
    # First Folds
    tr <- rbind(p[psub, ], n[nsub, ])
    te <- rbind(p[-psub, ], n[-nsub, ])
    result <- usbeTrain(tr)
    print(result[[2]])
    if (i == 1) {
      confusion <- usbeTest(te, result[[1]], result[[2]])
    } else {
      confusion <- confusion + usbeTest(te, result[[1]], result[[2]])
    }
    
    # Second Folds
    result <- usbeTrain(te)
    print(result[[2]])
    confusion <- confusion + usbeTest(tr, result[[1]], result[[2]])
    print(confusion)
  }
  
  confusion <- confusion / 2
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
}

# Main
bcancer <- read.csv("bcancer.csv", header=FALSE)
bcancer$V39 <- as.factor(bcancer$V39)

#mushroom <- read.arff("mushroom_processed.arff")

tmp <-read.arff("heartdisease_processed.arff")
heart <- data.frame(scale(tmp[,1:ncol(tmp) - 1]))
heart <- cbind(heart, V39 = tmp$class)

tmp <- read.arff("hepatitis_processed.arff")
hepatiris <- data.frame(scale(tmp[,1:ncol(tmp) - 1]))
hepatiris <- cbind(hepatiris, V39 = tmp$class)
rm(tmp)
myCV(bcancer)
