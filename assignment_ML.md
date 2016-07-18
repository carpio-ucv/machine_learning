machine\_learning
================
Juan Carpio
15 July 2016

R Markdown
----------

``` r
library("data.table")
```

    ## Warning: package 'data.table' was built under R version 3.2.5

``` r
library("caret")
```

    ## Warning: package 'caret' was built under R version 3.2.5

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## Warning: package 'ggplot2' was built under R version 3.2.3

``` r
library("mlbench")
```

    ## Warning: package 'mlbench' was built under R version 3.2.5

``` r
library("dplyr")
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:data.table':
    ## 
    ##     between, last

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
set.seed(673)


#Obtain Training data
data.train<-read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"),na.strings = c("", "NA","#DIV/0!" ))

df.train<-data.train[ , colSums(is.na(data.train)) 
                      < nrow(data.train)*0.8]

df.train<-sample_n(df.train,2000)
        
x.tr<-df.train[,-60] #x.tr<-data.train[,-160]
y.tr<-df.train[,60]


#Obtain Testing data
data.test<-read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"))
df.test<-data.test[ , colSums(is.na(data.test)) 
                      < nrow(data.test)*0.8]
x.test<-df.test[,-60]
y.test<-df.test[,60]

#testing
set.seed(62433)

#factor variables
levels(x.test$cvtd_timestamp) <- levels(df.train$cvtd_timestamp)
levels(x.test$new_window) <- levels(df.train$new_window)
levels(y.test) <- levels(y.tr)

#prediction 

#pred_rf <- predict(rf.fit, x.test)
#confusionMatrix(pred_rf, reference = y.test)


## Configuring parallell processing
library("parallel")
library("doParallel")
```

    ## Warning: package 'doParallel' was built under R version 3.2.5

    ## Loading required package: foreach

    ## Warning: package 'foreach' was built under R version 3.2.5

    ## Loading required package: iterators

    ## Warning: package 'iterators' was built under R version 3.2.3

``` r
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

####

#Configure trainControl object

fit.control <- trainControl(method = "cv",
                           number = 10,
                           allowParallel = TRUE)
```

### Develop training model

``` r
#rf.fit<-randomForest(x=x.tr,y=y.tr, do.trace= TRUE, prox=TRUE, 
#                     ntree=10, trControl = fit.control)

rf.fit2<-train(x=x.tr,y=y.tr, method="rf", do.trace= TRUE, ntree=10, prox=TRUE,  trControl = fit.control)
```

how you built your model,
-------------------------

how you used cross validation,
------------------------------

what you think the expected out of sample error is, and
-------------------------------------------------------

why you made the choices you did.
---------------------------------

![](assignment_ML_files/figure-markdown_github/pressure-1.png)<!-- -->

Note that the `echo = FALSE` parameter was added to the code chunk to prevent printing of the R code that generated the plot.
