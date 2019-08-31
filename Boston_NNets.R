#load libraries
library(keras)
library(mlbench)
library(dplyr)
library(magrittr)
library(neuralnet)
#load boston housing data
data("BostonHousing")
df <- as.data.frame(BostonHousing)
str(df)
#convert factor to numeric as NN can deal with numericdata easily
df <- df %>% mutate_if(is.factor,as.numeric)
#Neural Net Visualization
net <- neuralnet(medv ~ crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+b+lstat,
                 data=df,
                 hidden=c(10,5),
                 linear.output = F,
                 lifesign = "minimal",
                 rep=1)
plot(net,
     col.hidden = "darkgreen",
     col.hidden.synapse = "green",
     show.weights = F,
     information = F,
     fill="lightblue")
#information stored in net
names(net)
#matrix and data partition
data <- as.matrix(df)
dimnames(data) <- NULL
#partition
set.seed(123)
ind <- sample(2,nrow(data),replace=T,prob=c(0.8,0.2))
training_data <- data[ind==1,1:13]
test_data <- data[ind==2,1:13]
trainLab <- data[ind==1,14]
testLab <- data[ind==2,14]

#normalizing by {(value-mean)/std.dev}
data_mean <-colMeans(training_data)
data_std <- apply(training_data,2,sd)
training <- scale(training_data,center=data_mean,scale = data_std)
test <- scale(test_data,center=data_mean,scale=data_std)
#Creating a model using keras_tensorflow
model <- keras_model_sequential()
#similar model which we created above
model %>% 
  layer_dense(units=10,activation = "sigmoid",input_shape = 13) %>%
  layer_dense(units=5,activation = "relu",input_shape = 10) %>%
  layer_dense(units=1)
model
model %>% compile(loss="mse",
                  optimizer="rmsprop",
                  metrics="mae")
my_model <- model %>%
  fit(training,trainLab,epoch=130,batch_size=32,validation_split=0.25)
#Evaluate the model
performance_1 <- model %>% evaluate(test,testLab)
performance_1
#store prediction
pred <- model %>% predict(test)
#testLab and pred should be on a straight line for a better fit
#--------------------------------------------------------------------------
#tuning the model 
#Creating a model using keras_tensorflow
model_new <- keras_model_sequential()
#if overfitting occures we can use partial neural nets i.e we can drop neurons
model_new %>%
  layer_dense(units=80,activation = "tanh",input_shape = 13) %>%
  layer_dropout(0.35) %>%
  layer_dense(units=50,activation = "relu",input_shape = 80) %>%
  layer_dense(units=1)
model_new %>% compile(loss="mse",
                  optimizer="rmsprop",
                  metrics="mae")
model_new %>%
  fit(training,trainLab,epoch=150,batch_size=32,validation_split=0.15)
performance_2 <- model_new %>% evaluate(test,testLab)
performance_2
