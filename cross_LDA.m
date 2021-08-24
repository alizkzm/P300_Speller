function [Mdl success] = cross_LDA(X,foldNum)

dataNum = size(X , 1) ;
featureNum = size(X , 2)-1 ;
random_X = X(randperm(size(X, 1)), :) ;
fold_size = floor ( dataNum / foldNum ) ;
train_X = zeros (dataNum - fold_size ,  featureNum) ;
train_Y = zeros (dataNum - fold_size , 1) ;
test_X = zeros (fold_size , featureNum ) ;
test_Y = zeros (fold_size , 1) ;
success = 0 ;
for i = 0 : foldNum-1
    test_X = random_X (((fold_size * i)+1) : (fold_size * (i+1)) , 1:end-1 ) ;
    test_Y = random_X ((fold_size * i)+1 : fold_size * (i+1) , end ) ;
    random_X_temp = random_X ;
    random_X((fold_size * i)+1 : fold_size * (i+1) , :) = [] ;
    train_X = random_X (: , 1: end-1) ;
    train_Y = random_X (: , end) ;

    Model_temp = fitcdiscr(train_X,train_Y) ;
    YPred_test = predict(Model_temp , test_X) ;
    success_rate_temp = sum(YPred_test==test_Y)/fold_size ;
    if(success_rate_temp>success)
        Mdl = Model_temp ;
        success = success_rate_temp ;
    end
    random_X = random_X_temp ;
end
end