function [train_target train_nontarget test_target test_nontarget] =  TrialExtraction(subjectNum)
% Calling function IndexExtraction
[train_target_index train_nontarget_index test_target_index test_nontarget_index] = IndExtraction (subjectNum) ;
% Loading the subject data
subjectNum2 = subjectNum(1:2) ;
if(length(subjectNum)>6)
    subjectNum2 = [subjectNum2 '0'];
end
subject = (load(subjectNum)) ;
channelNum = 11 ;
% Extracting test and train ma67trix
test = subject.(subjectNum2).test ;
train = subject.(subjectNum2).train ;

% signal window specification
Fs = 256 ;
timewindow = 800 / 1000 ;
signalwindow = floor(Fs * timewindow) ;

% Filtering the signals
h = BandpassFilter ;
for i = 2 : 9
   train(i,:) = filter(h,train(i,:)) ;
   test(i,:) = filter(h,test(i,:)) ;
end

% Extracting the train target 
% initializing train target
Numoftrials = length(train_target_index) ;
train_target = zeros (channelNum , Numoftrials , signalwindow ) ;

for i = 1 : Numoftrials
    for j = 1 : channelNum
        train_target (j , i , 1 : signalwindow ) = train (j , train_target_index(i):(train_target_index(i) + signalwindow-1)) ;
    end
end

% Extracting the train nontarget 
% initializing train nontarget
Numoftrials = length(train_nontarget_index) ;
train_nontarget = zeros (channelNum , Numoftrials , signalwindow ) ;

for i = 1 : Numoftrials
    for j = 1 : channelNum
        train_nontarget (j , i , 1 : signalwindow ) = train (j , train_nontarget_index(i):(train_nontarget_index(i) + signalwindow-1)) ;
    end
end

% Extracting the test target 
% initializing test target
Numoftrials = length(test_target_index) ;
test_target = zeros (channelNum , Numoftrials , signalwindow ) ;

for i = 1 : Numoftrials
    for j = 1 : channelNum
        test_target (j , i , 1 : signalwindow ) = test (j , test_target_index(i):(test_target_index(i) + signalwindow-1)) ;
    end
end

% Extracting the train nontarget 
% initializing train nontarget
Numoftrials = length(test_nontarget_index) ;
test_nontarget = zeros (channelNum , Numoftrials , signalwindow ) ;

for i = 1 : Numoftrials
    for j = 1 : channelNum
        test_nontarget (j , i , 1 : signalwindow ) = test (j , test_nontarget_index(i):(test_nontarget_index(i) + signalwindow-1)) ;
    end
end
end