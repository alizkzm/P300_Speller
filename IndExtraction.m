function [train_target_index train_nontarget_index test_target_index test_nontarget_index] = IndExtraction (subjectNum)


% Loading the subject data
subjectNum2 = subjectNum(1:2) ;
if(length(subjectNum)>6)
    subjectNum2 = [subjectNum2 '0'];
end
subject = (load(subjectNum)) ;
% Extracting test and train matrix
test = subject.(subjectNum2).test ;
train = subject.(subjectNum2).train ;



% Extracting train target index
train_target_index = find (train(11,:)==1) ;
train_target_index = train_target_index(1:4:length(train_target_index)-3) ;
% Extracting the train nontarget index
train_nontarget_index = find(train(10,:)~=0 & train(11,:)==0) ;
train_nontarget_index = train_nontarget_index(1:4:length(train_nontarget_index)) ;
% Extracting the test target index
test_target_index = find (test(11,:)==1) ;
test_target_index = test_target_index(1:4:length(test_target_index)-3) ;
% Extracting the test nontarget index
test_nontarget_index = find(test(10,:)~=0 & test(11,:)==0) ;
test_nontarget_index = test_nontarget_index(1:4:length(test_nontarget_index)) ;
end