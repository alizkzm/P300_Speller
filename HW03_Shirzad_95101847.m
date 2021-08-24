%% Part 3 : Filtering the signals
% Specifying the Subject
subjectNum = 's7.mat' ;
% Loading the signals
[train_target train_nontarget test_target test_nontarget] =  TrialExtraction(subjectNum) ;
% Signal charactrestics
Fs = 256 ;
t = linspace (0 , 0.8 , 204) ;
timewindow = 800 / 1000 ;
signalwindow = floor(Fs * timewindow) ;
channelNum = 11 ;
Numof_target_trials = size (test_target,2) ;
Numof_nontarget_trials = size (test_nontarget,2) ;
correlation = zeros (1,channelNum) ;

% Loading the subject data
subjectNum2 = subjectNum(1:2) ;
if(length(subjectNum)>6)
    subjectNum2 = [subjectNum2 '0'];
end
subject = (load(subjectNum)) ;
% Extracting test and train matrix
test = subject.(subjectNum2).test ;
train = subject.(subjectNum2).train ;
% Filtering the signals
h1 = BandpassFilter ;
for i = 2 : 9
   train(i,:) = filter(h1,train(i,:)) ;
   test(i,:) = filter(h1,test(i,:)) ;
end

correlation = zeros (1,channelNum-3) ;

for i = 2 : 9
figure
hold on
% Initializinfg the Mean of target
train_target_ERP = zeros (1,signalwindow) ;
% Mean Of target
train_target_ERP = reshape(train_target(i , : , :),Numof_target_trials,signalwindow) ;
train_target_ERP = mean (train_target_ERP , 1) ;
plot(t , train_target_ERP, 'red' , 'LineWidth' , 1.2 );
hold on 
% Variance Of target
train_target_var = reshape(train_target(i , : , :),Numof_target_trials,signalwindow) ;
train_target_var = sqrt(var (train_target_ERP , 1)) ;
train_target_upper = train_target_var + train_target_ERP ;
plot (t , train_target_upper , 'red','LineWidth' , 0.5)
hold on
train_target_lower = train_target_ERP - train_target_var ;
plot (t , train_target_lower , 'red','LineWidth' , 0.5)


% Initializing the Mean of target
hold on
train_nontarget_ERP = zeros (1,signalwindow) ;
% Mean Of target
train_nontarget_ERP = reshape(train_nontarget(i , : , :),Numof_nontarget_trials,signalwindow) ;
train_nontarget_ERP = mean (train_nontarget_ERP , 1) ;
plot(t , train_nontarget_ERP , 'blue' , 'LineWidth' , 1.2);
% Variance Of target
train_nontarget_var = reshape(train_nontarget(i , : , :),Numof_nontarget_trials,signalwindow) ;
train_nontarget_var = sqrt(var (train_nontarget_ERP , 1)) ;
train_nontarget_upper = train_nontarget_var + train_nontarget_ERP ;
plot (t , train_nontarget_upper , 'blue','LineWidth' , 0.5)
hold on
train_nontarget_lower = train_nontarget_ERP - train_nontarget_var ;
plot (t , train_nontarget_lower , 'blue','LineWidth' , 0.5)
grid on

correlation (i-1) = sum(train_target_ERP .* train_nontarget_ERP)/abs(mean(train_target_ERP)*mean(train_nontarget_ERP)) ;
end
figure
bar (1:8,correlation,'g')
title ('Target _nontarget ERP Correlation versus channels')
%% DownSampling the Data
train_target_DS = zeros (channelNum , Numof_target_trials , 51 ) ;
for i = 1 : channelNum 
    for j = 1 : Numof_target_trials
    train_target_DS (i , j , :) = downsample (train_target(i , j , :),4) ;
    end
end
train_nontarget_DS = zeros (channelNum , Numof_nontarget_trials , 51 ) ;
for i = 1 : channelNum 
    for j = 1 : Numof_nontarget_trials
    train_nontarget_DS (i , j , :) = downsample (train_nontarget(i , j , :),4) ;
    end
end
test_target_DS = zeros (channelNum , Numof_target_trials , 51 ) ;
for i = 1 : channelNum 
    for j = 1 : Numof_target_trials
    test_target_DS (i , j , :) = downsample (test_target(i , j , :),4) ;
    end
end
test_nontarget_DS = zeros (channelNum , Numof_nontarget_trials , 51 ) ;
for i = 1 : channelNum 
    for j = 1 : Numof_nontarget_trials
    test_nontarget_DS (i , j , :) = downsample (test_nontarget(i , j , :),4) ;
    end
end
DownSample_Window = size (test_nontarget_DS,3) ;
%% Providing the table for LDA TRAIN

% Table for Train

Train_Trials = cat (2,train_target_DS , train_nontarget_DS) ;
train_X = zeros (size(Train_Trials,2) , 8*size(Train_Trials,3) ) ;
% Creating Labels _ TRAN
train_Y = [ones(Numof_target_trials,1);zeros(Numof_nontarget_trials,1)] ;
% Creating the X matrix _ TRAIN
for i = 1 : size(train_X,1)
   for j = 0 : channelNum - 4
       train_X (i , DownSample_Window * j + 1 : DownSample_Window * (j+1)) = reshape(Train_Trials(j+2 , i , :),1,DownSample_Window) ;
   end
end

X = cat (2 , train_X , train_Y ) ;

% Table for Test

Test_Trials = cat (2,test_target_DS , test_nontarget_DS) ;
test_X = zeros (size(Test_Trials,2) , (channelNum-3) *size(Test_Trials,3)) ;
% Creating Labels _ TRAN
test_Y = [ones(Numof_target_trials,1);zeros(Numof_nontarget_trials,1)] ;
% Creating the X matrix _ TRAIN
for i = 1 : size(test_X,1)
   for j = 0 : channelNum - 4
       test_X (i , DownSample_Window * j + 1 : DownSample_Window * (j+1)) = reshape(Test_Trials(j+2 , i , :),1,DownSample_Window) ;
   end
end
%% Training the LDA Model

[trainedClassifier, validationAccuracy] = trainClassifier(X) ;
[trainedClassifier_Crossval, CrossvalidationAccuracy] = trainClassifier_Crossval(X) ;
YPred_train = trainedClassifier_Crossval.predictFcn(train_X) ;
YPred_test =  trainedClassifier_Crossval.predictFcn(test_X) ;
%% Analyzing the Coefficients
Mdl = fitcdiscr (train_X,train_Y) ;
Coeffs = Mdl.Coeffs(1,2).Linear ;
[Coeffs_sort,Coeffs_Importance] = sort(abs(Coeffs)) ;
Important_Features = zeros (1,8)  ;
Important_times = zeros (1,17) ;
for i = 1 : 8
    a = floor(Coeffs_Importance (end-40:end) / DownSample_Window) ;
    b = a .* Coeffs_sort(end-40:end);
    Important_Features (i) = sum(b(find(a==i))) ;
end
for i = 1 : 17
    a = mod(Coeffs_Importance (end-40:end),DownSample_Window) ;
    a = a .* Coeffs_sort(end-40:end);
    Important_times (i) = length (find(a>i & a<i+2)) ;
end
figure
bar (1:8,Important_Features)
title ('Channel Importance among first 40 features')
figure
bar (linspace(0,800,17),Important_times)
title ('Time Importance among first 40 features')
%% TEST Confusion Matrix
test_Confusion_Matrix = zeros (2) ;
test_Confusion_Matrix(1,1) = sum (YPred_test&test_Y) ; % Trgets predicted right
test_Confusion_Matrix(1,2) = sum ((~YPred_test)&(test_Y)) ; % Trgets predicted wrong
test_Confusion_Matrix(2,2) = sum ((~YPred_test)&(~test_Y)) ; % NonTrgets predicted right
test_Confusion_Matrix(2,1) = sum ((YPred_test)&(~test_Y)) ; % NonTrgets predicted wrong
test_Confusion_Matrix
TEST_Totall_Accuracy = (test_Confusion_Matrix(1,1) + test_Confusion_Matrix(2,2))/(Numof_target_trials + Numof_nontarget_trials)
TEST_Target_Accuracy = test_Confusion_Matrix(1,1)/ (test_Confusion_Matrix(1,1) + test_Confusion_Matrix(2,1))
%% TRAIN Confusion Matrix
train_Confusion_Matrix = zeros (2) ;
train_Confusion_Matrix(1,1) = sum (YPred_train&train_Y) ; % Trgets predicted right
train_Confusion_Matrix(1,2) = sum ((~YPred_train)&(train_Y)) ; % Trgets predicted wrong
train_Confusion_Matrix(2,2) = sum ((~YPred_train)&(~train_Y)) ; % NonTrgets predicted right
train_Confusion_Matrix(2,1) = sum ((YPred_train)&(~train_Y)) ; % NonTrgets predicted wrong
train_Confusion_Matrix
train_Tottal_Accuracy = validationAccuracy
train_Target_Accuracy = train_Confusion_Matrix(1,1)/ (train_Confusion_Matrix(1,1) + train_Confusion_Matrix(2,1))
CrossvalidationAccuracy

%% Concatinating the Indexes
[train_target_index train_nontarget_index test_target_index test_nontarget_index] = IndExtraction (subjectNum) ;
test_X_sort = zeros (size(test_X)) ;
test_ind = cat (1,test_target_index',test_nontarget_index') ;
%% finding the Number of Light
Light = zeros (Numof_nontarget_trials + Numof_target_trials,1) ;
for i = 1 : length (Light)
    Light(i,1) = test (10,test_ind(i)) ;
end
%% Sorting the DATA
temp = cat (2 , test_X , Light , YPred_test , test_ind ) ;
temp = sortrows (temp , size(temp,2)) ;
test_X_sort = temp (:,1:size(test_X,2)) ;
YPred_test = temp (: , end-1 ) ;
Light = temp (: , end - 2) ;
%% Spelling
character_trial = ( Numof_nontarget_trials + Numof_target_trials ) / 5 ;
l = YPred_test (1 : character_trial ) .* Light (1 : character_trial )  ;
u = YPred_test (character_trial + 1 : 2 * character_trial ) .* Light (character_trial + 1 : 2 * character_trial ) ;
c = YPred_test (2 * character_trial + 1 : 3 * character_trial ) .* Light (2 * character_trial + 1 : 3 * character_trial ) ;
a = YPred_test (3 * character_trial + 1 : 4 * character_trial )  .* Light (3 * character_trial + 1 : 4 * character_trial ) ;
s = YPred_test (4 * character_trial + 1 : 5 * character_trial ) .* Light (4 * character_trial + 1 : 5 * character_trial ) ;
L = l(find (l ~= 0)) ;
U = u(find (u ~= 0)) ;
C = c(find (c ~= 0)) ;
A = a(find (a ~= 0)) ;
S = s(find (s ~= 0)) ;
Decision = zeros(6) ;

if (length(subjectNum2)==2 & ( subjectNum2(2)== '1' | subjectNum2(2)== '2' ))
    LDecision = zeros (1) ;
    UDecision = zeros (1) ;
    CDecision = zeros (1) ;
    ADecision = zeros (1) ;
    SDecision = zeros (1) ;
    
    
    LDecision = mode (L) ;
    Decision (ceil(LDecision/6) , mod(LDecision+5,6)+1) = 1 ;
    UDecision = mode (U) ;
    Decision (ceil(UDecision/6) , mod(UDecision+5,6)+1) = 2 ;
    CDecision = mode (C) ;
    Decision (ceil(CDecision/6) , mod(CDecision+5,6)+1) = 3 ;
    ADecision = mode (A) ;
    Decision (ceil(ADecision/6) , mod(ADecision+5,6)+1) = 4 ;
    SDecision = mode (S) ;
    Decision (ceil(SDecision/6) , mod(SDecision+5,6)+1) = 5 ;
    
else
% Decision Making for L
LDecision = zeros (6) ;
for i = 1 : length (L)
   if (L(i)<7)
       LDecision(:,L(i)) = LDecision(:,L(i)) + 1 ;
   end
      if (L(i)>6)
       LDecision(L(i)-6,:) = LDecision(L(i)-6,:) + 1 ;
   end
end
Decision (find(LDecision==max(max(LDecision)))) = 1 ;


% Decision Making for U
UDecision = zeros (6) ;
for i = 1 : length (U)
   if (U(i)<7)
       UDecision(:,U(i)) = UDecision(:,U(i)) + 1 ;
   end
      if (U(i)>6)
       UDecision(U(i)-6,:) = UDecision(U(i)-6,:) + 1 ;
   end
end
if(Decision (find(UDecision==max(max(UDecision)))) == 0)
Decision (find(UDecision==max(max(UDecision)))) = 2 ;
end


% Decision making for C
CDecision = zeros (6) ;
for i = 1 : length (C)
   if (C(i)<7)
       CDecision(:,C(i)) = UDecision(:,C(i)) + 1 ;
   end
      if (C(i)>6)
       UDecision(C(i)-6,:) = UDecision(C(i)-6,:) + 1 ;
   end
end
if (Decision (find(CDecision==max(max(CDecision)))) == 0)
Decision (find(CDecision==max(max(CDecision)))) = 3 ;
end


% Decision Making for A
ADecision = zeros (6) ;
for i = 1 : length (A)
   if (A(i)<7)
       ADecision(:,A(i)) = ADecision(:,A(i)) + 1 ;
   end
      if (A(i)>6)
       ADecision(A(i)-6,:) = ADecision(A(i)-6,:) + 1 ;
   end
end
if (Decision (find(ADecision==max(max(ADecision)))) == 0)
Decision (find(ADecision==max(max(ADecision)))) = 4 ;
end


% Decision Making for S
SDecision = zeros (6) ;
for i = 1 : length (S)
   if (S(i)<7)
       SDecision(:,S(i)) = SDecision(:,S(i)) + 1 ;
   end
      if (S(i)>6)
       SDecision(S(i)-6,:) = SDecision(S(i)-6,:) + 1 ;
   end
end
if (Decision (find(SDecision==max(max(SDecision)))) == 0)
Decision (find(SDecision==max(max(SDecision)))) = 5 ;
end
end
%% Ascii Code Matrix
ascii_Mat = zeros (6) ;
letter1 = [] ;
letter2 = [] ;
letter3 = [] ;
letter4 = [] ;
letter5 = [] ;
for (i = 1 : 36)
    if (i>26)
        ascii_Mat(i) = i + 22 ;
    else 
        ascii_Mat (i) = i + 64 ;
    end
end
ascii_Mat = ascii_Mat' ;
if (length(find(Decision==1))==1)
letter1 = sprintf ('%c' , 42) ;
else
letter1 = sprintf ('%c' , 42);
end
if (length(find(Decision==2))==1)
letter2 = sprintf ('%c' , ascii_Mat(find(Decision==2))) ;
else
letter2 = sprintf ('%c' , 42);
end
if (length(find(Decision==3))==1)
letter3 = sprintf ('%c' , ascii_Mat(find(Decision==3))) ;
else
letter3 = sprintf ('%c' , 42);
end
if (length(find(Decision==4))==1)
letter4 = sprintf ('%c' , ascii_Mat(find(Decision==4))) ;
else
letter4 = sprintf ('%c' , 42);
end
if (length(find(Decision==5))==1)
letter5 = sprintf ('%c' , ascii_Mat(find(Decision==5))) ;
else
letter5 = sprintf ('%c' , 42);
end
sprintf ('%c%c%c%c%c',letter1,letter2,letter3,letter4,letter5)
%% Accuracy Histogram
TEST_ACCURACY = [ 96.37 , 96.93 , 73.33 , 79.22 , 78.11 ,73.44, 74.44 , 80.33 , 80.00 , 83.89 ];
figure
histogram (TEST_ACCURACY,'binwidth' , 0.5) 
title ('Validation Accuracy on 10 Subjects')
