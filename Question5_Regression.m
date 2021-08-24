% Specifying the Subject
subjectNum = 's8.mat' ;
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
% Table for Train

Train_Trials = cat (2,train_target , train_nontarget) ;
train_X_prime = zeros (size(Train_Trials,2) , 8*size(Train_Trials,3) ) ;
% Creating Labels _ TRAN
% Creating the X matrix _ TRAIN
for i = 1 : size(train_X_prime,1)
   for j = 0 : channelNum - 4
       train_X_prime (i , signalwindow * j + 1 : signalwindow * (j+1)) = reshape(Train_Trials(j+2 , i , :),1,signalwindow) ;
   end
end

X = cat (2 , train_X_prime , train_Y ) ;

% Table for Test

Test_Trials = cat (2,test_target , test_nontarget) ;
test_X_prime = zeros (size(Test_Trials,2) , (channelNum-3) *size(Test_Trials,3)) ;
% Creating Labels _ TRAN
% Creating the X matrix _ TRAIN
for i = 1 : size(test_X,1)
   for j = 0 : channelNum - 4
       test_X_prime (i , signalwindow * j + 1 : signalwindow * (j+1)) = reshape(Test_Trials(j+2 , i , :),1,signalwindow) ;
   end
end
% Regression on raw data (Fs = 256 Hz)
LinMod1 = fitlm (train_X_prime,train_Y) ; % Regression Model
figure
histogram(LinMod1.Fitted(find(train_Y==1)),'Normalization','pdf')
hold on
histogram(LinMod1.Fitted(find(train_Y==0)),'Normalization','pdf')
title ('TRAIN Target & Nontarget Distributions -> Regression on raw data')
Y1 = LinMod1.predict(test_X_prime) ;
figure
histogram(Y1(find(test_Y==1)),'Normalization','pdf','BinWidth',0.5)
hold on
histogram(Y1(find(test_Y==0)),'Normalization','pdf','BinWidth',0.5) 
title ('TEST Target & Nontarget Distributions -> Regression on raw data')

% Regression on downsampled data (Fs = 64 Hz)
LinMod2 = fitlm (train_X,train_Y) ; % Regression Model
figure
histogram(LinMod2.Fitted(find(train_Y==1)),'Normalization','pdf')
hold on
histogram(LinMod2.Fitted(find(train_Y==0)),'Normalization','pdf')
title ('TRAIN Target & Nontarget Distributions -> Regression on downsampled data')
Y2 = LinMod2.predict(test_X) ;
figure
histogram(Y2(find(test_Y==1)),'Normalization','pdf','BinWidth',0.05)
hold on
histogram(Y2(find(test_Y==0)),'Normalization','pdf','BinWidth',0.05) 
title ('TEST Target & Nontarget Distributions -> Regression on downsampled data')
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
%% Regressing data
LinModel = fitlm (train_X,train_Y) ;
regressedY = LinModel.predict(test_X) ;
threshold = 0.3 ;
regressedY (find(regressedY<threshold))=0 ;
regressedY (find(regressedY>threshold))=1 ;
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
temp = cat (2 , test_X , Light , regressedY , test_ind ) ;
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
    
    if (logical(length(L)))
    LDecision = mode (L) ;
    Decision (ceil(LDecision/6) , mod(LDecision+5,6)+1) = 1 ;
    end
    if (logical(length(U)))
    UDecision = mode (U) ;
    Decision (ceil(UDecision/6) , mod(UDecision+5,6)+1) = 2 ;
    end
    if (logical(length(C)))
    CDecision = mode (C) ;
    Decision (ceil(CDecision/6) , mod(CDecision+5,6)+1) = 3 ;
    end
    if (logical(length(A)))
    ADecision = mode (A) ;
    Decision (ceil(ADecision/6) , mod(ADecision+5,6)+1) = 4 ;
    end
    if (logical(length(S)))
    SDecision = mode (S) ;
    Decision (ceil(SDecision/6) , mod(SDecision+5,6)+1) = 5 ;
    end
    
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
letter1 = sprintf ('%c' , ascii_Mat(find(Decision==1))) ;
end
if (length(find(Decision==2))==1)
letter2 = sprintf ('%c' , ascii_Mat(find(Decision==2))) ;
end
if (length(find(Decision==3))==1)
letter3 = sprintf ('%c' , ascii_Mat(find(Decision==3))) ;
end
if (length(find(Decision==4))==1)
letter4 = sprintf ('%c' , ascii_Mat(find(Decision==4))) ;
end
if (length(find(Decision==5))==1)
letter5 = sprintf ('%c' , ascii_Mat(find(Decision==5))) ;
end
sprintf ('%c%c%c%c%c',letter1,letter2,letter3,letter4,letter5)