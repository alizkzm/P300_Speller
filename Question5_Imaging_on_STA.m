%% Question 5 
target_Average = mean(train_X (1:Numof_target_trials,:)) ;
nontarget_Average = mean(train_X ((end-Numof_nontarget_trials):end,:)) ;
target_dist = sum(( test_X - target_Average ) .^ 2 , 2 ) ;
nontarget_dist = sum(( test_X - nontarget_Average ) .^ 2 , 2 ) ;
predictedY = zeros (Numof_nontarget_trials+Numof_target_trials,1) ;
predictedY(find(target_dist < nontarget_dist)) = 1 ;

test_Confusion_Matrix = zeros (2) ;
test_Confusion_Matrix(1,1) = sum (predictedY&test_Y) ; % Trgets predicted right
test_Confusion_Matrix(1,2) = sum ((~predictedY)&(test_Y)) ; % Trgets predicted wrong
test_Confusion_Matrix(2,2) = sum ((~predictedY)&(~test_Y)) ; % NonTrgets predicted right
test_Confusion_Matrix(2,1) = sum ((predictedY)&(~test_Y)) ; % NonTrgets predicted wrong
test_Confusion_Matrix
TEST_Totall_Accuracy = (test_Confusion_Matrix(1,1) + test_Confusion_Matrix(2,2))/(Numof_target_trials + Numof_nontarget_trials)
TEST_Target_Accuracy = test_Confusion_Matrix(1,1)/ (test_Confusion_Matrix(1,1) + test_Confusion_Matrix(2,1))
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
temp = cat (2 , test_X , Light , predictedY , test_ind ) ;
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