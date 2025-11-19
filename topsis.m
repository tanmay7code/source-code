
%%%%%TOPSIS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Input D_M data from NSGA2 or SPEA2

% Defined decision matrix
D_M=[fp1,fp2];


N_1=0;

for i=1:pop

    N_1=N_1+(fp1(i,1))^2;
end

Norm1= sqrt(N_1);



N_2=0;

for i=1:pop

    N_2=N_1+(fp2(i,1))^2;
end

Norm2= sqrt(N_2);


fp1N=(1/Norm1)*fp1;

fp2N=(1/Norm2)*fp2;



%%normalized decision matrix
Norm_D_M=[fp1N,fp2N];


%%% %%%
% Choose weightage for each criterion

 w_1=0.5; w_2=0.5; %w_1+w_2=1;

fp1NN=fp1N*w_1;
fp2NN=fp2N*w_2;


 D_Mw=[fp1NN,fp2NN];



 %%%%%%%%%%%%%%%%%%%%
 % Determine best and worst alternatives

 best1=min(fp1NN);
 best2=min(fp2NN);

 best=[best1,best2];

 worst1=max(fp1NN);
 worst2=max(fp2NN);

 worst=[worst1,worst2];


ED_plus= zeros(pop,1);

for i=1:pop
 ED_plus(i,1)=sqrt((fp1NN(i,1)-best1)^2+ (fp2NN(i,1)-best2)^2);
end

ED_minius=zeros(pop,1);

for i=1:pop
 ED_minius(i,1)=sqrt((fp1NN(i,1)-worst1)^2+ (fp2NN(i,1)-worst2)^2);
end

%%%Define performence vector

PV=zeros(pop,1);
for i=1:pop
    PV(i,1)=ED_minius(i,1)/(ED_plus(i,1)+ED_minius(i,1));
end

PV_sort= sort(PV);%Take last element 


