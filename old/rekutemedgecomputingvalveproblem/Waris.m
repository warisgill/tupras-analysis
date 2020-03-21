clear all
close all
clc

Data = xlsread('Valve_Problem_Tags_Final.xlsx','Valve_Stiction','B6:I48847');
for i=1:1440:45000
    PIC013_PV = Data(i:i+1440,1);
    PIC013_SP = Data(i:i+1440,3);
    PIC013_OP = Data(i:i+1440,2);
    Ts = 15; %unit second
    [stic(i),hyst(i),goodness_of_fit(i),OPfs,PVfs]=tpr_PVOP_v2(PIC013_PV,PIC013_SP,PIC013_OP,Ts);
end
%[M,I]=find(goodness_of_fit == nonzeros(goodness_of_fit(1,:)));
I = find(goodness_of_fit);
OPf={};
PVf={};

for i = 1:length(I)
    if goodness_of_fit(I(i))>80
       PIC013_PV = Data(I(i):I(i)+1440,1);
       PIC013_SP = Data(I(i):I(i)+1440,3);
       PIC013_OP = Data(I(i):I(i)+1440,2);
       [stic_1(i),hyst_1(i),goodness_of_fit_1(i),OPs,PVs]=tpr_PVOP_v2(PIC013_PV,PIC013_SP,PIC013_OP,Ts);
       OPf{i}=OPs;
       PVf{i}=PVs;
    else
        continue
    end
end
close all

for i = 1:length(I)
    if isempty(PVf{1,i}(:))==0 
        figure(i)
        plot(OPf{i},PVf{i})
    else
        continue
    end
end
