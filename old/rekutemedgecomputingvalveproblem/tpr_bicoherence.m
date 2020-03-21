function [NGI,NLI,f1_bic,f2_bic,bic2max] = tpr_bicoherence(er)
%Bicoherence hesaplanmasý
seglength=128;overlap=0.45;
LX=seglength;
erol=buffer(er,seglength,floor(seglength*overlap),'nodelay');

[m,n]=size(erol); %erol matrisinin boyutlarýnýn belirlenmesi

%Substract the mean
    erol_new=detrend(erol,'constant');

%Hanning Pencereleme iþlemi
for i=1:m
    w(i)=0.5-0.5*cos(2*pi*(i-1)/(seglength-1)); %Hanning pencere fonksiyonu
end

for i=1:m
    erol_1(i,:)=w(i)*erol_new(i,:);
end

%Her bir segmentin DFT'sinin hesaplanmasý
DFT_length=128;
A=fft(erol_1,DFT_length);

for ix=1:n
Y=A(:,ix);
for k=1:(DFT_length/2)
    for l=1:(DFT_length/2)
        Bsp(k,l)=Y(k)*Y(l)*conj(Y(k+l-1));
        de1(k,l)=abs(Y(k)*Y(l))^2;
        de2(k,l)=abs(Y(k+l-1))^2;
    end
end
D1{ix}=de1;D1_1{ix}=de1;
D2{ix}=de2;D2_1{ix}=de2;
Bs{ix}=Bsp;
end


%Squared Bicoherence hesabýnýn yapýlmasý
D1m = D1{1}; D2m = D2{1}; Bm = Bs{1};
for k = 2:length(Bs);
	D1m = D1m + D1{k};
	D2m = D2m + D2{k};
	Bm = Bm + Bs{k};
end
D1 = D1m/k; D2 = D2m/k; Bm = Bm/k;

 %Calculate squared bicoherence value
LX=DFT_length/2;
bic2 = zeros(LX);
for k = 1:DFT_length/2
	for l = 1:DFT_length/2
		bic2(k,l) = abs(Bm(k,l))^2/(D1(k,l)*D2(k,l));
	end
end

%%
bican=bic2;

waxis = linspace(0,0.5,length(bican)+1);
waxis = waxis(1:end-1);
%%

limit = waxis(end)*(2/3);

k = 1;
while k < length(waxis)
	if waxis(k)>limit
		limit = k-1;
		k = Inf;
	else 
		k = k + 1;
	end
end

for f1 = 1:limit
	for f2 = 1:length(waxis)
		if f2 == 1 | f2 >= f1;
			bican(f1,f2) = 0;
		end
	end
end

for f1 = limit:length(waxis)
	for f2 = 1:length(waxis)
		if f2 == 1 | f2 >= -2*f1 + length(waxis)*2
			bican(f1,f2) = 0;
		end
	end
end

cont = 1;
for f1 = 1:length(waxis)
	for f2 = 1:min([f1, -2*f1+length(Y)]);
		bicpermed(cont) = bican(f1,f2);
		cont = cont + 1;
	end
end

%figure('Name',strcat('Values for segment:',num2str(p)));
surf(waxis,waxis,bican');
axis([0 0.5 0 0.5 0 1]);
   % set(hcc,'view',[145 15],'Alim',[0 1],...
%        'Clim',[0 0.08]);
%    view([145 15])

[c,rows] = max(bican');
[c,column] = max(max(bican'));
f1 = rows(column);
f2 = column;    
output.f1 = waxis(f1);f1_bic=output.f1;
output.f2 = waxis(f2);f2_bic=output.f2;

K=n; calfa=5.99;
bicsign=bicpermed(bicpermed>calfa/(2*K));
TNLI=sum(bicsign);
NGI=(sum(bicsign)/length(bicsign))-calfa/(2*K*length(bicsign));
fprintf('NGI: %3.4f\n',NGI)
title('Squared bicoherence','fontweight','bold','fontsize',12)

bic2m = mean(nonzeros(bicpermed));
bic2v = std(nonzeros(bicpermed));
bic2max = max(bicpermed);
NLI=abs(bic2max-(bic2m+2*bic2v));
end