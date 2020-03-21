function [r_oscindex, Tperiod_s, Tperiod]=tpr_regularity(er,Ts)

%==================Thornhill Test Code===================================
% ACF zero crossing test (Thornhill)
n=length(er);
osvec=xcov(er,n,'coeff');
v=osvec(ceil(length(osvec)/2):length(osvec));
plot(v)

fp1=v(1:end-1);
fp2=v(2:end);
fp1fp2=fp1.*fp2;
acfzcindex_one=find(fp1fp2<0);
acfzcindex_two=find(fp1fp2==0);
acfzcindex=sort([acfzcindex_one',acfzcindex_two']);
         
acfzcindex_1=acfzcindex+1;

interpolindx=zeros(1,length(acfzcindex_1));
for s=1:length(acfzcindex_1)
    interpolindx(s)=((0-v(acfzcindex(s)))/(v(acfzcindex_1(s))-v(acfzcindex(s))))*(acfzcindex_1(s)-acfzcindex(s))+acfzcindex(s);
end



ss=length(interpolindx);
deltaT=zeros(1,ss-1);
%     if le(length(interpolindx),10)
        for i=1:length(interpolindx)-1
            deltaT(i)=interpolindx(i+1)-interpolindx(i);
        end
%     else
%         for i=1:10
%             deltaT(i)=interpolindx(i+1)-interpolindx(i);
%         end
%     end
 
Tperiod=(2/(length(deltaT)+1))*sum(deltaT); %length(deltaT) deðil
r_oscindex=(1/3)*(Tperiod/(2*std(deltaT)));
Tperiod_s=Tperiod*Ts;

