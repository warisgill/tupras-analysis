function [stic,hyst,goodness_of_fit,OPfs,PVfs]=tpr_PVOP_v2(PV,SP,OP,Ts)
y=PV;
u=OP;
SP=SP;

%Ts=15;
er=SP-PV;
[r_oscindex, ~, Tperiod]=tpr_regularity(OP,Ts);

%%
%Bicoherence&Filter Boundary
overlap=0;
seglength=128;
dftlength=128;
[~,~,f1_bic,f2_bic,bic2max]=tpr_bicoherence(er);

f1_min=min(f1_bic,f2_bic);
f2_max=max(f1_bic,f2_bic);
%_______________________________________
% Wiener filter boundary selection
%_______________________________________
fL=max(0.001,f1_min-0.05);
fH=min(0.5,f2_max+0.05);

%Wiener filter
%________________________________________
%OP filtering
N=length(u);
OPf=fft(u,N);
k=[floor(fL*N+1) ceil(fH*N+1)];
OPf(1:k(1)-1)=0;
OPf(k(2)+1:(N+2-k(2)-1))=0;
OPf(N+2-k(1)+1:end)=0;
Opff=OPf;
OPf=ifft(OPf);


%PV filtering
N=length(y);
PVf=fft(y,N);
k=[floor(fL*N+1) ceil(fH*N+1)];
PVf(1:k(1)-1)=0;
PVf(k(2)+1:(N+2-k(2)-1))=0;
PVf(N+2-k(1)+1:end)=0;
PVff=PVf;
PVf=ifft(PVf);

%%
%Segmentation
%_________________________________________
L=1000;Tps=0.1;count=0;
while L>5*Tps
if count>0
    L=floor(5*Tps);
end
ufseg=buffer(OPf,L);
if nnz(ufseg)<numel(ufseg);
    ufseg=ufseg(:,1:end-1);
end
s=size(ufseg);
r_oscindex=zeros(1,s(2));
Tperiod=zeros(1,s(2));
for i=1:s(2)
[r_oscindex(i),~,Tperiod(i)]=tpr_regularity(ufseg(:,i),Ts);
end
[rmax,m]=max(r_oscindex);
Tps=Tperiod(m);
count=count+1;
end

%% Degerlerin segmentlere b�l�nd�kten sonraki de�eri
yfseg=buffer(PVf,L);
if nnz(yfseg)<numel(yfseg);
    yfseg=yfseg(:,1:end-1);
end
OPfs=ufseg(:,m);
PVfs=yfseg(:,m);

%% fit ellipse
x1=OPfs; x2=PVfs;
mn=length(x1);

%Coefficient Matrix
B = [x1, x2, ones(mn, 1), x1.^2, sqrt(2) * x1 .* x2, x2.^2];

%QR Decomposition
[Q, R] = qr(B);
Q1=Q;

% Decompose R into blocks
R11 = R(1:3, 1:3);
R12 = R(1:3, 4:6);
R22 = R(4:6, 4:6);

% Solve R22 * w = 0 subject to norm(w) == 1
[U, S, V] = svd(R22);
w = V(:, 3);

% Solve for the remaining variables
v = -R11\R12 * w;

% Fill in the quadratic form
A        = zeros(2);
A(1)     = w(1);
A([2 3]) = 1 / sqrt(2) * w(2);
A(4)     = w(3);
bv       = v(1:2);
c        = v(3);

% Diagonalise A - find Q, D such at A = Q' * D * Q
[Q, D] = eig(A);
Q = Q';

% If the determinant < 0, it's not an ellipse
if prod(diag(D)) <= 0 
    %error('fitellipse:NotEllipse', 'Linear fit did not produce an ellipse');
    stic = 0;
    hyst = 0;
    goodness_of_fit = 0;
else
    
% We have b_h' = 2 * t' * A + b'
t = -0.5 * (A \ bv);

c_h = t' * A * t + bv' * t + c;
%assignin('base','c_h',c_h)

z = t;
a = sqrt(-c_h / D(1,1));
b = sqrt(-c_h / D(2,2));
alpha = atan2(Q(1,2), Q(1,1)); %buna bak tekrardan

%% Stiction & Hysteresis Calculation
major_ax=a;
minor_ax=b;

stic=2*major_ax*minor_ax/sqrt(major_ax^2*sin(alpha)^2+...
   minor_ax^2*cos(alpha)^2)

hyst=2*major_ax*minor_ax/sqrt(major_ax^2*cos(alpha)^2+...
   minor_ax^2*sin(alpha)^2)
%% calculate confidence intervals
x1=x1(:); x2=x2(:);
L=length(x1);
%=========CONFIDENCE INTERVAL===========
Design_mat=[x1.^2 x2.^2 x1.*x2 x1 x2 ones(L,1)];

Coef_mat=[A(1) A(4) 2*A(2) bv(1) bv(2) c];

resid=norm(Design_mat*Coef_mat');
sig_r=resid/sqrt(L); % residuals

f=1.5; % 95% confidence interval

c_outer=c+f*sig_r;
c_inner=c-f*sig_r;

%inner ellipse
c_h_inner = t' * A * t + bv' * t + c_inner;

%outer ellipse
c_h_outer = t' * A * t + bv' * t + c_outer;


a_inner = sqrt(-c_h_inner / D(1,1));
b_inner = sqrt(-c_h_inner / D(2,2));

a_outer = sqrt(-c_h_outer / D(1,1));
b_outer = sqrt(-c_h_outer / D(2,2));

%% plotter for fitted ellipse

% form the parameter vector
npts = 100;
t = linspace(0, 2*pi, npts);

% Rotation matrix
Q2 = [cos(alpha), -sin(alpha); sin(alpha) cos(alpha)];
% Ellipse points
X_fitted = Q2 * [a * cos(t); b * sin(t)] + repmat(z, 1, npts);

figure('Name','Ellipse Fit')
% The actual plotting one-liner
h_fitted = plot(X_fitted(1,:), X_fitted(2,:), 'r');
hold on
scatter(x1,x2,'b')
hold on

prompt = 'Enter to continue';
temp_hello = input(prompt)
temp_hello



%% plotter for outer ellipse

% Outer Ellipse points
X_outer = Q2 * [a_outer * cos(t); b_outer * sin(t)] + repmat(z, 1, npts);

% The actual plotting one-liner
h_outer = plot(X_outer(1,:), X_outer(2,:), 'g--');

%% plotter for inner ellipse

% Inner Ellipse points
X_inner = (Q2 * [a_inner * cos(t); b_inner * sin(t)] + repmat(z, 1, npts));

% The actual plotting one-liner
h_inner = plot(X_inner(1,:), X_inner(2,:), 'g--');
hold off
grid on
xlabel('OP')
ylabel('PV')

%% goodness of fit
%inner_ellipse_param=fliplr(ellipse_param); %clock-wise

xv=[X_outer(1,:) NaN fliplr(X_inner(1,:))];
yv=[X_outer(2,:) NaN fliplr(X_inner(2,:))];

in = inpolygon(x1,x2,xv,yv);
goodness_of_fit=100*numel(x1(in))/length(x1);
end
end







