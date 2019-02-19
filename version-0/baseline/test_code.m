clc
clear all
close all
M=100;
%D=[eye(M) dctmtx(M)];
 % N = size(D, 2);
 D=randn(M,2*M);
 D=normc(D);
  N = size(D, 2);  


%  conherence(D)
%  

 j=1;
for k=1:10:M
   
    tomp=0;
    tbp=0;
    for i=1:20
    xo=zeros(2*M,1);
Supp0 = randperm(N);
r = Supp0(1:k);
xo(r)=randn(k,1);
xo=xo/norm(xo);
y0=D*xo;
xOMP = OMP(D, y0, 1e-12, 1000);


tomp=norm(abs(xo-xOMP))/norm(xo)+tomp;


    end
    omp(j)=tomp/20;

    j=j+1
end

x_axis = 1:10:M;
figure;
plot(x_axis, omp, 'r-'); hold on;


