function [r,x,y]=rf_tuning(phi)
%find prefered orientation and prefered spatial frequency
N=length(phi);
sz=sqrt(N);
rf=reshape(phi,sz,sz);
x=0:0.1:6;
y=linspace(-pi,pi,30);
r=zeros(length(y),length(x));

for m=1:length(y)
    tmp_ori=y(m);
    for n=1:length(x)
        tmp_freq=x(n);
        movie=generate_cos(tmp_freq,100,200,sz,tmp_ori);
        temp=zeros(1,size(movie,3));
        for k=1:size(movie,3)
            tmp_img=movie(:,:,k);
            temp(k)=sum(sum(rf.*tmp_img))/sum(sum(tmp_img.^2));
        end
        r(m,n)=mean(temp);
    end
end
        