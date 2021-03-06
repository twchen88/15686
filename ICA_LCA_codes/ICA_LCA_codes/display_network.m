function display_network(A,S_var)
%
%  display_network -- displays the state of the network (weights and
%                     output variances)
%
%  Usage:
%
%    display_network(A,S_var);
%
%    A = basis function matrix
%    S_var = vector of coefficient variances
%

figure(1)

[L M]=size(A);

sz=sqrt(L);

buf=1;

if floor(sqrt(M))^2 ~= M
    if M<=5
        n=1;
        m=M;
    else
        n=floor(sqrt(M));
        m=ceil(M/n);
    end
else
    m=sqrt(M);
    n=m;
end
array=-ones(buf+n*(sz+buf),buf+m*(sz+buf));

k=1;

for j=1:m
    for i=1:n
        clim=max(abs(A(:,k)));
        array(buf+(i-1)*(sz+buf)+[1:sz],buf+(j-1)*(sz+buf)+[1:sz])=...
            reshape(A(:,k),sz,sz)/clim;
        k=k+1;
        if k>M
            break;
        end
    end
end

if exist('S_var','var')
    subplot(122);
    imagesc(array,[-1 1]); colormap(gray); axis image off
    title('basis functions');
    subplot(121)
    bar(S_var), axis([0 M+1 0 max(S_var)])
    title('coeff variance');
else
    axes('position',[0 0 1 1]);
    imagesc(array,[-1 1]); colormap(gray); axis image off
    title('basis functions');
end;

drawnow

