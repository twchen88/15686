function [patches,Nreal] = read_patch(IM,sz,N,mode)
% Xiong, March 31 2011

if nargin < 4
    rand_read = 1;
else
    rand_read = strcmp(mode,'random');
end
    
[H W] = size(IM);

D = sz^2;
BUFF = 4;

% read all images
patches = zeros(D,N);

if rand_read 
    % extract subimages at random from this image to make data array I
    for i = 1:N
        r = BUFF+ceil((H-sz-2*BUFF)*rand);
        c = BUFF+ceil((W-sz-2*BUFF)*rand);
        patches(:,i) = reshape(IM(r:r+sz-1,c:c+sz-1),D,1);
    end
else
    % sequentially read
    i = 1;
    for r = 1:sz:H-sz
        for c = 1:sz:W-sz
            patches(:,i) = reshape(IM(r:r+sz-1,c:c+sz-1),D,1);
            if i == N; break; end;
            i = i + 1;
        end
    end    
end

if i < N 
    Nreal = i - 1;
else
    Nreal = N;
end