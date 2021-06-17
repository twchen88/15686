% extract_patches.m - extracts patches from an image to form
% data matrix X 
%
% function X = extract_patches(im,sz,K)
%
% im = image
% sz = size of patch (assumed to be square)
% K = number of patches to extract
%
% X = output data matrix - (each column is a rastered image patch)


function X = extract_patches(im,sz,K)
[imsize_r,imsize_c]=size(im);
BUFF=4;

L=sz^2;

X=zeros(L,K);

for k=1:K
    r=BUFF+ceil((imsize_r-sz-2*BUFF)*rand);
    c=BUFF+ceil((imsize_c-sz-2*BUFF)*rand);
    X(:,k)=reshape(im(r:r+sz-1,c:c+sz-1),L,1);
end
