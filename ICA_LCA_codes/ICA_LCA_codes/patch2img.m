function new_img=patch2img(img_patch)
%% change image patches to image

[mrow,mcol]=size(img_patch);
sz=sqrt(mrow);
N=sqrt(mcol)*sz;
k=1;
new_img=zeros(N,N);
for r=1:sz:(N-sz)
    for c=1:sz:(N-sz)
        new_img(r:(r+sz-1),c:(c+sz-1))=reshape(img_patch(:,k),sz,sz);
        k=k+1;
    end
end