%Load images

x = [];
x = [x extract_patches(imread('imc.1.jpg'),8,500)];
x = [x extract_patches(imread('imc.2.jpg'),8,500)];
x = [x extract_patches(imread('imc.3.jpg'),8,500)];
x = [x extract_patches(imread('imc.4.jpg'),8,500)];
x = [x extract_patches(imread('imc.5.jpg'),8,500)];
x = [x extract_patches(imread('imc.6.jpg'),8,500)];
x = [x extract_patches(imread('imc.7.jpg'),8,500)];
x = [x extract_patches(imread('imc.8.jpg'),8,500)];
x = [x extract_patches(imread('imc.9.jpg'),8,500)];
x = [x extract_patches(imread('imc.10.jpg'),8,500)];
% Look at first 10 images for the first part
% x = [x extract_patches(imread('im.11.tif'),8,500)];

%get covariance matrix (transpose matrix such that cov produces correct result)
C = cov(transpose(x), 1);
%singular value decomposition on covariance matrix
[U,S,V] = svd(C);

%display results (top 10)
figure
colormap('gray');
for idx = 1:10
    subplot(2,5,idx);
    z = U(:,idx);
    %contrast renormalization
    z = z - min(min(z));
    z = z / max(max(z));
    imshow(reshape(z,8,8));
    set(gca,'xticklabel',[]);
    set(gca,'yticklabel',[]);
    title(sprintf('%0.5g',S(idx,idx)));

    %set(gca, 'visible', 'off'); 
end
