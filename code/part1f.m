%Load images

x = [];
x = [x extract_patches(imread('im.1.tif'),16,500)];
x = [x extract_patches(imread('im.2.tif'),16,500)];
x = [x extract_patches(imread('im.3.tif'),16,500)];
x = [x extract_patches(imread('im.4.tif'),16,500)];
x = [x extract_patches(imread('im.5.tif'),16,500)];
x = [x extract_patches(imread('im.6.tif'),16,500)];
x = [x extract_patches(imread('im.7.tif'),16,500)];
x = [x extract_patches(imread('im.8.tif'),16,500)];
x = [x extract_patches(imread('im.9.tif'),16,500)];
x = [x extract_patches(imread('im.10.tif'),16,500)];
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
    imshow(reshape(z,16,16));
    set(gca,'xticklabel',[]);
    set(gca,'yticklabel',[]);
    title(sprintf('%0.5g',S(idx,idx)));

    %set(gca, 'visible', 'off'); 
end


% TASK: you have to deal with eigenvalues to answer the questions. 
figure
e_values = sum(S); % 1x256 eigen values

gx = [0:255];
plot(gx, e_values);
title("Eigenvalues");
xlabel("rank");
ylabel("eigenvalue");
PC10 = U(:, 1:10); % take the first ten eigenvectors (pc basis) (cols)

% codes for taking Image 11 and uses 10 principal components to reconstruct it.
% one line of code commented out.

im = imread('im11.tif');
[x y] = size(im);
results = zeros(x,y);
step = 16;
numeig = 10;

for idx = 0:(x/step)-1
    for idy = 0:(y/step)-1
        patch = im(((idx*16)+1):((idx+1)*16),((idy*16)+1):((idy+1)*16));
        
        patch =double( reshape(patch, 1,256));
        
        n_patch = zeros(256,1);
        %dot product and recreate patch
        co = sum(PC10 .* patch');
        for idcoeff = 1:numeig
             n_patch = sum(transpose(PC10 .* co));
        end
        
        %write back results
        n_patch = reshape(n_patch,16,16);
        results(((idx*16)+1):((idx+1)*16),((idy*16)+1):((idy+1)*16)) = n_patch;
    end
end

results10 = results;

figure
colormap('gray');
%contrast normalization
results = results - min(min(results));
results = results / max(max(results));
imshow(results);
title('Recreated with PCA (10)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numeig = 1;
results = zeros(x,y);
PC = U(:, 1);

for idx = 0:(x/step)-1
    for idy = 0:(y/step)-1
        patch = im(((idx*16)+1):((idx+1)*16),((idy*16)+1):((idy+1)*16));
        
        patch =double( reshape(patch, 1,256));
        
        n_patch = zeros(256,1);
        %dot product and recreate patch
        co = sum(PC .* patch');
        for idcoeff = 1:numeig
             n_patch = transpose(PC * co);
        end
        
        
        %write back results
        n_patch = reshape(n_patch,16,16);
        results(((idx*16)+1):((idx+1)*16),((idy*16)+1):((idy+1)*16)) = n_patch;
    end
end

results1 = results;

figure
colormap('gray');
%contrast normalization
results = results - min(min(results));
results = results / max(max(results));
imshow(results);
title('Recreated with PCA (1)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numeig = 6;
results = zeros(x,y);
PC = U(:, 1:numeig);
for idx = 0:(x/step)-1
    for idy = 0:(y/step)-1
        patch = im(((idx*16)+1):((idx+1)*16),((idy*16)+1):((idy+1)*16));
        
        patch =double( reshape(patch, 1,256));
        
        n_patch = zeros(256,1);
        %dot product and recreate patch
        co = sum(PC .* patch');
        for idcoeff = 1:numeig
             n_patch = sum(transpose(PC .* co));
        end
        
        %write back results
        n_patch = reshape(n_patch,16,16);
        results(((idx*16)+1):((idx+1)*16),((idy*16)+1):((idy+1)*16)) = n_patch;
    end
end

results6 = results;

figure
colormap('gray');
%contrast normalization
results = results - min(min(results));
results = results / max(max(results));
imshow(results);
title('Recreated with PCA (6)');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%code for showing original image.

figure
colormap('gray');
imshow(im);
title('Original image');
