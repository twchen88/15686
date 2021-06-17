%Load images

x = [];
x = [x extract_patches(imread('im.1.tif'),8,500)];
x = [x extract_patches(imread('im.2.tif'),8,500)];
x = [x extract_patches(imread('im.3.tif'),8,500)];
x = [x extract_patches(imread('im.4.tif'),8,500)];
x = [x extract_patches(imread('im.5.tif'),8,500)];
x = [x extract_patches(imread('im.6.tif'),8,500)];
x = [x extract_patches(imread('im.7.tif'),8,500)];
x = [x extract_patches(imread('im.8.tif'),8,500)];
x = [x extract_patches(imread('im.9.tif'),8,500)];
x = [x extract_patches(imread('im.10.tif'),8,500)];
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
    imshow(reshape(z, 8, 8));
    set(gca,'xticklabel',[]);
    set(gca,'yticklabel',[]);
    title(sprintf('%0.5g',S(idx,idx)));

    %set(gca, 'visible', 'off'); 
end

%display all eigenvectors as images 
figure
colormap('gray');

for idx = 1:64
    subplot(8,8, idx);
    z = U(:,idx);
    %contrast renormalization
    z = z - min(min(z));
    z = z / max(max(z));
    imshow(reshape(z, 8, 8));
    set(gca,'xticklabel',[]);
    set(gca,'yticklabel',[]);
    title(sprintf('%0.5g',S(idx,idx)));
    %set(gca, 'visible', 'off'); 
end


% TASK: you have to deal with eigenvalues to answer the questions. 
figure
e_values = sum(S); % 1x64 eigen values
total_var = sum(e_values); % total variance = trace(S)
var90 = total_var * 0.9;
var99 = total_var * 0.99;
v = 0;
varc = 0;
while v < var90
    varc = varc + 1;
    v = v + e_values(varc);
end
var_90c = varc;
while v < var99
    varc = varc + 1;
    v = v+ e_values(varc);
end
gx = [0:63];
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
step = 8;
numeig = 10;
e = 0;
error = [];
for idx = 0:(x/step)-1
    for idy = 0:(y/step)-1
        patch = im(((idx*8)+1):((idx+1)*8),((idy*8)+1):((idy+1)*8));
        
        patch =double( reshape(patch, 1,64));
        
        n_patch = zeros(64,1);
        %dot product and recreate patch
        co = sum(PC10 .* patch');
        for idcoeff = 1:numeig
             n_patch = sum(transpose(PC10 .* co));
        end
        
        %update error
        e = e + sum(sum(abs(uint8(n_patch)-uint8(patch))));
        
        %write back results
        n_patch = reshape(n_patch,8,8);
        results(((idx*8)+1):((idx+1)*8),((idy*8)+1):((idy+1)*8)) = n_patch;
    end
end

error = [error e];
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
e = 0;
PC = U(:, 1);

for idx = 0:(x/step)-1
    for idy = 0:(y/step)-1
        patch = im(((idx*8)+1):((idx+1)*8),((idy*8)+1):((idy+1)*8));
        
        patch =double( reshape(patch, 1,64));
        
        n_patch = zeros(64,1);
        %dot product and recreate patch
        co = sum(PC .* patch');
        for idcoeff = 1:numeig
             n_patch = transpose(PC * co);
        end
        
        %update error
        e = e + sum(sum(abs(uint8(n_patch)-uint8(patch))));
        
        %write back results
        n_patch = reshape(n_patch,8,8);
        results(((idx*8)+1):((idx+1)*8),((idy*8)+1):((idy+1)*8)) = n_patch;
    end
end

results1 = results;
error = [error e];

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
e = 0;
PC = U(:, 1:numeig);
for idx = 0:(x/step)-1
    for idy = 0:(y/step)-1
        patch = im(((idx*8)+1):((idx+1)*8),((idy*8)+1):((idy+1)*8));
        
        patch =double( reshape(patch, 1,64));
        
        n_patch = zeros(64,1);
        %dot product and recreate patch
        co = sum(PC .* patch');
        for idcoeff = 1:numeig
             n_patch = sum(transpose(PC .* co));
        end
        
        %update error
        e = e + sum(sum(abs(uint8(n_patch)-uint8(patch))));
        
        %write back results
        n_patch = reshape(n_patch,8,8);
        results(((idx*8)+1):((idx+1)*8),((idy*8)+1):((idy+1)*8)) = n_patch;
    end
end

results6 = results;
error = [error e];

figure
colormap('gray');
%contrast normalization
results = results - min(min(results));
results = results / max(max(results));
imshow(results);
title('Recreated with PCA (6)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numeig = 8;
results = zeros(x,y);
e = 0;
PC = U(:, 1:numeig);
for idx = 0:(x/step)-1
    for idy = 0:(y/step)-1
        patch = im(((idx*8)+1):((idx+1)*8),((idy*8)+1):((idy+1)*8));
        
        patch =double( reshape(patch, 1,64));
        
        n_patch = zeros(64,1);
        %dot product and recreate patch
        co = sum(PC .* patch');
        for idcoeff = 1:numeig
             n_patch = sum(transpose(PC .* co));
        end
        
        %update error
        e = e + sum(sum(abs(uint8(n_patch)-uint8(patch))));
        
        %write back results
        n_patch = reshape(n_patch,8,8);
        results(((idx*8)+1):((idx+1)*8),((idy*8)+1):((idy+1)*8)) = n_patch;
    end
end

results8 = results;
error = [error e];

figure
colormap('gray');
%contrast normalization
results = results - min(min(results));
results = results / max(max(results));
imshow(results);
title('Recreated with PCA (8)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numeig = varc;
results = zeros(x,y);
e = 0;
PC = U(:, 1:numeig);
for idx = 0:(x/step)-1
    for idy = 0:(y/step)-1
        patch = im(((idx*8)+1):((idx+1)*8),((idy*8)+1):((idy+1)*8));
        
        patch =double( reshape(patch, 1,64));
        
        n_patch = zeros(64,1);
        %dot product and recreate patch
        co = sum(PC .* patch');
        for idcoeff = 1:numeig
             n_patch = sum(transpose(PC .* co));
        end
        
        %update error
        e = e + sum(sum(abs(uint8(n_patch)-uint8(patch))));
        
        %write back results
        n_patch = reshape(n_patch,8,8);
        results(((idx*8)+1):((idx+1)*8),((idy*8)+1):((idy+1)*8)) = n_patch;
    end
end

results99 = results;
error = [error e];

figure
colormap('gray');
%contrast normalization
results = results - min(min(results));
results = results / max(max(results));
imshow(results);
title('Recreated with PCA (99%)');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%code for showing original image.

figure
colormap('gray');
imshow(im);
title('Original image');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% error calculation
% 10, 1, 6, 8, 99%

NM = 64 * (x/step) * (y/step);
error = error * (1/NM);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% histograms
% 1000 patches
x = [];
x = [x extract_patches(imread('im.1.tif'),8,100)];
x = [x extract_patches(imread('im.2.tif'),8,100)];
x = [x extract_patches(imread('im.3.tif'),8,100)];
x = [x extract_patches(imread('im.4.tif'),8,100)];
x = [x extract_patches(imread('im.5.tif'),8,100)];
x = [x extract_patches(imread('im.6.tif'),8,100)];
x = [x extract_patches(imread('im.7.tif'),8,100)];
x = [x extract_patches(imread('im.8.tif'),8,100)];
x = [x extract_patches(imread('im.9.tif'),8,100)];
x = [x extract_patches(imread('im.10.tif'),8,100)];

%get covariance matrix (transpose matrix such that cov produces correct result)
C = cov(transpose(x), 1);
%singular value decomposition on covariance matrix
[U,S,V] = svd(C);
%get 10 PC
PC = U(:, 1:10);
co = [];

step = 8;
numeig = 10;
for idx = 1:1000
    patch = x(:, idx);
    
    %dot product
    co = [co transpose(sum(PC .* patch))];

end

figure
colormap('gray');
for idx = 1:10
    subplot(2,5,idx);

    histogram(co(idx, :));
    title(sprintf('%0.5g',idx));
    
end

std_comp = std(co');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Pearson correlation
PC3 = U(:, 1:3);
co = [];

for idx = 1:1000
    patch = x(:, idx);
    
    %dot product
    co = [co transpose(sum(PC3 .* patch))];

end
% a1 a2, a1 a3, a2 a3
corr = [corrcoef(co(1, :), co(2, :)) corrcoef(co(1, :), co(3, :)) corrcoef(co(2, :), co(3, :))];