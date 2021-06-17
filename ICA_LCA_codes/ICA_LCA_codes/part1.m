% load("mystery_images.mat");
% 
% pic1 = reshape(double(mystery_images{1}), 1, 234 * 256);
% pic2 = reshape(double(mystery_images{2}), 1, 234 * 256);
% pic3 = reshape(double(mystery_images{3}), 1, 234 * 256);
% pic4 = reshape(double(mystery_images{4}), 1, 234 * 256);
% pic5 = reshape(double(mystery_images{5}), 1, 234 * 256);
% pic6 = reshape(double(mystery_images{6}), 1, 234 * 256);
% 
% %%% part 1 a
% S = [pic1; pic2; pic3; pic4; pic5; pic6];
% [X, A, W] = fastica(S);
% 
% for i=1:6
%     figure
%     imagesc(reshape(X(i, :), 234, 256));
% end
% 
% %%% part 1 b
% S_b = S(1:5, :);
% [X, A, W] = fastica(S_b);
% 
% for i=1:5
%     figure
%     imagesc(reshape(X(i, :), 234, 256));
% end
% 
% %%% part 1 c
% pic7 = 0.5 * (pic1 + pic2);
% S_c = [S; pic7];
% [X, A, W] = fastica(S_c);
% 
% for i=1:7
%     figure
%     imagesc(reshape(X(i, :), 234, 256));
% end

%%% part 1 d
load("IMAGES.mat");
figure
imagesc(IMAGES(:, :, 1));
figure
imagesc(IMAGES(:, :, 2));
X = [reshape(IMAGES(:, :, 1), 1, 512 * 512); reshape(IMAGES(:, :, 2), 1, 512 * 512)];
a = [78, 67];
b = [3, 5];
mixa = a * X;
mixb = b * X;
S = [mixa; mixb];
figure
imagesc(reshape(mixa, 512, 512));
figure
imagesc(reshape(mixb, 512, 512));
figure
[X_n, A, W] = fastica(S);
figure
imagesc(reshape(X_n(1, :), 512, 512));
figure
imagesc(reshape(X_n(2, :), 512, 512));



%%% part 1 e
% IM = [];
% % extract image patches
% for i=1:10
%     IM = [IM extract_patches(IMAGES(:, :, i), 12, 10000)];
% end
% 
% [X, A, W] = fastica(IM);
% 
% figure
% for i=1:144
%     subplot(12, 12, i)
%     imagesc(reshape(A(:, i), 12, 12));
% end