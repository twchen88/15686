% path(PATH,'./FastICA_21');

p1 = double(rgb2gray(imread('./vh_1.png')));
p2 = double(rgb2gray(imread('./vh_2.png')));

iptsetpref('ImshowBorder','tight')
scale = [0,255];
figure;
imshow(p1,scale);
figure;
imshow(p2,scale);

sigs = [];
sigs(1,:) = reshape(p1,1,128*192);
sigs(2,:) = reshape(p2,1,128*192);

A = [0.2 0.8; 0.5 0.4]

mixedSigs = A * sigs;

figure;
imshow(reshape(mixedSigs(1,:),128,192),[min(mixedSigs(1,:)),max(mixedSigs(1,:))]);
figure;
imshow(reshape(mixedSigs(2,:),128,192),[min(mixedSigs(2,:)),max(mixedSigs(2,:))]);

[S, A, W] = fastica(mixedSigs,'displayMode','off');

p1recovered = reshape(S(1,:),128,192);
p2recovered = reshape(S(2,:),128,192);

figure;
p1recovered = p1recovered - min(p1recovered(:));
p1recovered = 256*p1recovered / max(p1recovered(:));
p2recovered = p2recovered - min(p2recovered(:));
p2recovered = 256*p2recovered / max(p2recovered(:));
imshow(p1recovered,[min(p1recovered(:)),max(p1recovered(:))]);
figure;
imshow(p2recovered,[min(p2recovered(:)),max(p2recovered(:))]);


figure; imhist(uint8(p2), 256);
figure; imhist(uint8(p1), 256);           
figure; imhist(uint8(p1recovered), 256);       
figure; imhist(uint8(p2recovered), 256);
