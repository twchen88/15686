load("data.mat");

% part 1a
figure
for i=1:10
    subplot(5, 2, i)
    imagesc(reshape(data_10(i, :), 28, 28));
end
