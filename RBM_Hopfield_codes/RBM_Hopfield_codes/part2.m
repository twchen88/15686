load("data.mat");



%n = {1, 10, 50, 200};
rbm = initialize_rbm(784, 500, 100);
rbm_n = train_rbm(rbm, data_1000, 200);
rbm_visualize_weights(rbm_n.w, 28, 28);
im = rbm_reconstruct(rbm_n, data_100);

figure
for i=1:100
    subplot(10, 10, i)
    imagesc(reshape(im(i, :), 28, 28));
end

% n = 200, p = 0.1 or 0.01
rbm = initialize_rbm(784, 500, 100, 0.01, 1);
rbm_n = train_rbm(rbm, data_1000, 200);
rbm_visualize_weights(rbm_n.w, 28, 28);
im = rbm_reconstruct(rbm_n, data_100);

figure
for i=1:100
    subplot(10, 10, i)
    imagesc(reshape(im(i, :), 28, 28));
end