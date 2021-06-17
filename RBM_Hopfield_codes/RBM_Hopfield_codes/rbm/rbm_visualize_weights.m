function [] = rbm_visualize_weights(w, dim1, dim2)
%RBM_VISUALIZE_WEIGHTS visualizes the weights of an RBM
% 
% INPUTS:
%   w ......... : an RBM weight matrix. Should of size (#visible units by #
%                   hidden units)
%   dim1 ...... : vertical dimension of the images
%   dim2 ...... : horizontal dimension of the images

% Make sure the product of dim1*dim2 is the same as the number of visible
% units
if(~isequal(dim1*dim2, size(w,1)))
    error('rbm_visualize_weights: the number of visible units must be the product of dim1 and dim2')
end

% Compute the receptive fields of the first 100 hidden units. This should be
% a (dim1 by dim2 by 100) matrix.
receptive_fields = [];
for i=1:100
    receptive_fields(:, :, i) = reshape(w(:, 1), dim1, dim2);
end

for i=1:100
    subaxis(10,10,i, 'Spacing', .0, 'Padding', 0, 'Margin', 0);
    imagesc(receptive_fields(:,:,i));
    axis off;
end
    

end

