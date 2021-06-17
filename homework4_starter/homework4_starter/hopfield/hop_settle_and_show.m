function [  ] = settle_and_show( w, states, iters_array, h)
%SETTLE_AND_SHOW Summary of this function goes here
%   Detailed explanation goes here

iters_diff = iters_array(2:end) - iters_array(1:end-1);
n = size(states,1);
iters_diff = [0 iters_diff];
niters = length(iters_array);

settled = states;

for i=1:niters
    iters = iters_diff(i);
    
    settled = settle_hopfield_net(w, settled, iters);
    images_settled = vectors_to_images(settled, 28, 28);
    
   for j=1:n
       subaxis(n,niters,(j-1)*niters+i, 'Spacing', .0, 'Padding', 0, 'Margin', 0);
       imshow(images_settled(:,:,j));
       axis tight;
   end
end




end

