function [ movie ] = generate_cos( freq, fps, length_in_frames, xydim, rot)
%GENERATE_COS  - This function generates a 2d cosine wave pattern movie
%   freq - spatial frequency of 2d cosine wave
%   fps -  frames per second of the resulting movie
%   length_in_frames - length of the movie to produce 
%   xydim - size of the movie to produce (square)
%   rot - rotation degrees of VIEWPORT (negate to rotate image)
%   output -
%      movie - movie containing the 2d cosine wave pattern. 
%              3d matrix, 1st,2nd extent are x,y of each frame, 
%                         3rd extent is the frame number
%               ie: movie(:,:,1) will extract the first frame
    movie = [];

    %generate rotation matrixies
    x_loc_mat = ones(xydim,1)*(0:xydim-1);
    y_loc_mat = (xydim-1:-1:0)'*ones(1,xydim);
    
    rot_mat = x_loc_mat.*cos(rot) - y_loc_mat.*sin(rot);
    
    for frame = 1:length_in_frames
        %generate evaluation matrix
        eval_mat = (rot_mat .* (2*pi*freq)/xydim) + ...
                ((mod(frame,fps)/fps)*2*pi*freq)  ;
        
        %evaluate matrix
        movie(:,:,frame) = cos(eval_mat);
                
    end
    
    

end

