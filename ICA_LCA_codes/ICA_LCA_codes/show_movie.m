function [  ] = show_movie( movie, fps )
%SHOW_MOVIE Summary of this function goes here
%   Detailed explanation goes here
    [x,y,z] = size(movie);

    conv_movie = []
    
    %convert movie 
    for frame = 1:z
        %conv_movie(:,:,frame) = uint8((movie(:,:,frame).*128)+128);
        conv_movie(:,:,frame) = ((movie(:,:,frame)+1)./2);
    end
    %colormap 'gray'
    implay(conv_movie,fps);
    
end

