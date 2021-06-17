function Vfinal = mytest(learning_rule,numPatterns,size,custom_load,updatemod,updatepara,inputmod,inputpara)


% e.g. mytest('hebbian', 2, 50, [“images/bike.jpg�?, “images/car.jpg�?], 'random', [10000, 1000], 'corrupt', [1, 20, 10,10])
% e.g. mytest(‘oja’, 2, 50, [1], 'random', [10000, 1000], 'corrupt', [1, 20, 10,10])
% e.g. mytest(‘storkey’, 2, 50, [1], ‘full’, [10000, 1000], 'corrupt', [1, 20, 10,10])
% learning rule = 'hebbian' (outer product), 'storkey', 'oja'
% 2 = number of patterns to store, 50 x 50 is the image size, customize load decide the sequence and collection of the images to store, 
% updatemod =  "random", "all", "checkerboard", updatepara = [x,y] x = number of iterations, y= number of checkpoints to print out intermediate result
% inputmod = types of  preprocessing the input to be tested, "corrupt", "partial", "full". 
% inputpara = is a list   [x,y,z,w] x= which image to be tested, e.g. 1, or 2. for corrupted, y= 20 is adding 20 points, z, w useless for corrupt.
%                   for "partial", x =which image, y = size of the patch visible, (z, w) is the upper left corner coordinate of the visible patch
%                   for "full", just need to provide x, the other numbers are not used or may not needed.


num_iterations=updatepara(1);
checkpoint_number=updatepara(2);
numNeurons = size*size;

%load mypattern.mat
if length(custom_load)==numPatterns
    mypatterns = zeros(size*size,1,numPatterns);
    for i =1:numPatterns
        mypatterns(:,:,i)=load_image_by_name(custom_load(i),size);
    end
else mypatterns=load_natural_image(numPatterns,size);
end

%aa=load_image_by_name("images/bike.jpg",size);
fhi=figure('Units','pixels','Position', [100 100 numPatterns*300 300]);
colormap gray;
for i=1:numPatterns
    subplot(1,numPatterns,i);
    imagesc(reshape(mypatterns(:,:,i),size,size));
end

T = zeros(numNeurons);
disp(learning_rule);
if strcmp(learning_rule,'hebbian')
    for alpha=1:numPatterns
        s = mypatterns(:, :, alpha);
        T = T + s' * s;
    end
    T = T./numPatterns;   % this normalized by the number of patterns.
elseif strcmp(learning_rule,'storkey')
    for alpha=1:numPatterns
        %disp(alpha);
        tmp=T;
        h=T*mypatterns(:,1,alpha);
        for i=1:numNeurons
            for j=1:numNeurons
                hij=h(i)-T(i,i)*mypatterns(i,1,alpha)-T(i,j)*mypatterns(j,1,alpha);
                hji=h(j)-T(j,j)*mypatterns(j,1,alpha)-T(j,i)*mypatterns(i,1,alpha);
                T(i,j)=tmp(i,j)+(mypatterns(i,1,alpha)*mypatterns(j,1,alpha)-mypatterns(i,1,alpha)*hji-mypatterns(j,1,alpha)*hij)/numNeurons;     
            end
        end
    end
elseif strcmp(learning_rule,'oja')
    T=ones(numNeurons);
    T = T./numNeurons;    
    lr=1.1*1/numNeurons;
    for epoch=1:100
        for iter=1:numPatterns
            % to be filled in by students implementing Oja’s rule. This is using interleave learning. 
            % you will have to rearrange some codes to do sequential learning. 
            xcur = reshape(mypatterns(:, :, 1:iter), 2500, iter);
            ycur = T * xcur;
            deltaT = lr * (xcur * ycur' - T.*repmat(sum(ycur .* ycur, 2), 1, 2500)');
            %ycur = xcur' * T';
            %deltaT = lr * (xcur - T * ycur') * ycur;
            T=T+deltaT;
        end
       %  if you wish, you can enforce the connection matrix to be symmetric by setting T = T*T'  here
    end
end
% 

fhi=figure();
colormap gray;
imagesc(T);

Vss=mypatterns(:,:,1);
if strcmp(inputmod,'corrupt')
    Vss =corrupt(mypatterns(:,:,inputpara(1)),inputpara(2));
elseif strcmp(inputmod,'partial')
    Vss =partial_image(mypatterns(:,:,inputpara(1)),inputpara(2),inputpara(3),inputpara(4));
elseif strcmp(inputmod,'full')
    Vss=mypatterns(:,:,inputpara(1));
end

fhi=figure();
colormap gray;
imagesc(reshape(Vss,size,size));


Vfinal = runHopnet(T,num_iterations,checkpoint_number,Vss,updatemod);

fhi=figure();
colormap gray;
imagesc(reshape(Vfinal,size,size));

end
