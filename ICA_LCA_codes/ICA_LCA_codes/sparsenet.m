% sparsenet.m - Olshausen & Field sparse coding algorithm
% Courtesy of Olshausen
% Before running you must first load the training data array IMAGES

load 'IMAGES.mat'

num_trials =  5000;  %5000
if ~exist('batch_size','var')
    batch_size = 200;
end

[imsize, imsize2, num_images] = size(IMAGES);
BUFF = 4;

% number of inputs (dim of patches)
D = 12*12;
sz = sqrt(D);

% number of outputs (num of basis)

if ~exist('M','var')
    M = D;
end

% initialize basis functions (comment out these lines if you wish to
% pick up where you left off)
Phi = randn(D,M);

Phi = Phi*diag(1./sqrt(sum(Phi.*Phi)));

% learning rate (start out large, then lower as solution converges), but
% here we probably just use eta = 1.0
eta = 2.0;

% lambda
% lambda = 0.1;

a_var = ones(M,1);
var_eta = .1;

I = zeros(D,batch_size);

display_every = 50;
display_network(Phi,a_var);

for t = 1:num_trials
    
    % % choose an image for this batch
    % imi = ceil(num_images*rand);
    %
    % % extract subimages at random from this image to make data array I
    % for i = 1:batch_size
    %     r = BUFF+ceil((imsize-sz-2*BUFF)*rand);
    %     c = BUFF+ceil((imsize-sz-2*BUFF)*rand);
    %     I(:,i) = reshape(IMAGES(r:r+sz-1,c:c+sz-1,imi),D,1);
    % end
    
    imi = ceil(num_images*rand);
    I = read_patch(IMAGES(:,:,imi),sz,batch_size);
    
    % calculate coefficients for these data via LCA
    ahat = sparsify(I,Phi,lambda);
    
    % calculate residual error
    R = I-Phi*ahat;     % Line commented out.
    
    dPhi = zeros(D,M,size(R,2));
    
    
    % update bases, learning rule here
    for i = 1:size(R,2)
        dPhi(:,:,i) = R(:,i)*ahat(:,i)';
    end
    dPhi = mean(dPhi,3);
    
    %update bases
    Phi = Phi + dPhi;
    Phi = Phi*diag(1./sqrt(sum(Phi.*Phi))); % normalize bases
    
    % accumulate activity statistics
    a_var = (1-var_eta)*a_var + var_eta*mean(ahat.^2,2);
    av_var = var_eta*mean(ahat.^2,2);
    
    %     display
    if (mod(t,display_every)==0)
        display_network(Phi,a_var);
        disp(t);
    end
end


