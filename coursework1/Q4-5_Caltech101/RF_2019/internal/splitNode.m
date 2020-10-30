function [node,nodeL,nodeR] = splitNode(data,node,param)
% Split node

visualise = 0;

% Initilise child nodes
iter = param.splitNum;
nodeL = struct('idx',[],'t',nan,'dim',0,'prob',[]);
nodeR = struct('idx',[],'t',nan,'dim',0,'prob',[]);

if length(node.idx) <= 5 % make this node a leaf if has less than 5 data points
    node.t = nan;
    node.dim = 0;
    return;
end

idx = node.idx;
data = data(idx,:);
[N,D] = size(data);
ig_best = -inf; % Initialise best information gain
idx_best = [];

for n = 1:iter
    % Split function - Modify here and try other types of split function
    switch(param.weakLearner)
        case 'Axis-aligned'
            dim = randi(D-1); % Pick one random dimension
            d_min = single(min(data(:,dim))) + eps; % Find the data range of this dimension
            d_max = single(max(data(:,dim))) - eps;
            t = d_min + rand*((d_max-d_min)); % Pick a random value within the range as threshold
            idx_ = data(:,dim) < t;
        case '2-pixel'
            dim = randperm((D-1), 2);
            idx_i = dim(1);
            idx_j = dim(2);
            
            x_i = data(:, idx_i);
            x_j = data(:, idx_j);
            
            dist = x_i - x_j;
            
            d_min = single(min(dist)) + eps;
            d_max = single(max(dist)) - eps;
            
            t = d_min + rand * ((d_max - d_min));
            idx_ = dist < t;
        case 'linear'
            cond = true;
            while cond
                dim = randperm(D-1, 2);
                
                t = randn(3, 1);
                idx_ = [data(:, dim), ones(N, 1)]*t < 0;
                cond = isequal(idx_, zeros(size(idx_))) || isequal(idx_, ones(size(idx_)));
            end
        case 'non-linear'
            cond = true;
            while cond
                dim = randperm(D-1, 2);
                t = randn(3, 3);
                %t = randn(3, 1);
                idx_ = zeros(0);
                for i = 1:N
                    isLeft = [data(i, dim), 1]*t*[data(i, dim), 1]' < 0;
                    if isLeft
                        idx_ = cat(1, idx_, i);
                    end
                end
                %disp((phi*t*phi'));
                size_idx = size(idx_);
                cond = isequal(idx_, zeros(size(idx_))) || isequal(idx_, ones(size(idx_))) || size_idx(1) < 3;
            end
    end
    
    
    ig = getIG(data,idx_); % Calculate information gain
    
    if visualise
        visualise_splitfunc(idx_,data,dim,t,ig,n);
        pause();
    end
    
    if (sum(idx_) > 0 & sum(~idx_) > 0) % We check that children node are not empty
        [node, ig_best, idx_best] = updateIG(node,ig_best,ig,t,idx_,dim,idx_best);
    end
    
end

nodeL.idx = idx(idx_best);
nodeR.idx = idx(~idx_best);

if visualise
    visualise_splitfunc(idx_best,data,dim,t,ig_best,0)
    fprintf('Information gain = %f. \n',ig_best);
    pause();
end

end

function ig = getIG(data,idx) % Information Gain - the 'purity' of data labels in both child nodes after split. The higher the purer.
L = data(idx);
R = data(~idx);
H = getE(data);
HL = getE(L);
HR = getE(R);
ig = H - sum(idx)/length(idx)*HL - sum(~idx)/length(idx)*HR;
end

function H = getE(X) % Entropy
cdist= histc(X(:,1:end), unique(X(:,end))) + 1;
cdist= cdist/sum(cdist);
cdist= cdist .* log(cdist);
H = -sum(cdist);
end

function [node, ig_best, idx_best] = updateIG(node,ig_best,ig,t,idx,dim,idx_best) % Update information gain
if ig > ig_best
    ig_best = ig;
    node.t = t;
    node.dim = dim;
    idx_best = idx;
else
    idx_best = idx_best;
end
end