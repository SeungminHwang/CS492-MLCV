function [ data_train, data_query ] = getData( MODE )
% Generate training and testing data

% Data Options:
%   1. Toy_Gaussian
%   2. Toy_Spiral
%   3. Toy_Circle
%   4. Caltech 101

showImg = 0; % Show training & testing images and their image feature vector (histogram representation)

PHOW_Sizes = [4 8 10]; % Multi-resolution, these values determine the scale of each layer.
PHOW_Step = 8; % The lower the denser. Select from {2,4,8,16}

switch MODE
    case 'Toy_Gaussian' % Gaussian distributed 2D points
        %rand('state', 0);
        %randn('state', 0);
        N= 150;
        D= 2;
        
        cov1 = randi(4);
        cov2 = randi(4);
        cov3 = randi(4);
        
        X1 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov1 0;0 cov1]);
        X2 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov2 0;0 cov2]);
        X3 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov3 0;0 cov3]);
        
        X= real([X1; X2; X3]);
        X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Toy_Spiral' % Spiral (from Karpathy's matlab toolbox)
        
        N= 50;
        t = linspace(0.5, 2*pi, N);
        x = t.*cos(t);
        y = t.*sin(t);
        
        t = linspace(0.5, 2*pi, N);
        x2 = t.*cos(t+2);
        y2 = t.*sin(t+2);
        
        t = linspace(0.5, 2*pi, N);
        x3 = t.*cos(t+4);
        y3 = t.*sin(t+4);
        
        X= [[x' y']; [x2' y2']; [x3' y3']];
        X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Toy_Circle' % Circle
        
        N= 50;
        t = linspace(0, 2*pi, N);
        r = 0.4
        x = r*cos(t);
        y = r*sin(t);
        
        r = 0.8
        t = linspace(0, 2*pi, N);
        x2 = r*cos(t);
        y2 = r*sin(t);
        
        r = 1.2;
        t = linspace(0, 2*pi, N);
        x3 = r*cos(t);
        y3 = r*sin(t);
        
        X= [[x' y']; [x2' y2']; [x3' y3']];
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Caltech' % Caltech dataset
        close all;
        imgSel = [15 15]; % randomly select 15 images each class without replacement. (For both training & testing)
        folderName = './Caltech_101/101_ObjectCategories';
        classList = dir(folderName);
        classList = {classList(3:end).name}; % 10 classes
        
        disp('Loading training images...')
        % Load Images -> Description (Dense SIFT)
        cnt = 1;
        
        %showImg = false; % Don't show the image
        if showImg
            figure('Units','normalized','Position',[.05 .1 .4 .9]);
            suptitle('Training image samples');
        end
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            imgIdx{c} = randperm(length(imgList));
            imgIdx_tr = imgIdx{c}(1:imgSel(1));
            imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));
            
            for i = 1:length(imgIdx_tr)
                I = imread(fullfile(subFolderName,imgList(imgIdx_tr(i)).name));
                
                % Visualise
                if i < 6 & showImg
                    subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
                    imshow(I);
                    cnt = cnt+1;
                    drawnow;
                end
                
                if size(I,3) == 3
                    I = rgb2gray(I); % PHOW work on gray scale image
                end
                
                % For details of image description, see http://www.vlfeat.org/matlab/vl_phow.html
                [~, desc_tr{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step); %  extracts PHOW features (multi-scaled Dense SIFT)
            end
        end
        
        disp('Building visual codebook...')
        num_desc = 10e4;
        
        % Build visual vocabulary (codebook) for 'Bag-of-Words method'
        desc_sel = single(vl_colsubset(cat(2,desc_tr{:}), num_desc)); % Randomly select 100k SIFT descriptors for clustering
        %disp(size(desc_sel));
        
        % K-means clustering
        numBins = 7; % for instance,
        
        
        % write your own codes here
        % ...
        % TODO!!
        
        num_class = 10; % For example, there are 10 classes
        num_elem = 15; % Each class has 15 elements
        
        
        % value to return
        % data_train: n by ((hist, label))
        %           : ex) data_train[k][
        %data_train = cell(num_class*num_elem, 3);
        
        
        % https://www.vlfeat.org/matlab/vl_kmeans.html
        % l1 for manhattan distance
        % l2 for euclidian distance
        [centers, assignments] = vl_kmeans(desc_sel, numBins, 'distance', 'l2');
        disp(size(centers));
        
        %{
        
        for i = 1:num_class
            disp(i);
            for j = 1:num_elem
                % Do some process for each Img
                
                
                
                
                % descriptors of class i, jth img
                % 128 by (num of descriptor) matrix
                % number of descriptors may be different by image
                desc_ij = double(desc_tr{i, j}); 
                
                
                % 1. K-means clustering(using internal lib)
                X = transpose(desc_ij);
                [idx, C] = kmeans(X, numBins);
                
                
                % 2. Assign the nearest descriptor for each class mean
                [K, d] = size(C); % k by n matrix C, k is num of clusters
                
                codewords = cell(1, K);
                hist = zeros(1, K);
                
                
                for k = 1:K
                    cluster_center = C(k,:); % extract kth row(cluster center vector in d dim)
                    descriptors = transpose(desc_sel); % 100K descriptors
                    
                    % compute Euclidean distance
                    distances = sqrt(sum(bsxfun(@minus, descriptors, cluster_center).^2, 2));
                    
                    % find the closest
                    closest = descriptors(find(distances == min(distances)), :);
                    
                    
                    % assign value to hist_matrix and codewords
                    [freq, temp] = size(X(idx == k, 1));
                    hist(k) = freq;
                    codewords(k) = {closest(1,:)};
                    
                end
                
                % training Dataset
                data_train((i - 1)*15 + j,:) = {hist, codewords, i};
                %disp(data_train((i - 1)*15 + j, :));
                
            end
        end
        %}
        
        
                  
        disp('Encoding Images...')
        % Vector Quantisation
        
        % write your own codes here
        % ...
        
        data_train = zeros(num_class*num_elem, numBins);
        for i = 1:num_class
            for j = 1:num_elem
                desc_ij = double(desc_tr{i, j});
                [d, num_desc] = size(desc_ij);
                %disp(size(desc_ij));
                
                
                for k = 1:num_desc
                    feature = desc_ij(:,k);
                    min_dist = inf;
                    idx = 1;
                    for l = 1:numBins % iterate for each classes
                        dist = sum((feature - centers(:,l)).^2)/length(feature);
                        
                        if min_dist > dist
                            min_dist = dist;
                            idx = l;
                        end
                    end
                    data_train((i-1)*num_elem+j,idx) = data_train((i-1)*num_elem+j,idx) + 1;
                end
                
            end
        end
        
        
  
        
        % Clear unused varibles to save memory
        clearvars desc_tr desc_sel
end

switch MODE
    case 'Caltech'
        if showImg
        figure('Units','normalized','Position',[.05 .1 .4 .9]);
        suptitle('Test image samples');
        end
        disp('Processing testing images...');
        cnt = 1;
        % Load Images -> Description (Dense SIFT)
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));
            
            for i = 1:length(imgIdx_te)
                I = imread(fullfile(subFolderName,imgList(imgIdx_te(i)).name));
                
                % Visualise
                if i < 6 & showImg
                    subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
                    imshow(I);
                    cnt = cnt+1;
                    drawnow;
                end
                
                if size(I,3) == 3
                    I = rgb2gray(I);
                end
                [~, desc_te{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step);
            
            end
            
        end
        
        %suptitle('Testing image samples');
%                 if showImg
%             figure('Units','normalized','Position',[.5 .1 .4 .9]);
%         suptitle('Testing image representations: 256-D histograms');
%         end

        % Quantisation
        
        % write your own codes here
        % ...
        num_class = 10; % For example, there are 10 classes
        num_elem = 15; % Each class has 15 elements
        
        
        % value to return
        %data_query = cell(num_class*num_elem, 3);
        
        data_query = zeros(num_class*num_elem, numBins);
        for i = 1:num_class
            for j = 1:num_elem
                desc_ij = double(desc_te{i, j});
                [d, num_desc] = size(desc_ij);
                
                
                for k = 1:num_desc
                    feature = desc_ij(:,k);
                    min = inf;
                    idx = 1;
                    for l = 1:numBins % iterate for each classes
                        dist = sum((feature - centers(:,l)).^2)/length(feature);
                        if min > dist
                            min = dist;
                            idx = l;
                        end
                    end
                    data_query((i-1)*num_elem+j,idx) = data_query((i-1)*num_elem+j,idx) + 1;
                end
                
            end
        end
        
        
        
        
    otherwise % Dense point for 2D toy data
        xrange = [-1.5 1.5];
        yrange = [-1.5 1.5];
        inc = 0.02;
        [x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
        data_query = [x(:) y(:) zeros(length(x)^2,1)];
end
end

