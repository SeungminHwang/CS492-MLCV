% K-means codebook source

%[data_train, data_test] = getData('Caltech');




% parameters
showImg = 0;
PHOW_Sizes = [4 8 10]; % Multi-resolution, these values determine the scale of each layer.
PHOW_Step = 8; % The lower the denser. Select from {2,4,8,16}


% Load Caltech101 Dataset
close all;

imgSel = [15 15]; % randomly select 15 images each class without replacement. (For both training & testing)
folderName = './Caltech_101/101_ObjectCategories';
classList = dir(folderName);
classList = {classList(3:end).name}; % 10 classes


disp('Loading training images...')
% Load Images -> Description (Dense SIFT)
cnt = 1;


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
    
    for i = 1: length(imgIdx_tr)
        I = imread(fullfile(subFolderName,imgList(imgIdx_tr(i)).name));

        % Visualise
        if i < 6 & showImg
            subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
            imshow(I);
            cnt = cnt+1;
            drawnow;
        end
        
        if size(I, 3) == 3
            I = rgb2gray(I);
        end
        % For details of image description, see http://www.vlfeat.org/matlab/vl_phow.html
        [~, desc_tr{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step); %  extracts PHOW features (multi-scaled Dense SIFT)

    end
    
end



disp('Building visual codebook...');
num_desc = 10e4; % total number of desctiptors

% Build visual vocabulary (codebook) for 'Bag-of-Words method'
desc_sel = single(vl_colsubset(cat(2, desc_tr{:}), num_desc));

% K-means clustering
numBins = 30; % number of clusters


% write your own codes here

num_class = 10;
num_elem = 15;

hist_matrix = zeros(num_class*num_elem, numBins); % to save frequencies
codewords = zeros(num_class*num_elem, numBins, 128); % to save corresponding descriptor vector

for i = 1:num_class
    for j = 1:num_elem
        desc_ij = double(desc_tr{i, j}); % descriptors of class i, j th img
        % 128 by ??? dim matrix.
        % number of descriptors may differ by each img
        
        
        % 1. Let's do k-means clustering with this data
        X = transpose(desc_ij);
        [idx, C] = kmeans(X, numBins); % C is matrix of row vectors. each row is d dimensional center position vector
        
        %disp(C(1,:)); % 1 by 128 row vector( centre for cluster 1)
        
        
        % 2. Assign the nearest descriptor
        [n, m] = size(C); % n be the number of clusters
        for k = 1:n
            cluster_center = C(k,:);
            descriptors = transpose(desc_sel);% for matching dimension
            
            % compute Euclidean dist
            distances = sqrt(sum(bsxfun(@minus, descriptors, cluster_center).^2, 2));
            
            % find closest
            closest = descriptors(find(distances == min(distances)), :);
            
            temp = size(X(idx == k,1)); % get frequency of this cluster
            hist_matrix(i*j,k) = temp(1);
            codewords(i*j, k, :) = closest(1,:);
        end
        
        
        %{
        figure;
        x = 1 : length(hist_matrix(i*j, :));
        plot(x, hist_matrix(i*j,:));
        %}
        %disp(hist_matrix(i*j, :));
        
    end
end
        
        

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
    
    % write some processing code for the test data


end

    %suptitle('Testing image samples');
    %                 if showImg
    %             figure('Units','normalized','Position',[.5 .1 .4 .9]);
    %         suptitle('Testing image representations: 256-D histograms');
    %         end

    % Quantisation

    % write your own codes here
    % ...




