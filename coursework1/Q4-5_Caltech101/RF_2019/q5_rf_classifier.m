% Random Forest with Descriptors from K-means clustering



% generate training and testing data

showImg = 1;

PHOW_Sizes = [4 8 10];
PHOW_Step = 8;


% Caltech Dataset
close all;
imgSel = [15 15];
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

% training image data
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
num_desc = 10e5;

% Build visual vocabulary (codebook) for 'Bag-of-Words method'
desc_sel = single(vl_colsubset(cat(2,desc_tr{:}), num_desc)); % Randomly select 100k SIFT descriptors for clustering



% K-means clustering
numBins = 64;


num_class = 10;
num_elem = 15;

% K-means clustering with Euclidian distance.
[centers, assignments] = vl_kmeans(desc_sel, numBins, 'distance', 'l2');


disp('Encoding Images...')
% Vector Quantisation
data_train = zeros(num_class * num_elem, numBins + 1); % one more space for storing label.
for i = 1:num_class
    for j = 1:num_elem
        desc_ij = single(desc_tr{i, j});
        [~, num_desc] = size(desc_ij); % how many descriptors of this image?
        
        
        for k = 1:num_desc
            feature = desc_ij(:, k);
            
            min_dist = inf;
            idx = 1;
            
            for l = 1:numBins % find the nearest descriptor and assign to the codeword
                
                dist = sum((feature - centers(:,l)).^2)/length(feature);
                
                if min_dist > dist
                    min_dist = dist;
                    idx = l;
                end
            end
            data_train((i - 1)*num_elem + j, idx) = data_train((i - 1)*num_elem + j, idx) + 1;
            data_train((i - 1)*num_elem + j , end) = i;
        end
    end
end

clearvars desc_tr desc_sel



% testing image data
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



% vector quantization
data_query = zeros(num_class * num_elem, numBins + 1);
for i = 1:num_class
    for j = 1:num_elem
        desc_ij = single(desc_te{i, j});
        [~, num_desc] = size(desc_ij); % how many descriptors of this image?
        
        for k = 1:num_desc
            feature = desc_ij(:, k);
            
            min_dist = inf;
            idx = 1;
            
            for l = 1:numBins % find the nearest descriptor and assign to the codeword
                
                dist = sum((feature - centers(:,l)).^2)/length(feature);
                
                if min_dist > dist
                    min_dist = dist;
                    idx = l;
                end
            end
            data_query((i - 1)*num_elem + j, idx) = data_query((i - 1)*num_elem + j, idx) + 1;
            data_query((i - 1)*num_elem + j , end) = i;
        end
    end
end

clearvars desc_te


%% Random Forest Test

[time_train, time_test, accur] = rf_classifier_basic(data_train, data_query);


%% numTree
numTrees_payload = [4, 16, 64, 128, 256, 512];
%numTrees_payload = [1, 2, 3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
[t_train_nT, t_test_nT, accur_nT] = rf_classifier_analysis('numTree', numTrees_payload, data_train, data_query, 0);

%% numDepth
numDepth_payload = [2, 4, 8, 10, 12, 16];
[t_train_nD, t_test_nD, accur_nD] = rf_classifier_analysis('numDepth', numDepth_payload, data_train, data_query, 1);

%% numSplitNum
numSplitNum_payload = [1, 3, 10, 50, 100, 150];
numSplitNum_payload = [1, 3, 10, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180];
[t_train_nSN, t_test_nSN, accur_nSN] = rf_classifier_analysis('numSplitNum', numSplitNum_payload, data_train, data_query, 0);

