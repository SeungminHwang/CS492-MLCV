%rf_classifier_analysis_helper
function [time_train, time_test, accur] = rf_classifier_analysis( MODE , payload, data_train, data_query, showplt)
    len = length(payload);
    text = '';

    % Default Settings
    param.num = 100;
    param.depth = 5;
    param.splitNum = 3;
    param.split = 'IG';
    param.weakLearner = 'Axis-aligned';

    
    time_train = zeros(1, len);
    time_test = zeros(1, len);
    accur = zeros(1, len);
    
    switch MODE
        case 'numTree'
            text = 'numTree';
            
            forests = cell(len);
            for i = 1:len
                tic; % time measure starts
                param.num = payload(i);
                forests{i} = growTrees(data_train, param);
                time_train(i) = toc;
                
            end
            
        case 'numDepth'
            text = 'numDepth';
            
            forests = cell(len);
            for i = 1:len
                tic; % time measure starts
                param.depth = payload(i);
                forests{i} = growTrees(data_train, param);
                time_train(i) = toc;
            end
            
        case 'numSplitNum'
            text = 'numSplitNum';
            
            forests = cell(len);
            for i = 1:len
                tic; % time measure starts
                param.splitNum = payload(i);
                forests{i} = growTrees(data_train, param);
                time_train(i) = toc;
            end
                
            
    end
    
    
    figure;
    if showplt
        tiledlayout(2, 3);
    end
    
    for i = 1:len
        tic;
        leaves = testTrees_fast(data_query, forests{i});
        for img = 1:length(data_query(:,1))
            prob = forests{i}(1).prob(leaves(img,:), :);
            prob_mean(img, :) = mean(prob, 1);
        end
        
        % accuracy
        [~,c] = max(prob_mean');
        accuracy = sum(c == data_query(:, end)')/length(c);
        
        accur(i) = accuracy;
        time_test(i) = toc;
        
        

        if showplt
            %plot confusion matrix
            nexttile;
            ax = confusionchart(single(c'), single(data_query(:, end)));
            ax.Title = sprintf('%s = %d, Accuracy: %.2f',text, payload(i), accuracy);
        end
        
    end
    
    
end