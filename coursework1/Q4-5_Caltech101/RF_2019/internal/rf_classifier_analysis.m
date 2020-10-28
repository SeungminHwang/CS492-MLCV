%rf_classifier_analysis_helper
function [time_train, time_test] = rf_classifier_analysis( MODE , payload, data_train, data_query)
    len = length(payload);
    text = '';

    % Default Settings
    param.num = 100;
    param.depth = 5;
    param.splitNum = 3;
    param.split = 'IG';
    param.weakLearner = 'Axis_aligned';

    
    tic;
    switch MODE
        case 'numTree'
            text = 'numTree';
            
            forests = cell(len);
            for i = 1:len
                param.num = payload(i);
                forests{i} = growTrees(data_train, param);
            end
            
        case 'numDepth'
            text = 'numDepth';
            
            forests = cell(len);
            for i = 1:len
                param.depth = payload(i);
                forests{i} = growTrees(data_train, param);
            end
            
        case 'numSplitNum'
            text = 'numSplitNum';
            
            forests = cell(len);
            for i = 1:len
                param.splitNum = payload(i);
                forests{i} = growTrees(data_train, param);
            end
                
            
    end
    time_train = toc;
    
    
    figure;
    tiledlayout(2, 3);
    
    time_test = 0;
    
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
        
        dt = toc;
        time_train = time_train + dt; % add test time_i

        
        %plot confusion matrix
        nexttile;
        ax = confusionchart(single(c'), single(data_query(:, end)));
        ax.Title = sprintf('%s = %d, Accuracy: %.2f',text, payload(i), accuracy);
        
        
    end
    
    
end