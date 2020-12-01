%rf_classifier_analysis_helper
function [time_train, time_test, accuracy] = rf_classifier_basic(data_train, data_query)

    % Default Settings
    param.num = 500;
    param.depth = 5;
    param.splitNum = 3;
    param.split = 'IG';
    param.weakLearner = 'Axis-aligned';
    
    %train
    tic;
    forest = growTrees(data_train, param);
    time_train = toc;
    
    
    figure;
    tiledlayout(1, 1);
    
    
    
    
    tic;
    leaves = testTrees_fast(data_query, forest);
    for img = 1:length(data_query(:,1))
        prob = forest(1).prob(leaves(img,:), :);
        prob_mean(img, :) = mean(prob, 1);
    end

    % accuracy
    [~,c] = max(prob_mean');
    accuracy = sum(c == data_query(:, end)')/length(c);

    time_test = toc;
    
    disp(single(c'));
    disp(single(data_query(:,end)));
    
    %plot confusion matrix
    nexttile;
    ax = confusionchart(single(c'), single(data_query(:, end)));
    ax.Title = sprintf('Accuracy: %.2f',accuracy);
    
    
end