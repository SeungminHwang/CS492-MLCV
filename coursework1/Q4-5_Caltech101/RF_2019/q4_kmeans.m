[train_data, test_data] = getData('Caltech');
%% 



cnt = 1;
figure;
tiledlayout(10, 3);
for i = 1:10
    for j = 1:3
        nexttile;
        bar(train_data((i - 1)*10 + j, 1:end - 1));
        cnt = cnt + 1;
        
        
    end
end