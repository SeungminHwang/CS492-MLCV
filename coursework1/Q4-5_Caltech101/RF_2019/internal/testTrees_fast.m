function label = testTrees_fast(data,tree)
% Faster version - pass all data at same time
cnt = 1;

weaklearner = 'Axis-aligned';

switch weaklearner
    case 'Axis-aligned'
        for T = 1:length(tree)
            idx{1} = 1:size(data,1);
            for n = 1:length(tree(T).node);
                if ~tree(T).node(n).dim
                    leaf_idx = tree(T).node(n).leaf_idx;
                    if ~isempty(tree(T).leaf(leaf_idx))
                        label(idx{n}',T) = tree(T).leaf(leaf_idx).label;
                    end
                    continue;
                end

                idx_left = data(idx{n},tree(T).node(n).dim) < tree(T).node(n).t;
                idx{n*2} = idx{n}(idx_left');
                idx{n*2+1} = idx{n}(~idx_left');
            end
        end
    case 'linear'
        for T = 1:length(tree)
            idx{1} = 1:size(data, 1);
            for n = 1:length(tree(T).node)
                if ~tree(T).node(n).dim
                    leaf_idx = tree(T).node(n).leaf_idx;
                    if ~isempty(tree(T).leaf(leaf_idx))
                        label(idx{n}', T) = tree(T).leaf(leaf_idx).label;              
                    end
                    continue;
                end
                t = tree(T).node(n).t;
                dim1 = tree(T).node(n).dim(1);
                dim2 = tree(T).node(n).dim(2);
                idx_left = data(idx{n},dim1)*t(1) + data(idx{n},dim2)*t(2) + ones(size(data(idx{n},:),1), 1)*t(3) < 0;
                idx{n*2} = idx{n}(idx_left');
                idx{n*2+1} = idx{n}(~idx_left');
            end
        end
        case '2-pixel'
        for T = 1:length(tree)
            idx{1} = 1:size(data, 1);
            for n = 1:length(tree(T).node)
                if ~tree(T).node(n).dim
                    leaf_idx = tree(T).node(n).leaf_idx;
                    if ~isempty(tree(T).leaf(leaf_idx))
                        label(idx{n}', T) = tree(T).leaf(leaf_idx).label;              
                    end
                    continue;
                end
                
                
                t = tree(T).node(n).t;
                dim1 = tree(T).node(n).dim(1);
                dim2 = tree(T).node(n).dim(2);
                idx_left = data(idx{n},tree(T).node(n).dim(1)) - data(idx{n},tree(T).node(n).dim(2)) < tree(T).node(n).t;
                idx{n*2} = idx{n}(idx_left');
                idx{n*2+1} = idx{n}(~idx_left');
            end
        end
        
        case 'non-linear'
        for T = 1:length(tree)
            idx{1} = 1:size(data, 1);
            for n = 1:length(tree(T).node)
                if ~tree(T).node(n).dim
                    leaf_idx = tree(T).node(n).leaf_idx;
                    if ~isempty(tree(T).leaf(leaf_idx))
                        label(idx{n}', T) = tree(T).leaf(leaf_idx).label;              
                    end
                    continue;
                end
                
                
                t = tree(T).node(n).t;
                disp(tree(T).node(n).dim);
                dim1 = tree(T).node(n).dim(1);
                dim2 = tree(T).node(n).dim(2);
                
                idx_left = zeros(0);
                num = length(idx{n});
                for i = 1:num
                    isLeft = [data(i, dim), 1]*t*[data(i, dim), 1]' < 0;
                    if isLeft
                        idx_left = cat(1, idx_left, i);
                    end
                end
                
                phi = [data(idx{n}, tree(T).node(n).dim(1)), data(idx{n}, tree(T).node(n).dim(2)), ones(length(idx{n}), 1)];
                idx_left = phi*t*phi' < 0;
                idx{n*2} = idx{n}(idx_left');
                idx{n*2+1} = idx{n}(~idx_left');
            end
        end


end

