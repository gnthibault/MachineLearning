function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
sizeGrid = 10
paramGrid = zeros(sizeGrid,sizeGrid,2);
resultGrid = zeros(sizeGrid,sizeGrid);

paramGrid(1,1,:)=[0.01 0.01];
for m = 1:sizeGrid
    if m > 1
        paramGrid(m,1,1)=paramGrid(m-1,1,1)*3.3333;
        paramGrid(m,1,2)=paramGrid(m-1,1,2);
    end
    for n = 1:sizeGrid
        if n > 1
            paramGrid(m,n,1)=paramGrid(m,n-1,1);
            paramGrid(m,n,2)=paramGrid(m,n-1,2)*3.3333;
        end
        model= svmTrain(X, y, paramGrid(m,n,1), @(x1, x2) gaussianKernel(x1, x2, paramGrid(m,n,2)));
        predictions = svmPredict(model, Xval);
        resultGrid(m,n) = mean(double(predictions ~= yval));
    end
end

[M,I] = min(resultGrid(:));
[idx1 idx2] = ind2sub(size(resultGrid),I);
C = paramGrid(idx1,idx2,1);
sigma = paramGrid(idx1,idx2,2);


% =========================================================================

end
