function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

numOfColumns = size(X, 2);
numOfRows = size(X, 1);

meanFeatures = zeros(numOfColumns, 1);

for column = 1:numOfColumns
 
    total = sum(X(:,column));
    mu(1,column) = total / numOfRows;
    sigma(1,column) = std(X(:, column));
    
    for row = 1:numOfRows
        X_norm(row, column) = (X_norm(row) - mu(column)) / sigma(1,column);
    endfor
    
endfor


% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       









% ============================================================

end
