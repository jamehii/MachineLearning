function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% X has included the 1's in the front as bias term
hypothesis = X * theta;
error = hypothesis - y;
sumOfSquaredError = error' * error;

% Beware that you DO NOT want to include the bias term
noBiasTheta = theta(2:end);
sumOfSquaredTheta = noBiasTheta' * noBiasTheta;

% Calculate the J cost function
J = (1/(2*m))*(sumOfSquaredError) + (lambda/(2*m))*(sumOfSquaredTheta);

% Calcualte gradient
for i = 1 : size(theta,1)
  
  if i == 1
    % Gradient for bias term. No regularization term
    grad(i) = (1/m)* sum(error .* X(:,i));
  else
    grad(i) = (1/m)* sum(error .* X(:,i)) + (lambda / m)*theta(i)
  endif
  
endfor

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
