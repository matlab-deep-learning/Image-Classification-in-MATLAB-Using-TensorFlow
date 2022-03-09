%% There can be library conflicts if Python is loaded 'InProcess'
% On starting the Python the first time, it's recommended to use
% 'OutOfProcess' to avoid potential library conflicts

pyenv(ExecutionMode="OutOfProcess");

% To increase performance try restarting MATLAB and launching Python
% 'InProcess'. If a library conflict occurs, revert back to 'OutOfProcess'
% pyenv('ExecutionMode', 'InProcess');

