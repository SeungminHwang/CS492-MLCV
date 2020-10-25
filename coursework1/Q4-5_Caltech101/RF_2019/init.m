% Please run this script under the root folder

clearvars -except N;
close all;

% addpaths
addpath('./internal');
addpath('./external');
addpath('./external/libsvm-3.18/matlab');

% initialise external libraries
run('external/vlfeat-0.9.21/toolbox/vl_setup.m'); % vlfeat library
cd('external/libsvm-3.18/matlab'); % libsvm library
run('make');
cd('../../..');

