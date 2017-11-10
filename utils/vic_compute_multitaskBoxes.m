function [det_obj_act_boxes] = vic_compute_multitaskBoxes(GT_List, paths, options)

% -------------------------------------------------------------------------
% Vicky Kalogeiton, Philippe Weinzaepfel, Vittorio Ferrari, Cordelia Schmid
% email: vicky.kalogeiton@gmail.com

% If you use this software please cite our ICCV 2017 paper: 
% Joint learning of object and action detectors
% Vicky Kalogeiton, Philippe Weinzaepfel, Vittorio Ferrari, Cordelia Schmid
% In ICCV 2017

%--------------------------------------------------------------------------
% toy function that shows how to compute the detection boxes 
% for the multitask case 
%--------------------------------------------------------------------------

% Input 
%--------------------------------------------------------------------------
% paths.test_detections: the path where the detections are stored 

% GT_List:cell array (1xN) that contains the N ground truth frames 

% In A2D: c_obj = 7, c_act = 9, C = 63, V = 43 and N = 2365

if (nargin < 3) 
    options.c_obj = 7; % number of object classes
    options.c_act = 9; % number of action classes
    options.learning_case = 'multitask'; 
end
%--------------------------------------------------------------------------
% Output
%--------------------------------------------------------------------------
% det_obj_act_boxes: multitask detections: cell array (NxC), where C is the number
% of object-action classes and N is the number of frames
% Each cell is a (Kx5) single matrix, where K is the number of detections
% of the class C for the frame N, and 5 are [bbox coordinates, score]: [x1, y1, x2, y2, score]
%--------------------------------------------------------------------------

n_frames = length(GT_List); 

% evaluate both actions and objects
for ii=1:n_frames
    det = [];
    det = load([paths.test_detections GT_List{ii} '_' options.learning_case '.mat']);   
    % we assume that the det is a struct with fields: 
    % -- boxes: Kx[(c_obj+1)x4]: [bbox_c_obj1, bbox_c_obj2, ..., bbox_c_obj]
    % the regression is done per object (the first one is for background)
    % -- score: Kx[(c_obj+1)] object scores
    % -- act_score Kx[(c_act)] action scores
    % we include an example 
    C = 0; 
    for cls_obj = 1:options.c_obj       
        starting = (cls_obj)*4 + 1; 
        ending = starting + 3; 
        bboxes = [];
        bboxes = det.boxes(:, starting:ending); 
        bboxes = bboxes + 1; % boxes from python to matlab format
        for cls_act = 1:options.c_act
            C = C + 1;
            bboxes(:, 5) = det.score(:, cls_obj+1) .* det.act_score(:, cls_act);
            bboxes = single(bboxes); 
            [~, ind] = sort(bboxes(:, 5), 'descend');
            det_obj_act_boxes{ii, C} = bboxes(ind, :);   
        end      
    end   
end   

        
end