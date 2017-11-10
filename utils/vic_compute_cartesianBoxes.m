function det_obj_act_boxes = vic_compute_cartesianBoxes(GT_List, paths, options)

% -------------------------------------------------------------------------
% Vicky Kalogeiton, Philippe Weinzaepfel, Vittorio Ferrari, Cordelia Schmid
% email: vicky.kalogeiton@gmail.com

% If you use this software please cite our ICCV 2017 paper: 
% Joint learning of object and action detectors
% Vicky Kalogeiton, Philippe Weinzaepfel, Vittorio Ferrari, Cordelia Schmid
% In ICCV 2017

%--------------------------------------------------------------------------
% toy function that shows how to compute the detection boxes 
% for the cartesian case 

% Input 
%--------------------------------------------------------------------------
% paths.test_detections: the path where the detections are stored 

% GT_List:cell array (1xN) that contains the N ground truth frames 

% In A2D: c_obj = 7, c_act = 9, C = 63, V = 43 and N = 2365

if (nargin < 3) 
    options.c_obj = 7; % number of object classes
    options.c_act = 9; % number of action classes
    options.learning_case = 'cartesian'; 
end
%--------------------------------------------------------------------------
% Output
%--------------------------------------------------------------------------
% det_obj_act_boxes: multitask detections: cell array (NxV), where V is the number
% of valid object-action and N is the number of frames
% Each cell is a (Kx5) single matrix, where K is the number of detections
% of the class C for the frame N, and 5 are [bbox coordinates, score]: [x1, y1, x2, y2, score]
%--------------------------------------------------------------------------

n_frames = length(GT_List); 

% evaluate both actions and objects
for ii=1:n_frames
    det = [];
    det = load([paths.test_detections GT_List{ii} '_' options.learning_case '.mat']);  
    % we assume that the det is a struct with fields: 
    % -- boxes: Kx[(V+1)x4]: [bbox_c_obj1, bbox_c_obj2, ..., bbox_c_obj]
    % the regression is done per object-action pair (the first one is for background)
    % -- score: Kx[(V+1)] object-action pair scores
    % we include an example 
    for cls = 1:options.num_valid
        bboxes = [];
        starting = (cls) * 4 + 1; 
        ending = starting + 3; 
        bboxes = det.boxes(:, starting:ending); 
        bboxes = bboxes + 1; % boxes from python to matlab format
        bboxes(:, 5) = det.score(:, cls+1);
        bboxes = single(bboxes); 
        [~, ind] = sort(bboxes(:, 5),'descend');
        det_obj_act_boxes{ii, cls} = bboxes(ind, :);   
    end    
end   

end